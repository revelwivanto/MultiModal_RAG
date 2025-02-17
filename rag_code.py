import os
import torch
from torch.amp import autocast
from qdrant_client import models
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer
import base64
from io import BytesIO
from tqdm import tqdm
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Optimize CUDA memory handling if GPU is available
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
    except Exception as e:
        print(f"Warning: Unable to set memory fraction: {e}")

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def image_to_base64(image):
    """Convert a PIL image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class EmbedData:
    def __init__(self, embed_model_name="vidore/colpali-v1.2", batch_size=2, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.embed_model_name = embed_model_name
        self.device = device
        self.embed_model, self.processor = self._load_embed_model()
        self.batch_size = batch_size
        self.embeddings = []

    def _load_embed_model(self):
        embed_model = ColPali.from_pretrained(
            self.embed_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir="./Janus/hf_cache"
        ).to(self.device)
        processor = ColPaliProcessor.from_pretrained(self.embed_model_name)
        return embed_model, processor
    
    def get_query_embedding(self, query):
        with torch.no_grad():
            processed_query = self.processor.process_queries([query]).to(self.device)
            query_embedding = self.embed_model(**processed_query).cpu().float().numpy().tolist()
        torch.cuda.empty_cache()
        return query_embedding[0]

    def generate_embedding(self, images):
        with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
            batch_images = self.processor.process_images(images).to(self.device)
            image_embeddings = self.embed_model(**batch_images).cpu().float().numpy().tolist()
        torch.cuda.empty_cache()
        return image_embeddings

    def embed(self, images):
        self.images = images
        self.all_embeddings = []
        for image in tqdm(images, desc="Generating embeddings"):
            batch_embeddings = self.generate_embedding([image])
            self.embeddings.extend(batch_embeddings)
            del batch_embeddings
            torch.cuda.empty_cache()

class QdrantVDB_QB:
    def __init__(self, collection_name, vector_dim=128, batch_size=1):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        self.client = QdrantClient(
            url="https://f1d8cf59-63e8-42e7-be31-5fce0d82a5db.europe-west3-0.gcp.cloud.qdrant.io",
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2NTA0MzY3fQ.dc7I7oq2o3HU11RNmRvOW6XXxj3rmxN3kYnaZRVoEkM",
            prefer_grpc=True
        )

    def create_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                on_disk_payload=True,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )

    def ingest_data(self, embeddata):
        for i, batch_embeddings in tqdm(enumerate(batch_iterate(embeddata.embeddings, self.batch_size)), desc="Ingesting data"):
            points = []
            for j, embedding in enumerate(batch_embeddings):
                image_bs64 = image_to_base64(embeddata.images[i * self.batch_size + j])
                current_point = models.PointStruct(
                    id=i * self.batch_size + j, vector=embedding, payload={"image": image_bs64}
                )
                points.append(current_point)
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            torch.cuda.empty_cache()

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        query_embedding = self.embeddata.get_query_embedding(query)
        query_result = self.vector_db.client.query_points(
            collection_name=self.vector_db.collection_name,
            query=query_embedding,
            limit=4,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True, rescore=True, oversampling=2.0
                )
            )
        )
        return query_result

class RAG:
    def __init__(self, retriever, llm_name="deepseek-ai/Janus-Pro-1B"):
        self.llm_name = llm_name
        self.device = retriever.embeddata.device
        self._setup_llm()
        self.retriever = retriever
        self.template = None
        self.template_placeholders = []

    def _setup_llm(self):
        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.llm_name, cache_dir="./Janus/hf_cache")
        self.tokenizer = self.vl_chat_processor.tokenizer

        # Initialize the model with empty weights
        with init_empty_weights():
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                self.llm_name, 
                torch_dtype=torch.float16,
                trust_remote_code=True, 
                cache_dir="./Janus/hf_cache",
            )

        # Load the model checkpoint and dispatch it to available devices
        self.vl_gpt = load_checkpoint_and_dispatch(
            self.vl_gpt,
            checkpoint=self.llm_name,
            device_map="auto",  # Automatically offloads parts of the model
            offload_folder="./offload",  # Directory for offloading
            no_split_module_classes=["ModuleName"],  # Specify modules to keep together
        ).eval()

    def set_template(self, template_text):
        """Set the template and extract placeholders."""
        self.template = template_text
        # Extract all text within square brackets
        self.template_placeholders = re.findall(r'\[(.*?)\]', template_text)

    def generate_context(self, query):
        result = self.retriever.search(query)
        if not result.points:
            return "No relevant images found."
        # Retrieve the image from the payload
        image_bs64 = result.points[0].payload["image"]
        return image_bs64

    def process_template(self, context_text):
        """Process the template with context text."""
        if not self.template:
            return context_text
        
        processed_template = self.template
        # Replace each placeholder with corresponding context text
        for placeholder in self.template_placeholders:
            processed_template = processed_template.replace(f'[{placeholder}]', context_text)
        
        return processed_template

    def query(self, query):
        image_context = self.generate_context(query=query)
        
        # Prepare the prompt with the image and query
        qa_prompt_tmpl_str = f"""The user has asked the following question:

                        ---------------------
                        
                        Query: {query}
                        
                        ---------------------

                        Some images are available to you
                        for this question. You have
                        to understand these images thoroughly and 
                        extract all relevant information that will 
                        help you answer the query.
                                     
                        ---------------------
                        """
                        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder> \n {qa_prompt_tmpl_str}",
                "images": [image_context],  # Pass the image context directly
            },
            {"role": "Assistant", "content": ""},
        ]
        
        # Load the images and process them with Janus
        pil_images = load_pil_images(conversation)
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation, 
                images=pil_images, 
                force_batchify=True
            ).to(self.vl_gpt.device)
            
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            )
        
        # Decode the generated response
        response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        # If there's a template, process the response through it
        if self.template:
            response = self.process_template(response)
            
        torch.cuda.empty_cache()
        return response

    def feedback(self, query, user_feedback):
        # Store feedback for analysis
        logging.info(f"Feedback received for query '{query}': {user_feedback}")
        # You can implement logic to store this feedback in a database or file

# Load the model and tokenizer from the Hugging Face Model Hub
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/Janus-Pro-1B")
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/Janus-Pro-1B")