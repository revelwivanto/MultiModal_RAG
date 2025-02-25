import os
import torch
from torch.amp import autocast
from qdrant_client import models
from qdrant_client import QdrantClient
from colpali_engine.models import ColPali, ColPaliProcessor
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from transformers import AutoModelForCausalLM
import base64
from io import BytesIO
from tqdm import tqdm
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_BATCH_SIZE = 4
DEFAULT_VECTOR_DIM = 128
DEFAULT_EMBED_MODEL_NAME = "vidore/colpali-v1.2"
DEFAULT_LLM_NAME = "deepseek-ai/Janus-Pro-1B"
CACHE_DIR = "./Janus/hf_cache"
DEFAULT_COLLECTION_NAME = "default_collection"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_NEW_TOKENS = 512

# Defensive Programming: Ensure CUDA settings are correctly handled
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        torch.cuda.set_per_process_memory_fraction(0.8, device=0)
    except RuntimeError as e:
        logger.warning(f"Unable to set memory fraction: {e}")

# Set the default dtype for model loading
default_dtype = torch.float16

def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from lst."""
    if not isinstance(lst, list):
        raise ValueError("Expected a list for batch iteration.")
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0.")
    
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

def image_to_base64(image):
    """Convert a PIL image to a base64-encoded string."""
    if image is None:
        logger.error("Attempted to convert a None image to base64.")
        return ""
    
    try:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""

def convert_dict_to_half(d):
    """
    Recursively convert all torch.Tensor values in a dictionary (or list)
    to torch.float16.
    """
    if isinstance(d, dict):
        return {k: convert_dict_to_half(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_dict_to_half(v) for v in d]
    elif isinstance(d, torch.Tensor):
        return d.to(torch.float16)
    else:
        return d

class EmbedData:
    def __init__(self, embed_model_name=DEFAULT_EMBED_MODEL_NAME, batch_size=DEFAULT_BATCH_SIZE, device=DEFAULT_DEVICE):
        """Handles image embeddings using ColPali."""
        self.embed_model_name = embed_model_name
        self.device = device
        self.batch_size = batch_size
        self.embeddings = []
        self.images = []
        
        try:
            self.embed_model, self.processor = self._load_embed_model()
        except Exception as e:
            logger.critical(f"Failed to load embed model: {e}")
            raise SystemExit("Embedding model initialization failed. Exiting.")

    def _load_embed_model(self):
        """Loads the embedding model safely."""
        if not isinstance(self.embed_model_name, str) or not self.embed_model_name:
            raise ValueError("Invalid model name for embeddings.")
        
        try:
            embed_model = ColPali.from_pretrained(
                self.embed_model_name,
                torch_dtype=default_dtype,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            ).to(self.device)
            processor = ColPaliProcessor.from_pretrained(self.embed_model_name)
            return embed_model, processor
        except RuntimeError as e:
            logger.critical(f"Runtime error loading embedding model: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading embedding model: {e}")
            raise

    def get_query_embedding(self, query):
        """Generates an embedding for a given query."""
        if not isinstance(query, str) or not query.strip():
            logger.error("Invalid query input for embedding.")
            return None
        
        try:
            with torch.no_grad():
                processed_query = self.processor.process_queries([query]).to(self.device)
                query_embedding = self.embed_model(**processed_query).cpu().float().numpy().tolist()
            torch.cuda.empty_cache()
            return query_embedding[0]
        except RuntimeError as e:
            logger.error(f"Runtime error generating query embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during query embedding: {e}")
            return None

    def generate_embedding(self, images):
        try:
            with torch.no_grad(), autocast(device_type='cuda', dtype=default_dtype):
                batch_images = self.processor.process_images(images).to(self.device)
                image_embeddings = self.embed_model(**batch_images).cpu().float().numpy().tolist()
            torch.cuda.empty_cache()
            return image_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def embed(self, images):
        self.images = images
        self.all_embeddings = []
        for image in tqdm(images, desc="Generating embeddings"):
            try:
                batch_embeddings = self.generate_embedding([image])
                self.embeddings.extend(batch_embeddings)
                del batch_embeddings
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error processing image for embeddings: {e}")

class QdrantVDB_QB:
    def __init__(self, collection_name=DEFAULT_COLLECTION_NAME, vector_dim=DEFAULT_VECTOR_DIM, batch_size=1):
        """Handles interactions with the Qdrant vector database."""
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("Collection name must be a non-empty string.")
        if vector_dim <= 0:
            raise ValueError("Vector dimension must be positive.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero.")
        
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.client = None
        
        try:
            self.define_client()
        except Exception as e:
            logger.critical(f"Failed to initialize Qdrant client: {e}")
            raise SystemExit("Vector database initialization failed. Exiting.")

    def define_client(self):
        """Connects to Qdrant server."""
        try:
            self.client = QdrantClient(
                url="https://f1d8cf59-63e8-42e7-be31-5fce0d82a5db.europe-west3-0.gcp.cloud.qdrant.io",
                api_key="API_KEY_HERE",
                prefer_grpc=True
            )
        except Exception as e:
            logger.error(f"Error connecting to Qdrant client: {e}")
            raise

    def create_collection(self):
        """Creates a collection in Qdrant if it does not exist."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,
                        distance=models.Distance.COSINE,
                        on_disk=True
                    )
                )
        except Exception as e:
            logger.error(f"Error creating collection: {e}")

    def ingest_data(self, embeddata):
        for i, batch_embeddings in tqdm(enumerate(batch_iterate(embeddata.embeddings, self.batch_size)), desc="Ingesting data"):
            points = []
            for j, embedding in enumerate(batch_embeddings):
                try:
                    image_index = i * self.batch_size + j
                    image_bs64 = image_to_base64(embeddata.images[image_index])
                    current_point = models.PointStruct(
                        id=image_index, vector=embedding, payload={"image": image_bs64}
                    )
                    points.append(current_point)
                except Exception as e:
                    logger.error(f"Error processing embedding at index {i * self.batch_size + j}: {e}")
            try:
                self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            except Exception as e:
                logger.error(f"Error upserting points into collection: {e}")
            torch.cuda.empty_cache()

class Retriever:
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        try:
            query_embedding = self.embeddata.get_query_embedding(query)
            if query_embedding is None:
                raise ValueError("Query embedding is None.")
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
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return None

class RAG:
    def __init__(self, retriever, llm_name=DEFAULT_LLM_NAME):
        self.llm_name = llm_name
        self.device = retriever.embeddata.device
        self.retriever = retriever
        try:
            self._setup_llm()
        except Exception as e:
            logger.error(f"Error setting up LLM: {e}")
            raise

    def _setup_llm(self):
        try:
            self.vl_chat_processor = VLChatProcessor.from_pretrained(self.llm_name, cache_dir=CACHE_DIR)
            self.tokenizer = self.vl_chat_processor.tokenizer
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                self.llm_name, 
                torch_dtype=default_dtype,
                trust_remote_code=True, 
                cache_dir=CACHE_DIR
            ).eval().to(self.device)
        except Exception as e:
            logger.error(f"Error loading VLChatProcessor or LLM: {e}")
            raise

    def generate_context(self, query):
        try:
            result = self.retriever.search(query)
            if result is None or not result.points:
                logger.warning("No points found in search results. Falling back to default image.")
                return "/content/images/page0.jpg"
            image_path = f"/content/images/page{result.points[0].id}.jpg"
            if not os.path.exists(image_path):
                logger.warning(f"{image_path} not found. Defaulting to /content/images/page0.jpg")
                image_path = "/content/images/page0.jpg"
            return image_path
        except Exception as e:
            logger.error(f"Error generating context from query: {e}")
            return "/content/images/page0.jpg"

    def query(self, query):
        try:
            image_context = self.generate_context(query=query)
            
            qa_prompt_tmpl_str = f"""The user has asked the following question:

                ---------------------
                Query: {query}
                ---------------------

                Some images are available to you for this question. You have to understand these images thoroughly and 
                extract all relevant information, especially the text present in these images, as it will help you answer the query.
                ---------------------"""
            
            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder> \n {qa_prompt_tmpl_str}",
                    "images": [image_context],
                },
                {"role": "Assistant", "content": ""},
            ]
            
            pil_images = load_pil_images(conversation)
            with torch.cuda.amp.autocast(dtype=default_dtype):
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
                  max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                  do_sample=True,  # Enable sampling for diversity
                  temperature=0.7,  # Control randomness (lower = more deterministic)
                  top_k=50,  # Keep top 50 tokens per step
                  top_p=0.9,  # Nucleus sampling for diverse outputs
                  repetition_penalty=1.2,  # Penalize repetition
                  no_repeat_ngram_size=3,  # Prevents 3-gram repetition
                  use_cache=True,
              )

            
            streaming_response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            torch.cuda.empty_cache()
            return streaming_response
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return "An error occurred during processing."
