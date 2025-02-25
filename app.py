# Adapted from https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps#build-a-simple-chatbot-gui-with-streaming
import os
import base64
import gc
import random
import tempfile
import time
import uuid
import platform

from IPython.display import Markdown, display

import streamlit as st

import torch
import time
import numpy as np
from tqdm import tqdm
from pdf2image import convert_from_path

from rag_code import EmbedData, QdrantVDB_QB, Retriever, RAG

# Add device selection logic
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

collection_name = "multimodal_rag_with_deepseek-new"

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
query_engine = None  # Initialize as None to prevent NameError

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()

def display_pdf(file):
    # Opening file from file path
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%">
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header("Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):
                    # Determine poppler path based on OS.
                    if platform.system() == "Windows":
                        poppler_path = r'C:\Release-24.08.0-0\poppler-24.08.0\Library\bin'
                    else:
                        poppler_path = None  # On Colab/Linux, poppler is installed via apt-get

                    # Convert PDF to images using pdf2image
                    images = convert_from_path(file_path, poppler_path=poppler_path)

                    # Ensure the 'images' directory exists before saving the images
                    images_dir = "/content/images"
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)

                    for i in range(len(images)):
                        # Save pages as images in the pdf
                        image_path = os.path.join(images_dir, f'page{i}.jpg')
                        images[i].save(image_path, 'JPEG')

                    # Embed data    
                    embeddata = EmbedData(device=get_device())
                    embeddata.embed(images)

                    # Set up vector database
                    qdrant_vdb = QdrantVDB_QB(collection_name=collection_name, vector_dim=128)
                    qdrant_vdb.define_client()
                    qdrant_vdb.create_collection()
                    qdrant_vdb.ingest_data(embeddata=embeddata)

                    # Set up retriever
                    retriever = Retriever(vector_db=qdrant_vdb, embeddata=embeddata)

                    # Set up RAG
                    query_engine = RAG(retriever=retriever)

                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.markdown(
        """
        # Multimodal RAG powered by <img src="data:image/png;base64,{}" width="170" style="vertical-align: -3px;"> Janus
        """.format(
            base64.b64encode(open("assets/bnu.png", "rb").read()).decode()
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        streaming_response = query_engine.query(prompt)
        for chunk in streaming_response:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.01)
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
