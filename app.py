"""
AI Legal Explainer: A RAG-based Streamlit Application
======================================================

This script implements a Retrieval-Augmented Generation (RAG) system designed to
explain complex legal documents in plain language. It features a Streamlit web
interface where users can upload PDF documents and ask questions about them.

The application is built with configurability in mind, supporting multiple
embedding dimensions for vectorization.

Core Functionalities:
---------------------
1.  **Document Processing:**
    -   Handles PDF uploads through a Streamlit interface.
    -   Utilizes Google Cloud Document AI for robust text extraction from PDFs.
    -   Chunks extracted text into smaller, overlapping segments for effective retrieval.
    -   Generates vector embeddings using SentenceTransformer models.

2.  **Vector Storage and Retrieval:**
    -   Uses Pinecone as the vector database for storing and querying document embeddings.
    -   Supports configurable embedding dimensions (e.g., 384, 1024).
    -   Performs cosine similarity searches to find relevant document chunks based on user queries.

3.  **Generative AI and Language Support:**
    -   Leverages Google's Gemini 1.5 Flash model for response generation.
    -   Implements a multi-language agent that detects the user's query language and responds accordingly.
    -   Maintains conversation history for contextual responses.

4.  **User Interface and Experience:**
    -   Built with Streamlit for a clean and interactive web application.
    -   Features session management, chat history, and document management.
    -   Provides options to export conversations and clear session memory.

This file is now fully commented to improve readability and maintainability.
"""

import streamlit as st
import google.generativeai as genai
from google.cloud import documentai
from google.api_core.client_options import ClientOptions
import langdetect
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import hashlib
import uuid
from io import BytesIO
from datetime import datetime 
import json
import time
import os
from dotenv import load_dotenv

# ===========================
# CONFIGURATION SECTION
# ===========================
# Load API keys from the .env file for secure access to services.
load_dotenv()

# Retrieve API keys and configuration details from environment variables.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DOCAI_PROJECT_ID = os.getenv("DOCAI_PROJECT_ID")
DOCAI_LOCATION = os.getenv("DOCAI_LOCATION")
DOCAI_PROCESSOR_ID = os.getenv("DOCAI_PROCESSOR_ID")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-west-2")  # Default to us-west-2 if not set

# ===========================
# EMBEDDING CONFIGURATION
# ===========================
# This section allows you to choose the sentence transformer model for generating embeddings.
# The choice of model affects the dimensionality of the embeddings, which in turn impacts
# storage requirements and retrieval performance. Different models offer trade-offs between
# speed, size, and quality.

# Option 1: 384-dimensional multilingual embeddings (Recommended for broad language support).
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2' 
EMBEDDING_DIMENSIONS = 384

# Option 2: 384-dimensional English-focused embeddings (Faster, for English-only documents).
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# EMBEDDING_DIMENSIONS = 384

# Option 3: 768-dimensional high-quality embeddings (Larger, slower, primarily for English).
# EMBEDDING_MODEL = 'all-mpnet-base-v2'
# EMBEDDING_DIMENSIONS = 768

# Initialize the Google Gemini API.
# This step configures the generative AI library with the provided API key.
# It's crucial for enabling the AI's text generation capabilities.
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # If the API key is not found, display an error and stop the application.
    st.error("❌ Google API key not found. Please check your .env file.")
    st.stop()

# ===========================
# STREAMLIT PAGE CONFIGURATION
# ===========================
# Set up the Streamlit page with a custom title, a wide layout for more space,
# a relevant icon, and an initially expanded sidebar for easy access to controls.
st.set_page_config(
    page_title="⚖️ AI Legal Explainer",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

# Display the current embedding model configuration in the sidebar.
# This provides users with transparency about the system's setup.
st.sidebar.header("📊 Current Configuration")
st.sidebar.info(f"""
**Embedding Model:** {EMBEDDING_MODEL}
**Dimensions:** {EMBEDDING_DIMENSIONS}
**Index Name:** rag-documents-{EMBEDDING_DIMENSIONS}d
""")

# ===========================
# DOCUMENT PROCESSOR CLASS
# ===========================
class DocumentProcessor:
    """
    Handles the entire pipeline of processing PDF documents, from text extraction to embedding generation.
    This class encapsulates the logic for:
    - Reading uploaded PDF files.
    - Using Google Cloud Document AI to extract text content accurately.
    - Splitting the extracted text into manageable, overlapping chunks.
    - Generating vector embeddings for each chunk using a SentenceTransformer model.
    - Caching extracted text to avoid redundant processing of the same document.
    """
    
    def __init__(self, model_name=EMBEDDING_MODEL):
        """
        Initializes the DocumentProcessor.

        Args:
            model_name (str): The name of the SentenceTransformer model to use for creating embeddings.
        """
        st.write(f"🔄 Initializing Document Processor with {model_name}...")
        # Load the specified pre-trained sentence transformer model.
        self.model = SentenceTransformer(model_name)
        # A simple in-memory cache to store extracted text, using the file's hash as the key.
        self.memory_cache = {}
        # Store the embedding dimensions based on the global configuration.
        self.dimensions = EMBEDDING_DIMENSIONS
        st.write(f"✅ Document Processor ready! ({self.dimensions}D embeddings)")
    
    def extract_text_from_pdf(self, pdf_file):
        """
        Extracts text from an uploaded PDF file using Google Cloud Document AI.

        This method first checks if the document has been processed before by comparing its hash.
        If it's a new document, it calls the Document AI API to perform OCR and text extraction.

        Args:
            pdf_file: A file-like object representing the uploaded PDF.

        Returns:
            str: The extracted text content of the PDF. Returns an empty string on failure.
        """
        try:
            file_content = pdf_file.read()
            # Generate a unique hash for the file content to use as a cache key.
            file_hash = hashlib.md5(file_content).hexdigest()
            pdf_file.seek(0)  # Reset file pointer for subsequent use.

            # Check the cache first to avoid reprocessing.
            if file_hash in self.memory_cache:
                st.info("📄 Found document in cache. Skipping extraction.")
                return self.memory_cache[file_hash]

            st.write("🤖 Calling Google Cloud Document AI for text extraction...")
            
            # Configure the Document AI client with the specified location.
            opts = ClientOptions(api_endpoint=f"{DOCAI_LOCATION}-documentai.googleapis.com")
            client = documentai.DocumentProcessorServiceClient(client_options=opts)
            # Construct the full resource name of the processor.
            name = client.processor_path(DOCAI_PROJECT_ID, DOCAI_LOCATION, DOCAI_PROCESSOR_ID)

            # Prepare the document for processing.
            raw_document = documentai.RawDocument(content=file_content, mime_type="application/pdf")
            request = documentai.ProcessRequest(name=name, raw_document=raw_document)

            # Send the request to Document AI and get the result.
            result = client.process_document(request=request)
            text = result.document.text

            # Handle cases where no text is extracted.
            if not text.strip():
                st.warning("⚠️ Document AI did not extract any text from this PDF.")
                return ""

            # Cache the extracted text for future use.
            self.memory_cache[file_hash] = text
            st.success(f"✅ Extracted {len(text)} characters using Document AI.")
            return text
        except Exception as e:
            st.error(f"❌ Failed to process document with Google Document AI: {str(e)}")
            st.error("Please ensure your GCP project, location, processor ID, and authentication are correctly configured.")
            return ""
    
    def chunk_text(self, text, chunk_size=1000, overlap=200):
        """
        Splits a large text into smaller, overlapping chunks.

        Chunking is essential for RAG systems as it allows the model to process and retrieve
        more focused and relevant pieces of information. Overlapping ensures that context is not
        lost at the boundaries of chunks.

        Args:
            text (str): The input text to be chunked.
            chunk_size (int): The desired character length of each chunk.
            overlap (int): The number of characters to overlap between consecutive chunks.

        Returns:
            list: A list of text chunks.
        """
        st.write("✂️ Splitting text into chunks...")
        chunks = []
        start = 0
        
        # Iterate through the text and create chunks.
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            # Move the start position back by the overlap amount to create overlapping chunks.
            start = end - overlap
        
        st.success(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def create_embeddings(self, texts):
        """
        Converts a list of text chunks into their corresponding vector embeddings.

        Embeddings are dense numerical vectors that capture the semantic meaning of the text.
        These are crucial for performing similarity searches in the vector database.

        Args:
            texts (list): A list of text strings to be embedded.

        Returns:
            list: A list of embedding vectors. Returns an empty list on failure.
        """
        try:
            if not texts:
                st.warning("⚠️ No text provided for embedding creation")
                return []
            
            # Ensure that we only process non-empty text chunks.
            valid_texts = [text for text in texts if text.strip()]
            if len(valid_texts) != len(texts):
                st.warning(f"⚠️ Filtered out {len(texts) - len(valid_texts)} empty text chunks")
            
            if not valid_texts:
                st.error("❌ No valid text chunks to process")
                return []
            
            st.write(f"🔢 Creating {self.dimensions}D embeddings for {len(valid_texts)} chunks...")
            with st.spinner("Generating embeddings..."):
                # Encode the texts into embeddings using the pre-loaded model.
                embeddings = self.model.encode(valid_texts, show_progress_bar=False).tolist()
            st.success(f"✅ Created {len(embeddings)} {self.dimensions}D embeddings")
            return embeddings
        except Exception as e:
            st.error(f"❌ Failed to create embeddings: {str(e)}")
            return []

# ===========================
# PINECONE HANDLER CLASS
# ===========================
class PineconeHandler:
    """
    Manages all interactions with the Pinecone vector database.
    This class is responsible for:
    - Establishing a connection to Pinecone.
    - Ensuring the required index exists, creating it if necessary.
    - Upserting (uploading) document embeddings and their associated metadata.
    - Querying the index to find the most relevant document chunks for a given user query.
    """
    
    def __init__(self, dimensions=EMBEDDING_DIMENSIONS):
        """
        Initializes the PineconeHandler.

        Args:
            dimensions (int): The dimensionality of the embeddings to be stored in the index.
        """
        if not PINECONE_API_KEY:
            st.error("❌ Pinecone API key not found. Please check your .env file.")
            st.stop()
            
        st.write("🔄 Connecting to Pinecone...")
        
        try:
            # Initialize the Pinecone client with the API key.
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.dimensions = dimensions
            # Define a unique name for the index based on the embedding dimensions.
            self.index_name = f"rag-documents-{dimensions}d"
            # Ensure that the index exists before proceeding.
            self._ensure_index_exists()
            st.write(f"✅ Pinecone connection established! (Index: {self.index_name})")
        except Exception as e:
            st.error(f"❌ Failed to connect to Pinecone: {str(e)}")
            st.error("Please check your Pinecone API key and try again.")
            st.stop()
        
    def _ensure_index_exists(self):
        """
        Checks if the target Pinecone index exists, and creates it if it does not.
        This ensures that we can always connect to a valid index.
        """
        try:
            # Get a list of all existing index names.
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            # If the index does not exist, create it.
            if self.index_name not in existing_indexes:
                st.write(f"🏗️ Creating new Pinecone index with {self.dimensions} dimensions...")
                
                # Define the index configuration.
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimensions,
                    metric="cosine",  # Use cosine similarity for semantic search.
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=PINECONE_REGION
                    )
                )
                # Wait for the index to be fully provisioned by checking its status.
                st.write("⏳ Waiting for Pinecone index to be ready...")
                wait_time = 0
                max_wait_time = 60  # Maximum wait time of 60 seconds
                while wait_time < max_wait_time:
                    try:
                        index_description = self.pc.describe_index(self.index_name)
                        if index_description.status.state == 'Ready':
                            st.write(f"✅ Pinecone index is ready after {wait_time} seconds!")
                            break
                        else:
                            time.sleep(5)
                            wait_time += 5
                            st.write(f"⏳ Still waiting... ({wait_time}s elapsed)")
                    except Exception as e:
                        # If there's an error checking status, wait and try again
                        time.sleep(5)
                        wait_time += 5
                if wait_time >= max_wait_time:
                    st.warning("⚠️ Index creation is taking longer than expected. Proceeding anyway.")
                st.success(f"✅ Pinecone index created! ({self.dimensions}D)")
            else:
                st.write(f"✅ Using existing Pinecone index ({self.dimensions}D)")
            
            # Get a handle to the index object for performing operations.
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            st.error(f"❌ Failed to create/connect to Pinecone index: {str(e)}")
            raise
    
    def upsert_vectors(self, vectors, texts, file_name):
        """
        Uploads vectors and their associated metadata to the Pinecone index.

        Args:
            vectors (list): A list of embedding vectors.
            texts (list): The corresponding text chunks for each vector.
            file_name (str): The name of the source document.
        """
        try:
            st.write(f"💾 Storing {len(vectors)} vectors ({self.dimensions}D) in Pinecone...")
            
            # Structure the data in the format required by Pinecone.
            vectors_to_upsert = [
                {
                    "id": f"{hashlib.md5(f'{file_name}_{i}'.encode()).hexdigest()}",  # Create a unique ID for each vector.
                    "values": vector,
                    "metadata": {
                        "text": text[:1000],  # Store the text chunk (truncated to avoid size limits).
                        "source": file_name,
                        "timestamp": datetime.now().isoformat(),
                        "dimensions": self.dimensions
                    }
                }
                for i, (vector, text) in enumerate(zip(vectors, texts))
            ]
            
            # Upsert the vectors in batches to avoid overwhelming the API.
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
                
            st.success(f"✅ Stored {len(vectors_to_upsert)} vectors in Pinecone")
        except Exception as e:
            st.error(f"❌ Failed to store vectors: {str(e)}")
            raise
    
    def query_vectors(self, query_vector, top_k=3, filter_source=None):
        """
        Queries the Pinecone index to find vectors similar to the query vector.

        Args:
            query_vector (list): The embedding vector of the user's query.
            top_k (int): The number of top similar results to retrieve.
            filter_source (str, optional): If provided, filters the search to a specific source document.

        Returns:
            A Pinecone query result object containing the matched vectors and their metadata.
        """
        try:
            st.write("🔍 Searching for relevant documents...")
            
            # Create a metadata filter if a specific source document is selected.
            query_filter = None
            if filter_source:
                query_filter = {"source": {"$eq": filter_source}}
                st.info(f"📄 Filtering results for document: {filter_source}")
            else:
                st.info("🌐 Searching across all documents")
            
            # Define the parameters for the Pinecone query.
            query_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True  # We need the metadata to get the original text.
            }
            
            # Add the filter to the query if it exists.
            if query_filter is not None:
                query_params["filter"] = query_filter
            
            # Execute the query.
            results = self.index.query(**query_params)
            
            st.success(f"✅ Found {len(results.matches)} relevant matches")
            return results
        except Exception as e:
            st.error(f"❌ Failed to search vectors: {str(e)}")
            # Return a mock empty result object to prevent the app from crashing on error.
            from types import SimpleNamespace
            return SimpleNamespace(matches=[])

# ===========================
# GEMINI HANDLER CLASS
# ===========================
class GeminiHandler:
    """
    Handles interactions with the Google Gemini large language model.
    This class is responsible for:
    - Initializing the generative model.
    - Managing conversational history for context.
    - Sending prompts to the model and receiving generated responses.
    """
    
    def __init__(self):
        """
        Initializes the GeminiHandler.
        """
        st.write("🔄 Initializing Gemini AI...")
        # Use the 'gemini-1.5-flash' model, which is optimized for speed and efficiency.
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        # A dictionary to store ongoing chat sessions, enabling conversation memory.
        self.chat_sessions = {}
        st.write("✅ Gemini AI ready!")
    
    def get_response(self, prompt, session_id=None):
        """
        Generates a response from the Gemini model, maintaining conversational context.

        If a session_id is provided, it retrieves the existing chat history for that session
        to provide a more context-aware response.

        Args:
            prompt (str): The user's input prompt.
            session_id (str, optional): A unique identifier for the conversation session.

        Returns:
            str: The text response generated by the Gemini model.
        """
        try:
            # If a session exists, continue the conversation.
            if session_id and session_id in self.chat_sessions:
                chat = self.chat_sessions[session_id]
                response = chat.send_message(prompt)
            # Otherwise, start a new conversation.
            else:
                chat = self.model.start_chat(history=[])
                response = chat.send_message(prompt)
                # Store the new chat session if a session_id is provided.
                if session_id:
                    self.chat_sessions[session_id] = chat
            
            return response.text
        except Exception as e:
            # Provide a user-friendly error message if the API call fails.
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return error_msg


# ===========================
# INITIALIZATION FUNCTION
# ===========================
@st.cache_resource
def initialize_system():
    """
    Initializes and caches all the major components of the RAG system.

    The `@st.cache_resource` decorator ensures that these components are created only once
    per session, which significantly improves performance by avoiding re-initialization on
    every user interaction.

    Returns:
        tuple: A tuple containing the initialized doc_processor, pinecone_handler, and gemini_handler.
    """
    st.write("🚀 Initializing RAG System...")
    
    # Use a progress bar to provide visual feedback during initialization.
    progress_bar = st.progress(0)
    
    # Initialize the document processor.
    progress_bar.progress(25)
    doc_processor = DocumentProcessor(EMBEDDING_MODEL)
    
    # Initialize the Pinecone handler.
    progress_bar.progress(50)
    pinecone_handler = PineconeHandler(EMBEDDING_DIMENSIONS)
    
    # Initialize the Gemini handler.
    progress_bar.progress(75)
    gemini_handler = GeminiHandler()
    
    progress_bar.progress(100)
    
    # Clean up the UI by removing the progress bar.
    progress_bar.empty()
    st.success("✅ All systems initialized!")
    
    return doc_processor, pinecone_handler, gemini_handler

# ===========================
# SESSION STATE INITIALIZATION
# ===========================
# Streamlit's session state is used to persist variables across user interactions.
# This is crucial for maintaining the chat history, user session, and other stateful information.

# Initialize the chat history list if it doesn't already exist.
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Assign a unique session ID to each user session for conversation tracking.
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Keep track of documents that have already been processed to avoid redundant work.
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = {}

# Add a key for the file uploader to allow for programmatic clearing.
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# ===========================
# MAIN UI LAYOUT
# ===========================
st.title("⚖️ AI Legal Explainer")
st.markdown("Upload a legal document (e.g., rental agreement, contract, terms of service) and ask questions to get plain-language explanations.")
st.warning("**Disclaimer:** This is an AI tool and not a substitute for legal advice. For binding legal decisions, please consult a licensed attorney.")

col1, col2 = st.columns([2, 3])

doc_processor, pinecone_handler, gemini_handler = initialize_system()

# --- Column 1: Document Management ---
with col1:
    st.header("📄 Document Manager")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to chat with",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read()
            file_hash = hashlib.md5(file_content).hexdigest()
            uploaded_file.seek(0)
            
            if file_hash not in st.session_state.processed_docs:
                st.info("🔄 Processing document... This may take a moment.")
                with st.expander("👁️ View Processing Details", expanded=True):
                    text = doc_processor.extract_text_from_pdf(uploaded_file)
                    if not text.strip():
                        st.error("❌ Could not extract text from this PDF. Please try a different file.")
                    else:
                        chunks = doc_processor.chunk_text(text)
                        if not chunks:
                            st.error("❌ Could not create text chunks. Please try a different file.")
                        else:
                            embeddings = doc_processor.create_embeddings(chunks)
                            if not embeddings:
                                st.error("❌ Could not create embeddings. Please try again.")
                            else:
                                pinecone_handler.upsert_vectors(embeddings, chunks, uploaded_file.name)
                                st.session_state.processed_docs[file_hash] = {
                                    'name': uploaded_file.name,
                                    'chunks': len(chunks),
                                    'processed_at': datetime.now().isoformat(),
                                    'dimensions': EMBEDDING_DIMENSIONS
                                }
                                st.success("✅ Document processed successfully!")
            else:
                st.info("📋 This document has already been processed")
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            st.error("Please try uploading a different PDF file.")
    
    if st.session_state.processed_docs:
        st.subheader("📚 Processed Documents")
        for doc_hash, doc_info in st.session_state.processed_docs.items():
            with st.container():
                st.write(f"**{doc_info['name']}**")
                st.write(f"- {doc_info['chunks']} chunks created")
                st.write(f"- Processed: {doc_info['processed_at'][:19]}")
                if 'dimensions' in doc_info:
                    st.write(f"- Dimensions: {doc_info['dimensions']}D")
                st.markdown("---")

    st.subheader("🧹 Memory Management")
    col_clear1, col_clear2 = st.columns(2)
    with col_clear1:
        if st.button("🗑️ Clear Chat Only", type="secondary"):
            try:
                st.session_state.messages = []
                # Clear Gemini chat sessions
                gemini_handler.chat_sessions = {}
                st.success("✅ Chat history cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing chat: {str(e)}")
    with col_clear2:
        if st.button("🗑️ Clear All Memory", type="primary"):
            try:
                st.session_state.messages = []
                st.session_state.processed_docs = {}
                gemini_handler.chat_sessions = {}
                # Increment the uploader key to force a reset of the file uploader.
                st.session_state.uploader_key += 1
                st.success("✅ All memory cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing memory: {str(e)}")

# --- Column 2: Chat Interface ---
with col2:
    st.header("💬 Chat Interface")
    
    if st.session_state.processed_docs:
        st.subheader("📄 Select Document to Query")
        doc_names = [doc_info['name'] for doc_info in st.session_state.processed_docs.values()]
        doc_options = doc_names + ["All Documents"]
        default_index = 0
        if doc_names:
            latest_doc = max(st.session_state.processed_docs.values(), key=lambda x: x['processed_at'])
            try:
                default_index = doc_names.index(latest_doc['name'])
            except ValueError:
                default_index = 0
        selected_doc = st.selectbox(
            "Choose which document to ask questions about:",
            options=doc_options,
            index=default_index,
            help="Select a specific document or 'All Documents' to search across all uploaded files"
        )
        if selected_doc == "All Documents":
            st.session_state.selected_document = None
        else:
            st.session_state.selected_document = selected_doc
        if st.session_state.selected_document:
            st.info(f"🎯 Currently querying: **{st.session_state.selected_document}**")
        else:
            st.info("🌐 Currently querying: **All Documents**")
    
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "language" in message:
                    st.caption(f"🌐 Language: {message['language'].upper()}")
                if "timestamp" in message:
                    st.caption(f"⏰ {message['timestamp']}")
    
    if query := st.chat_input("Ask a question about your legal document..."):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add the user's message to the chat history.
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": timestamp
        })
        
        
        # Show a spinner while processing the query.
        with st.spinner("🤔 Processing your question..."):
            try:
                # First, check if any documents have been uploaded.
                if not st.session_state.processed_docs:
                    response = "Please upload a legal document first before asking a question."
                    language = 'en'
                else:
                    # 1. Create an embedding for the user's query.
                    query_embeddings = doc_processor.create_embeddings([query])
                    if not query_embeddings:
                        response = "I could not process your question. Please try rephrasing it."
                        language = 'en'
                    else:
                        query_embedding = query_embeddings[0]
                        
                        # 2. Retrieve the most relevant document chunks from Pinecone.
                        filter_doc_name = getattr(st.session_state, 'selected_document', None)
                        results = pinecone_handler.query_vectors(query_vector=query_embedding, top_k=5, filter_source=filter_doc_name)
                        
                        # 3. Extract the text content from the search results to form the context.
                        context_clauses = []
                        if results.matches:
                            for match in results.matches:
                                if match.metadata and 'text' in match.metadata:
                                    context_clauses.append(match.metadata['text'])
                        
                        # If no relevant clauses are found, inform the user.
                        if not context_clauses:
                            response = "I couldn't find any relevant clauses in the document to answer your question."
                            language = 'en'
                        else:
                            # 4. Detect the language of the user's query to respond in the same language.
                            try:
                                language = langdetect.detect(query)
                            except:
                                language = 'en'  # Default to English if detection fails.

                            # 5. Construct the final prompt for the Gemini model.
                            context = "\n\n---\n\n".join(context_clauses)
                            # The system prompt provides detailed instructions and guidelines to the LLM.
                            system_prompt = """
                            You are an AI Legal Explainer. Your task is to help users understand complex legal documents (rental agreements, loan contracts, terms of service, house documents, etc.) in plain language.

                            Guidelines:
                            1. Use ONLY the quoted text provided from the uploaded document (extracted from text-based PDF or OCR). Do NOT use any external knowledge for factual answers.
                            2. Provide a 1–3 sentence plain-language summary of the entire document when asked.
                            3. When explaining clauses:
                               - Quote the original clause (with page/section info if available).
                               - Paraphrase it clearly in simple language.
                               - Highlight potential risk levels:
                                 - Green: low/no risk
                                 - Amber: medium risk/negotiable
                                 - Red: high risk/seek professional advice
                               - Suggest practical next steps or follow-up questions the user might consider.
                            4. Answer user questions in context of the document only. If the document does not mention the topic, say: "Not stated in document."
                            5. Be concise but complete; avoid unnecessary legal jargon.
                            6. Always encourage safe action: remind users this is NOT legal advice, and for binding legal decisions, they should consult a licensed attorney.
                            7. Detect the language of the user's question and respond in the same language (supports English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada).
                            """
                            
                            # Combine the system prompt, context, and user query into a single prompt.
                            prompt = f"""
                            {system_prompt}

                            DOCUMENT CONTEXT:
                            ```
                            {context}
                            ```

                            USER QUESTION: "{query}"

                            Based on the document context, please answer the user's question in the detected language ({language}) following all the guidelines.
                            """
                            
                            # 6. Generate the response using the Gemini handler.
                            response = gemini_handler.get_response(prompt, st.session_state.session_id)
            except Exception as e:
                # Gracefully handle any exceptions that occur during query processing.
                import traceback
                error_details = traceback.format_exc()
                st.error(f"❌ Error processing query: {str(e)}")
                st.error(f"📋 Error details: {error_details}")
                response = f"Sorry, I encountered an error while processing your question: {str(e)}"
                language = 'en'
        
        # --- Display Assistant Response ---
        # Add the assistant's response to the chat history.
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "language": language,
            "timestamp": timestamp
        })
        
        # Rerun to refresh the chat display with the new message
        st.rerun()

# ===========================
# SIDEBAR INFORMATION
# ===========================
# The sidebar provides additional information and functionality.
with st.sidebar:
    st.header("📊 System Information")
    
    # Display key metrics about the current session.
    st.metric("Messages in Chat", len(st.session_state.messages))
    st.metric("Documents Processed", len(st.session_state.processed_docs))
    st.metric("Session ID", st.session_state.session_id[:8] + "...")

    # --- Supported Languages ---
    st.header("🌐 Supported Languages")
    st.markdown("""
    The AI can understand and respond in the following languages:
    - **English**
    - **Indian Languages:** Bengali, Hindi, Kannada, Malayalam, Tamil, Telugu
    - **Other Languages:** Chinese, French, German, Italian, Japanese, Portuguese, Spanish
    """)
    
    # --- Export Functionality ---
    st.header("💾 Export Options")
    # Button to export the current conversation to a JSON file.
    if st.button("📥 Export Conversation"):
        export_data = {
            'session_id': st.session_state.session_id,
            'messages': st.session_state.messages,
            'processed_docs': st.session_state.processed_docs,
            'configuration': {
                'embedding_model': EMBEDDING_MODEL,
                'dimensions': EMBEDDING_DIMENSIONS
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert the export data to a JSON string.
        json_data = json.dumps(export_data, indent=2)
        
        # Create a download button for the JSON data.
        st.download_button(
            label="💾 Download JSON",
            data=json_data,
            file_name=f"rag_conversation_{st.session_state.session_id[:8]}.json",
            mime="application/json"
        )
    
    # --- Configuration Guide ---
    # An expander to guide users on how to change the embedding model configuration.
    st.header("⚙️ Configuration Guide")
    with st.expander("📋 How to Change Embedding Model", expanded=False):
        st.markdown("""
        **To change the embedding model:**
        
        1. **Edit the configuration section** at the top of the script.
        2. **Comment out the active model** (Option 1 by default).
        3. **Uncomment your desired model** (e.g., Option 2 for English-only focus).
        
        **Example: Switching to the English-only model:**
        ```python
        # Option 1: 384-dimensional multilingual embeddings (Recommended for broad language support).
        # EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
        # EMBEDDING_DIMENSIONS = 384

        # Option 2: 384-dimensional English-focused embeddings (Faster, for English-only documents).
        EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
        EMBEDDING_DIMENSIONS = 384
        ```
        
        4. **Clear All Memory** using the button in the sidebar and restart the app.
        """)
    
    # --- Model Comparison ---
    # A brief comparison of the available embedding models.
    st.header("🔍 Model Comparison")
    st.markdown("""
    **Multilingual Model (Default):**
    - `paraphrase-multilingual-MiniLM-L12-v2` (384D)
    - **Best for:** Supporting a wide range of languages, including the ones listed in the prompt.
    
    **English-focused Models:**
    - `all-MiniLM-L6-v2` (384D): Fast, good quality for English.
    - `all-mpnet-base-v2` (768D): Higher quality for English, but larger and slower.
    """)
    
    # --- Help Section ---
    # An expander with step-by-step instructions on how to use the application.
    st.header("❓ Help & Instructions")
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        **Step 1:** Upload a legal document (PDF) using the file uploader.
        
        **Step 2:** Wait for the document to be processed.
        
        **Step 3:** Use the dropdown to select the document you want to ask questions about.
        
        **Step 4:** Ask questions in the chat input, such as 'Summarize this document' or 'Explain the termination clause.'
        
        **Step 5:** The AI will provide a plain-language explanation based on the document's content.
        
        **Step 6:** Use the 'Clear' buttons to manage your session.
        """)
    
    
    # --- Sample Questions ---
    # A section to provide users with examples of questions they can ask.
    st.header("💡 Sample Questions")
    st.write("Try asking questions like:")
    
    sample_queries = [
        "Can you summarize this document in plain language?",
        "What are the key deadlines mentioned in this contract?",
        "Explain the 'Termination' clause.",
        "What are my obligations regarding confidentiality?"
    ]
    
    # Display the sample questions as buttons.
    for i, query in enumerate(sample_queries):
        if st.button(query, key=f"sample_query_{i}"):
            # Add the sample query to messages and process it
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": timestamp
            })
            
            # Process the sample query
            with st.spinner("🤔 Processing your question..."):
                try:
                    # First, check if any documents have been uploaded.
                    if not st.session_state.processed_docs:
                        response = "Please upload a legal document first before asking a question."
                        language = 'en'
                    else:
                        # 1. Create an embedding for the sample query.
                        query_embeddings = doc_processor.create_embeddings([query])
                        if not query_embeddings:
                            response = "I could not process your question. Please try rephrasing it."
                            language = 'en'
                        else:
                            query_embedding = query_embeddings[0]
                            
                            # 2. Retrieve the most relevant document chunks from Pinecone.
                            filter_doc_name = getattr(st.session_state, 'selected_document', None)
                            results = pinecone_handler.query_vectors(query_vector=query_embedding, top_k=5, filter_source=filter_doc_name)
                            
                            # 3. Extract the text content from the search results to form the context.
                            context_clauses = []
                            if results.matches:
                                for match in results.matches:
                                    if match.metadata and 'text' in match.metadata:
                                        context_clauses.append(match.metadata['text'])
                            
                            # If no relevant clauses are found, inform the user.
                            if not context_clauses:
                                response = "I couldn't find any relevant clauses in the document to answer your question."
                                language = 'en'
                            else:
                                # 4. Detect the language of the sample query to respond in the same language.
                                try:
                                    language = langdetect.detect(query)
                                except:
                                    language = 'en'  # Default to English if detection fails.

                                # 5. Construct the final prompt for the Gemini model.
                                context = "\n\n---\n\n".join(context_clauses)
                                # The system prompt provides detailed instructions and guidelines to the LLM.
                                system_prompt = """
                                You are an AI Legal Explainer. Your task is to help users understand complex legal documents (rental agreements, loan contracts, terms of service, house documents, etc.) in plain language.

                                Guidelines:
                                1. Use ONLY the quoted text provided from the uploaded document (extracted from text-based PDF or OCR). Do NOT use any external knowledge for factual answers.
                                2. Provide a 1–3 sentence plain-language summary of the entire document when asked.
                                3. When explaining clauses:
                                   - Quote the original clause (with page/section info if available).
                                   - Paraphrase it clearly in simple language.
                                   - Highlight potential risk levels:
                                     - Green: low/no risk
                                     - Amber: medium risk/negotiable
                                     - Red: high risk/seek professional advice
                                   - Suggest practical next steps or follow-up questions the user might consider.
                                4. Answer user questions in context of the document only. If the document does not mention the topic, say: "Not stated in document."
                                5. Be concise but complete; avoid unnecessary legal jargon.
                                6. Always encourage safe action: remind users this is NOT legal advice, and for binding legal decisions, they should consult a licensed attorney.
                                7. Detect the language of the user's question and respond in the same language (supports English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Hindi, Bengali, Tamil, Telugu, Malayalam, Kannada).
                                """
                                
                                # Combine the system prompt, context, and user query into a single prompt.
                                prompt = f"""
                                {system_prompt}

                                DOCUMENT CONTEXT:
                                ```
                                {context}
                                ```

                                USER QUESTION: "{query}"

                                Based on the document context, please answer the user's question in the detected language ({language}) following all the guidelines.
                                """
                                
                                # 6. Generate the response using the Gemini handler.
                                response = gemini_handler.get_response(prompt, st.session_state.session_id)
                except Exception as e:
                    # Gracefully handle any exceptions that occur during query processing.
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.error(f"📋 Error details: {error_details}")
                    response = f"Sorry, I encountered an error while processing your question: {str(e)}"
                    language = 'en'
            
            # Add the assistant's response to the chat history.
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "language": language,
                "timestamp": timestamp
            })
            
            # Rerun to update the UI
            st.rerun()
    
    # --- Features & Technical Details ---
    st.header("🔧 Features & Technical Details")
    with st.expander("⚙️ View System Components", expanded=False):
        st.markdown(f"""
        - **PDF Processing**: Extracts text from uploaded PDFs using Google Cloud Document AI.
        - **Multilingual Embeddings**: Uses the `{EMBEDDING_MODEL}` model to create vector embeddings that support multiple languages.
        - **Vector Search**: Stores and searches document embeddings using the Pinecone vector database ({EMBEDDING_DIMENSIONS} dimensions).
        - **AI-Powered Answers**: Leverages the Google Gemini 1.5 Flash model to generate responses based on the retrieved document context.
        - **Chat Interface**: Provides a real-time chat experience built with Streamlit.
        - **Conversation Memory**: Remembers the context of the current conversation.
        """)

# ===========================
# FOOTER
# ===========================
# A simple footer for the application.
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, Google Gemini, and Pinecone</p>
    <p>AI Legal Explainer</p>
</div>
""", unsafe_allow_html=True)