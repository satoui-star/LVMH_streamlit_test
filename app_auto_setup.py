import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
from typing import List, Dict, Tuple
import re

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Othello AI Companion",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
Othello RAG Chatbot Application.

This app combines vector search (ChromaDB) with Generative AI (OpenAI)
to answer questions about Shakespeare's play.
"""

# -----------------------------------------------------------------------------
# DATA PROCESSING FUNCTIONS
# -----------------------------------------------------------------------------

def download_othello(url: str = "https://www.gutenberg.org/files/1531/1531-0.txt") -> str:
    """Download Othello text from Project Gutenberg."""
    # Send HTTP GET request to the URL
    response = requests.get(url)
    # Ensure we got a valid response (200 OK), otherwise raise error
    response.raise_for_status()
    return response.text

def clean_text(text: str) -> str:
    """Post-process raw text to isolate the play content."""
    # Define markers used by Project Gutenberg to bracket the actual book content
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    # Locate the indices of these markers
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    # If both markers are found, slice the text to get the middle content
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx]
    
    # Return original text if markers aren't found (fallback)
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Segment text into overlapping windows for context preservation."""
    words = text.split()
    chunks = []
    
    # Iterate through words with a sliding window approach
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        
        # Skip empty chunks
        if not chunk_words: continue
            
        # Create a dictionary for each chunk with metadata
        chunks.append({
            'text': ' '.join(chunk_words),
            'chunk_id': len(chunks),
            'start_word': i, # tracking position helps with citations later
            'end_word': min(i + chunk_size, len(words))
        })
    return chunks

# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS
# -----------------------------------------------------------------------------

def _embed_batch(batch: List[Dict], collection, model):
    """Helper: Embed and store a single batch of chunks."""
    # Extract just the text strings for embedding
    texts = [chunk['text'] for chunk in batch]
    
    # Generate vector embeddings using the SentenceTransformer model
    embeddings = model.encode(texts).tolist()
    
    # Create unique IDs for ChromaDB
    ids = [f"chunk_{chunk['chunk_id']}" for chunk in batch]
    
    # Prepare metadata for retrieval
    metadatas = [
        {'chunk_id': c['chunk_id'], 'start': c['start_word'], 'end': c['end_word']}
        for c in batch
    ]
    
    # Add everything to the vector database collection
    collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

def _build_new_database(client, progress_bar, status_text):
    """Helper: Orchestrate the download, clean, chunk, and embed process."""
    # Step 1: Download and Clean
    status_text.text("📥 Downloading and processing Othello...")
    text = clean_text(download_othello())
    chunks = chunk_text(text)
    
    # Step 2: Initialize Model
    status_text.text("🤖 Loading model and creating collection...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = client.create_collection(name="othello_collection")
    
    # Step 3: Batch Process (Embedding takes time, so we batch it)
    total = len(chunks)
    batch_size = 100
    
    for i in range(0, total, batch_size):
        # Update UI progress bar
        progress = int((i / total) * 100)
        progress_bar.progress(progress)
        
        # Process specific slice of chunks
        _embed_batch(chunks[i:i + batch_size], collection, model)
        
    progress_bar.progress(100)
    return collection

@st.cache_resource
def initialize_database():
    """
    Initialize DB. Uses @st.cache_resource to ensure this only runs once.
    """
    # Create persistent client pointing to local folder
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Try to load existing collection to avoid rebuilding
    try:
        return client, client.get_collection("othello_collection")
    except:
        pass # Collection missing, proceed to build

    # UI elements for loading state
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Build the database from scratch
        collection = _build_new_database(client, progress_bar, progress_text)
        
        # Cleanup UI
        progress_text.empty()
        progress_bar.empty()
        return client, collection
    except Exception as e:
        st.error(f"DB Init Error: {e}")
        raise

@st.cache_resource
def load_embedding_model():
    """Load and cache the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------------------------------------------------------
# RETRIEVAL & AI LOGIC
# -----------------------------------------------------------------------------

def retrieve_relevant_chunks(query: str, collection, model, n: int = 3) -> List[Dict]:
    """Query ChromaDB for relevant text chunks."""
    # Convert user query to vector
    emb = model.encode([query]).tolist()
    
    # Perform nearest neighbor search
    res = collection.query(query_embeddings=emb, n_results=n)
    
    chunks = []
    # Parse results back into friendly dictionary format
    if res['documents']:
        for i, doc in enumerate(res['documents'][0]):
            chunks.append({
                'text': doc,
                'metadata': res['metadatas'][0][i]
            })
    return chunks

def generate_simple_response(query: str, chunks: List[Dict]) -> str:
    """Fallback response generator (No AI)."""
    if not chunks: return "No relevant info found."
    
    # Simple string concatenation of found chunks
    res = f"Based on Othello, here are relevant passages for '{query}':\n\n"
    for i, c in enumerate(chunks, 1):
        res += f"**Passage {i}:**\n{c['text'][:300]}...\n\n"
    return res

def call_openai_api(query: str, chunks: List[Dict], api_key: str) -> str:
    """Generates response using OpenAI. Includes HTTPX fix."""
    try:
        import openai
        import httpx  # Required to fix proxy/environment issues
        
        # Context window construction
        context = "\n\n".join([c['text'] for c in chunks])
        
        # Prompt engineering
        messages = [
            {"role": "system", "content": "You are an expert on Shakespeare's Othello."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        # FIX: Explicitly pass httpx.Client to handle proxy environments
        client = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
        
        # API Call
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.7
        )
        return resp.choices[0].message.content

    except Exception as e:
        # Graceful fallback if API fails
        st.warning(f"⚠️ OpenAI Error ({type(e).__name__}). Switching to basic mode.")
        return generate_simple_response(query, chunks)

# -----------------------------------------------------------------------------
# UI & HOMEPAGE DESIGN
# -----------------------------------------------------------------------------

def _render_hero_section():
    """Renders the main title and introduction."""
    # Main visual title with emoji
    st.markdown("""
        <h1 style='text-align: center; color: #2e4057;'>
            🎭 The Othello AI Companion
        </h1>
        <p style='text-align: center; font-style: italic; font-size: 1.2em;'>
            "I will wear my heart upon my sleeve for daws to peck at"
        </p>
    """, unsafe_allow_html=True)
    
    st.divider()

def _render_features_grid():
    """Renders the feature value props in a grid."""
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.info("🔍 **Context-Aware Discovery**")
        st.write("Find passages based on meaning, not just keywords.")
        
        st.info("📖 **Verifiable Evidence**")
        st.write("Every answer is backed by direct citations from the play.")

    with col2:
        st.success("🤖 **Intelligent Analysis**")
        st.write("Get natural language summaries of themes and characters.")
        
        st.success("💬 **Interactive Chat**")
        st.write("Ask follow-up questions to dive deeper into the tragedy.")

def homepage():
    """Assemble the homepage components."""
    _render_hero_section()
    _render_features_grid()
    
    # Call to action
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center;'>
            <h3>Ready to explore?</h3>
            <p>👈 <strong>Open the Sidebar</strong> and select <strong>'Chat'</strong> to begin.</p>
        </div>
    """, unsafe_allow_html=True)

def _render_sidebar() -> Tuple[str, int, bool]:
    """Helper: Renders sidebar and returns settings."""
    st.sidebar.header("⚙️ Configuration")
    
    # API Key Input
    api_key = st.sidebar.text_input(
        "OpenAI API Key (Optional)", 
        type="password",
        help="Leave empty to use free 'Basic Mode' (Text Search only)."
    )
    
    # Status indicator
    if api_key: st.sidebar.success("✅ AI Mode Enabled")
    else: st.sidebar.info("ℹ️ Basic Mode Active")
        
    # Advanced settings
    st.sidebar.markdown("---")
    n_chunks = st.sidebar.slider("Context Depth (Chunks)", 1, 5, 3)
    
    # Force sources to be always shown (Removed Checkbox)
    show_sources = True 
    
    # Reset button
    if st.sidebar.button("🗑️ Reset Conversation"):
        st.session_state.messages = []
        st.rerun()
        
    return api_key, n_chunks, show_sources

# -----------------------------------------------------------------------------
# CHAT PAGE LOGIC
# -----------------------------------------------------------------------------

def _display_chat_history(show_sources: bool):
    """Helper: Renders existing chat messages."""
    for msg in st.session_state.get('messages', []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Render sources if they exist and user enabled them
            if show_sources and "sources" in msg:
                with st.expander("📚 View Source Text"):
                    for s in msg["sources"]:
                        st.text(f"...{s['text'][:200]}...")

def _process_user_query(prompt, api_key, n_chunks, collection, model):
    """Helper: Orchestrates the search and answer generation."""
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Shakespeare's text..."):
            # 1. Search DB
            chunks = retrieve_relevant_chunks(prompt, collection, model, n_chunks)
            
            # 2. Generate Answer (AI or Basic)
            if api_key:
                resp = call_openai_api(prompt, chunks, api_key)
            else:
                resp = generate_simple_response(prompt, chunks)
                
            st.markdown(resp)
            
            # Display sources immediately for the current answer
            # Since we removed the checkbox, we always show them here.
            if chunks:
                with st.expander("📚 View Source Text"):
                    for s in chunks:
                        st.text(f"...{s['text'][:200]}...")
            
            return resp, chunks

def chat_page():
    """Render the main chat interface."""
    st.title("💬 Discuss the Tragedy")
    api_key, n_chunks, show_src = _render_sidebar()
    
    # Ensure DB is ready before showing chat inputs
    try:
        client, collection = initialize_database()
        model = load_embedding_model()
    except Exception as e:
        return st.error(f"System Error: {e}")

    # Init chat history
    if 'messages' not in st.session_state: st.session_state.messages = []
    _display_chat_history(show_src)

    # Handle User Input
    if prompt := st.chat_input("Ask about Iago, Desdemona, or the Handkerchief..."):
        # Save and show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Generate and show assistant response
        resp, sources = _process_user_query(prompt, api_key, n_chunks, collection, model)
        
        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": resp, "sources": sources}
        )

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    """Main entry point for navigation."""
    st.sidebar.title("🧭 Navigation")
    
    # Sidebar Radio for Page Switching
    page = st.sidebar.radio("Go to", ["Home", "Chat"], label_visibility="collapsed")
    
    if page == "Home":
        homepage()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
