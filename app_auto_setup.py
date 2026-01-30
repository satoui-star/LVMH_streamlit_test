import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import os
from typing import List, Dict, Tuple
import re

# Page configuration
st.set_page_config(
    page_title="Othello Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

"""
Othello RAG Chatbot Application.

This Streamlit application provides an interactive chat interface for querying
Shakespeare's 'Othello'. It utilizes Retrieval-Augmented Generation (RAG)
to fetch relevant text chunks from the play and generates context-aware
responses using OpenAI's GPT models (optional) or a keyword-based fallback.

Key Features:
- Automated vector database initialization (ChromaDB).
- Semantic search using SentenceTransformers.
- Specialized text cleaning and chunking for dramatic texts.
"""

def download_othello(url: str = "https://www.gutenberg.org/files/1531/1531-0.txt") -> str:
    """Download Othello text from Project Gutenberg."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def clean_text(text: str) -> str:
    """
    Post-process raw text to isolate the play content.

    Removes standard Project Gutenberg headers, footers, and license information
    to ensure embeddings are generated only from the play's actual dialogue
    and stage directions.

    Args:
        text (str): Raw string content from the source URL.

    Returns:
        str: Cleaned text string containing only the dramatic work.
    """
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx]
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Segment the text into overlapping windows for vectorization.

    Overlapping ensures that context (like ongoing dialogue) is not lost
    at arbitrary cut-off points between chunks.

    Args:
        text (str): The full cleaned text of the play.
        chunk_size (int): The number of tokens/words per segment.
        overlap (int): The number of words shared between consecutive segments.

    Returns:
        List[Dict]: A list of dictionaries with chunk text and metadata.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words: continue
            
        chunks.append({
            'text': ' '.join(chunk_words),
            'chunk_id': len(chunks),
            'start_word': i,
            'end_word': min(i + chunk_size, len(words))
        })
    return chunks

def _embed_batch(batch: List[Dict], collection, model):
    """Helper: Embed and store a single batch of chunks."""
    texts = [chunk['text'] for chunk in batch]
    embeddings = model.encode(texts).tolist()
    ids = [f"chunk_{chunk['chunk_id']}" for chunk in batch]
    metadatas = [
        {'chunk_id': c['chunk_id'], 'start': c['start_word'], 'end': c['end_word']}
        for c in batch
    ]
    collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)

def _build_new_database(client, progress_bar, status_text):
    """Helper: Orchestrate the download, clean, chunk, and embed process."""
    status_text.text("📥 Downloading and processing Othello...")
    text = clean_text(download_othello())
    chunks = chunk_text(text)
    
    status_text.text("🤖 Loading model and creating collection...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    collection = client.create_collection(name="othello_collection")
    
    total = len(chunks)
    batch_size = 100
    for i in range(0, total, batch_size):
        progress = int((i / total) * 100)
        progress_bar.progress(progress)
        _embed_batch(chunks[i:i + batch_size], collection, model)
        
    progress_bar.progress(100)
    return collection

@st.cache_resource
def initialize_database():
    """
    Set up the vector search backend.

    Downloads source text, processes it, and generates embeddings on the 
    first run. Subsequent runs detect the existing collection.

    Returns:
        tuple: ChromaDB client and collection object.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        return client, client.get_collection("othello_collection")
    except:
        pass # Collection missing, continue to build

    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        collection = _build_new_database(client, progress_bar, progress_text)
        progress_text.empty()
        progress_bar.empty()
        return client, collection
    except Exception as e:
        st.error(f"DB Init Error: {e}")
        raise

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_relevant_chunks(query: str, collection, model, n: int = 3) -> List[Dict]:
    """Query ChromaDB for relevant text chunks based on semantic similarity."""
    emb = model.encode([query]).tolist()
    res = collection.query(query_embeddings=emb, n_results=n)
    
    chunks = []
    if res['documents']:
        for i, doc in enumerate(res['documents'][0]):
            chunks.append({
                'text': doc,
                'metadata': res['metadatas'][0][i]
            })
    return chunks

def generate_simple_response(query: str, chunks: List[Dict]) -> str:
    """Fallback response generator using raw text when LLM is unavailable."""
    if not chunks: return "No relevant info found in the text."
    
    res = f"Based on Othello, here are relevant passages for '{query}':\n\n"
    for i, c in enumerate(chunks, 1):
        res += f"**Passage {i}:**\n{c['text'][:300]}...\n\n"
    return res

def call_openai_api(query: str, chunks: List[Dict], api_key: str) -> str:
    """
    Generate a natural language answer using an external LLM.
    
    Constructs a prompt with relevant context and handles client initialization 
    (including httpx fix) and error catching.
    """
    try:
        import openai, httpx
        
        context = "\n\n".join([c['text'] for c in chunks])
        messages = [
            {"role": "system", "content": "You are an expert on Shakespeare's Othello."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        client = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.7
        )
        return resp.choices[0].message.content

    except Exception as e:
        st.warning(f"⚠️ OpenAI Error ({type(e).__name__}). Switching to basic mode.")
        return generate_simple_response(query, chunks)

def _render_home_content():
    """Helper: Renders the enhanced copy for the homepage."""
    st.subheader("Ask questions, uncover themes, and analyze characters.")
    st.write(
        "Unlock the depths of Shakespeare's tragedy without searching through pages "
        "of text. Get precise answers supported by direct citations from the play."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### **Context-Aware Discovery**")
        st.write("Find relevant passages based on the *meaning* of your question.")
        st.markdown("### **Intelligent Analysis**")
        st.write("Receive clear, natural language explanations of complex themes.")
    with col2:
        st.markdown("### **Verifiable Evidence**")
        st.write("Every answer includes exact excerpts from the text for verification.")
        st.markdown("### **Continuous Dialogue**")
        st.write("Refine your analysis without losing context.")

def homepage():
    """Render the homepage with the new value proposition."""
    st.title("Explore Othello. Instantly.")
    st.markdown("---")
    _render_home_content()
    st.markdown("---")
    st.success("👈 Open the sidebar and select **Chat** to begin!")

def _render_sidebar() -> Tuple[str, int, bool]:
    """Helper: Renders sidebar with improved labels and returns settings."""
    st.sidebar.header("⚙️ Settings")
    api_key = st.sidebar.text_input(
        "OpenAI Key (Optional)", 
        type="password",
        help="Required for AI summaries. If skipped, returns raw text passages."
    )
    
    if api_key: st.sidebar.success("✅ OpenAI Enabled")
    else: st.sidebar.info("ℹ️ Basic Mode (Text Only)")
        
    n_chunks = st.sidebar.slider("Context Chunks", 1, 5, 3)
    show_sources = st.sidebar.checkbox("Show Sources", value=True)
    
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.messages = []
        st.rerun()
        
    return api_key, n_chunks, show_sources

def _display_chat_history(show_sources: bool):
    """Helper: Renders existing chat messages."""
    for msg in st.session_state.get('messages', []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if show_sources and "sources" in msg:
                with st.expander("📚 Sources"):
                    for s in msg["sources"]:
                        st.text(s['text'][:200] + "...")

def _process_user_query(prompt, api_key, n_chunks, collection, model):
    """Helper: Handles the core logic of processing a new query."""
    with st.chat_message("assistant"):
        with st.spinner("Analyzing text..."):
            chunks = retrieve_relevant_chunks(prompt, collection, model, n_chunks)
            
            if api_key:
                resp = call_openai_api(prompt, chunks, api_key)
            else:
                resp = generate_simple_response(prompt, chunks)
                
            st.markdown(resp)
            return resp, chunks

def chat_page():
    """Render the chat interface."""
    st.title("💬 Chat with Othello")
    api_key, n_chunks, show_src = _render_sidebar()
    
    try:
        client, collection = initialize_database()
        model = load_embedding_model()
    except Exception as e:
        return st.error(f"DB Error: {e}")

    if 'messages' not in st.session_state: st.session_state.messages = []
    _display_chat_history(show_src)

    if prompt := st.chat_input("Ask about Othello..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        resp, sources = _process_user_query(prompt, api_key, n_chunks, collection, model)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": resp, "sources": sources}
        )

def main():
    """Main navigation handler."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Chat"], label_visibility="collapsed")
    
    if page == "Home":
        homepage()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
