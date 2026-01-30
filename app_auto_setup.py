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

def download_othello(url: str = "https://www.gutenberg.org/files/1531/1531-0.txt") -> str:
    """Download Othello text from Project Gutenberg."""
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def clean_text(text: str) -> str:
    """Clean the downloaded text by removing header and footer."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        return text[start_idx:end_idx]
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Split text into overlapping chunks."""
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
    """Initialize DB. Checks existence first to avoid re-running."""
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
    """Query ChromaDB for relevant text chunks."""
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
    """Fallback response generator without LLM."""
    if not chunks: return "No relevant info found."
    
    res = f"Based on Othello, here are relevant passages for '{query}':\n\n"
    for i, c in enumerate(chunks, 1):
        res += f"**Passage {i}:**\n{c['text'][:300]}...\n\n"
    return res

def call_openai_api(query: str, chunks: List[Dict], api_key: str) -> str:
    """Generates response using OpenAI. FIX: Uses custom httpx client."""
    try:
        import openai
        import httpx 
    except ImportError:
        return generate_simple_response(query, chunks)

    context = "\n\n".join([c['text'] for c in chunks])
    messages = [
        {"role": "system", "content": "You are an expert on Shakespeare's Othello."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    # FIX: Explicitly pass httpx.Client to handle proxy environments
    client = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7
    )
    return resp.choices[0].message.content

def _render_home_content():
    """Helper: Renders the text content for homepage."""
    st.header("About This Application")
    st.write("Explore Shakespeare's **Othello** through AI-powered Q&A.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Semantic Search")
        st.write("Finds relevant passages contextually.")
    with col2:
        st.subheader("🤖 AI Responses")
        st.write("Generates answers using retrieved text.")

    st.info("Example: 'What is the significance of the handkerchief?'")

def homepage():
    """Render the homepage."""
    st.title("📚 Welcome to the Othello Chatbot")
    st.markdown("---")
    _render_home_content()
    st.markdown("---")
    st.success("👈 Navigate to **Chat** to start!")

def _render_sidebar() -> Tuple[str, int, bool]:
    """Helper: Renders sidebar and returns settings."""
    st.sidebar.header("⚙️ Settings")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    if api_key: st.sidebar.success("✅ OpenAI Enabled")
    else: st.sidebar.info("ℹ️ Basic Mode")
        
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
        with st.spinner("Thinking..."):
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
