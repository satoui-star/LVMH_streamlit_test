"""
Streamlit Chatbot Application with RAG capabilities for Othello.
This app provides a chat interface to ask questions about Shakespeare's Othello.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Tuple


# Page configuration
st.set_page_config(
    page_title="Othello Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_embedding_model():
    """
    Load and cache the embedding model for semantic search.
    
    Returns:
        SentenceTransformer: Loaded embedding model
    """
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_llm_model(model_name: str):
    """
    Load and cache the language model for text generation.
    
    Args:
        model_name: Name of the HuggingFace model to load
        
    Returns:
        Tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return tokenizer, model


@st.cache_resource
def get_chromadb_client():
    """
    Initialize and cache ChromaDB client.
    
    Returns:
        chromadb.Client: ChromaDB client instance
    """
    return chromadb.PersistentClient(path="./chroma_db")


def retrieve_relevant_chunks(query: str, collection, embedding_model, n_results: int = 3) -> List[Dict]:
    """
    Retrieve the most relevant chunks from ChromaDB based on the query.
    
    Args:
        query: User's question
        collection: ChromaDB collection to query
        embedding_model: Model to generate query embedding
        n_results: Number of relevant chunks to retrieve
        
    Returns:
        List[Dict]: List of relevant chunks with metadata
    """
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query]).tolist()
    
    # Query ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    # Format results
    relevant_chunks = []
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            relevant_chunks.append({
                'text': doc,
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
    
    return relevant_chunks


def generate_response(query: str, context_chunks: List[Dict], tokenizer, model) -> str:
    """
    Generate a response using the LLM with retrieved context.
    
    Args:
        query: User's question
        context_chunks: Relevant text chunks from Othello
        tokenizer: HuggingFace tokenizer
        model: HuggingFace language model
        
    Returns:
        str: Generated response
    """
    # Prepare context from retrieved chunks
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
    # Create prompt with context and question
    prompt = f"""Based on the following excerpts from Shakespeare's Othello, answer the question.

Context from Othello:
{context}

Question: {query}

Answer based on the context above:"""
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the answer part (remove the prompt)
    answer = response[len(prompt):].strip()
    
    return answer


def homepage():
    """
    Render the homepage of the application.
    """
    st.title("📚 Welcome to the Othello Chatbot")
    st.markdown("---")
    
    # Introduction section
    st.header("About This Application")
    st.write("""
    This intelligent chatbot allows you to explore Shakespeare's **Othello** through 
    natural language questions. Using advanced AI technology, it retrieves relevant 
    passages from the play and generates contextual answers.
    """)
    
    # Features section
    st.header("✨ Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Semantic Search")
        st.write("Finds the most relevant passages from Othello based on your question.")
        
        st.subheader("🤖 AI-Powered Responses")
        st.write("Generates accurate answers using state-of-the-art language models.")
    
    with col2:
        st.subheader("📖 Source Citations")
        st.write("Every answer includes the exact text passages used as sources.")
        
        st.subheader("💬 Conversation History")
        st.write("Maintains context across multiple questions in your chat session.")
    
    # How to use section
    st.header("🚀 How to Use")
    st.write("""
    1. Navigate to the **Chat** page using the sidebar
    2. Select your preferred language model
    3. Type your question about Othello in the chat input
    4. View the AI-generated response along with source citations
    5. Continue the conversation with follow-up questions
    """)
    
    # Example questions
    st.header("💡 Example Questions")
    st.info("""
    - "What is the main conflict in Othello?"
    - "Who is Iago and what role does he play?"
    - "What happens to Desdemona?"
    - "Describe Othello's character traits"
    - "What is the significance of the handkerchief?"
    """)
    
    # Navigation
    st.markdown("---")
    st.success("👈 Navigate to the **Chat** page from the sidebar to start asking questions!")


def chat_page():
    """
    Render the chat interface page.
    """
    st.title("💬 Chat with Othello")
    
    # Sidebar for model selection
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_options = {
        "TinyLlama (Fast, Low Memory)": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "Phi-2 (Balanced)": "microsoft/phi-2",
        "Mistral-7B (High Quality, Requires GPU)": "mistralai/Mistral-7B-Instruct-v0.1"
    }
    
    selected_model_name = st.sidebar.selectbox(
        "Choose Language Model",
        options=list(model_options.keys()),
        index=0
    )
    
    selected_model = model_options[selected_model_name]
    
    # Number of context chunks
    n_chunks = st.sidebar.slider(
        "Number of Context Chunks",
        min_value=1,
        max_value=5,
        value=3,
        help="More chunks provide more context but may slow down responses"
    )
    
    # Show sources toggle
    show_sources = st.sidebar.checkbox("Show Source Citations", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tip: Start with a simple question about Othello!")
    
    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and show_sources:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Othello..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching Othello and generating response..."):
                try:
                    # Load models and database
                    embedding_model = load_embedding_model()
                    tokenizer, llm_model = load_llm_model(selected_model)
                    client = get_chromadb_client()
                    collection = client.get_collection("othello_collection")
                    
                    # Retrieve relevant chunks
                    relevant_chunks = retrieve_relevant_chunks(
                        prompt, 
                        collection, 
                        embedding_model, 
                        n_results=n_chunks
                    )
                    
                    # Generate response
                    response = generate_response(prompt, relevant_chunks, tokenizer, llm_model)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources if enabled
                    if show_sources:
                        with st.expander("📚 View Sources"):
                            for i, chunk in enumerate(relevant_chunks, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])
                                st.markdown("---")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": relevant_chunks
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}\n\nPlease ensure you have run `generate_vector_db.py` first!"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def main():
    """
    Main function to handle page navigation and rendering.
    """
    # Sidebar navigation
    st.sidebar.title("📚 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Chat"],
        label_visibility="collapsed"
    )
    
    # Render selected page
    if page == "Home":
        homepage()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()
