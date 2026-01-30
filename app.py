"""
Streamlit Chatbot Application with RAG capabilities for Othello.
This app provides a chat interface to ask questions about Shakespeare's Othello.
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict


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


def generate_simple_response(query: str, context_chunks: List[Dict]) -> str:
    """
    Generate a simple response by summarizing the retrieved context.
    This is a fallback when LLM is not available.
    
    Args:
        query: User's question
        context_chunks: Relevant text chunks from Othello
        
    Returns:
        str: Response with context
    """
    if not context_chunks:
        return "I couldn't find relevant information in Othello to answer your question."
    
    response = f"Based on the text of Othello, here are the most relevant passages:\n\n"
    
    for i, chunk in enumerate(context_chunks, 1):
        response += f"**Passage {i}:**\n{chunk['text'][:400]}...\n\n"
    
    response += "\nThese passages from Othello should help answer your question about: " + query
    
    return response


def call_openai_api(query: str, context_chunks: List[Dict], api_key: str) -> str:
    """
    Call OpenAI API to generate a response with the retrieved context.
    
    Args:
        query: User's question
        context_chunks: Relevant text chunks from Othello
        api_key: OpenAI API key
        
    Returns:
        str: Generated response
    """
    import openai
    
    # Prepare context from retrieved chunks
    context = "\n\n".join([chunk['text'] for chunk in context_chunks])
    
    # Create prompt with context and question
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about Shakespeare's Othello based on provided text excerpts."},
        {"role": "user", "content": f"""Based on the following excerpts from Shakespeare's Othello, answer the question.

Context from Othello:
{context}

Question: {query}

Please provide a clear, concise answer based only on the context above."""}
    ]
    
    # Call OpenAI API
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    
    return response.choices[0].message.content


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
    passages from the play and provides contextual answers.
    """)
    
    # Features section
    st.header("✨ Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Semantic Search")
        st.write("Finds the most relevant passages from Othello based on your question.")
        
        st.subheader("🤖 AI-Powered Responses")
        st.write("Generates accurate answers using retrieved context.")
    
    with col2:
        st.subheader("📖 Source Citations")
        st.write("Every answer includes the exact text passages used as sources.")
        
        st.subheader("💬 Conversation History")
        st.write("Maintains context across multiple questions in your chat session.")
    
    # How to use section
    st.header("🚀 How to Use")
    st.write("""
    1. Navigate to the **Chat** page using the sidebar
    2. (Optional) Enter your OpenAI API key for enhanced responses
    3. Type your question about Othello in the chat input
    4. View the response along with source citations
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
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # API Key input (optional)
    st.sidebar.subheader("OpenAI API (Optional)")
    api_key = st.sidebar.text_input(
        "API Key",
        type="password",
        help="Enter your OpenAI API key for better responses. Leave empty to use basic mode."
    )
    
    use_openai = len(api_key) > 0
    
    if use_openai:
        st.sidebar.success("✅ Using OpenAI API")
    else:
        st.sidebar.info("ℹ️ Using basic mode (context only)")
    
    # Number of context chunks
    n_chunks = st.sidebar.slider(
        "Number of Context Chunks",
        min_value=1,
        max_value=5,
        value=3,
        help="More chunks provide more context"
    )
    
    # Show sources toggle
    show_sources = st.sidebar.checkbox("Show Source Citations", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Tip: Start with a simple question about Othello!")
    
    # Check if database exists
    try:
        client = get_chromadb_client()
        collection = client.get_collection("othello_collection")
    except Exception as e:
        st.error(f"""
        ❌ **Vector database not found!**
        
        Please run `python generate_vector_db.py` first to create the database.
        
        Error details: {str(e)}
        """)
        return
    
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
                    # Load embedding model
                    embedding_model = load_embedding_model()
                    
                    # Retrieve relevant chunks
                    relevant_chunks = retrieve_relevant_chunks(
                        prompt, 
                        collection, 
                        embedding_model, 
                        n_results=n_chunks
                    )
                    
                    # Generate response
                    if use_openai:
                        try:
                            response = call_openai_api(prompt, relevant_chunks, api_key)
                        except Exception as e:
                            st.warning(f"OpenAI API error: {str(e)}. Falling back to basic mode.")
                            response = generate_simple_response(prompt, relevant_chunks)
                    else:
                        response = generate_simple_response(prompt, relevant_chunks)
                    
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
                    error_msg = f"❌ Error: {str(e)}"
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
