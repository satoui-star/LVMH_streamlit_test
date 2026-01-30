"""
Script to download Othello, chunk it, generate embeddings, and store in ChromaDB.
This script must be run once before launching the Streamlit app.
"""

import requests
import chromadb
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict


def download_othello(url: str = "https://www.gutenberg.org/files/1531/1531-0.txt") -> str:
    """
    Download Othello text from Project Gutenberg.
    
    Args:
        url: URL to download Othello text from
        
    Returns:
        str: Full text of Othello
    """
    print("Downloading Othello from Project Gutenberg...")
    response = requests.get(url)
    response.raise_for_status()
    text = response.text
    print("Download complete!")
    return text


def clean_text(text: str) -> str:
    """
    Clean the downloaded text by removing Project Gutenberg header and footer.
    
    Args:
        text: Raw text from Project Gutenberg
        
    Returns:
        str: Cleaned text containing only the play
    """
    # Remove Project Gutenberg header (before "OTHELLO")
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, any]]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: The full text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List[Dict]: List of dictionaries containing chunk text and metadata
    """
    # Split text into words
    words = text.split()
    chunks = []
    
    # Create overlapping chunks
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append({
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_word': i,
                'end_word': min(i + chunk_size, len(words))
            })
    
    print(f"Created {len(chunks)} chunks from the text")
    return chunks


def generate_embeddings_and_store(chunks: List[Dict[str, any]], 
                                  collection_name: str = "othello_collection") -> None:
    """
    Generate embeddings for chunks and store them in ChromaDB.
    
    Args:
        chunks: List of text chunks with metadata
        collection_name: Name of the ChromaDB collection to create
    """
    print("Loading embedding model...")
    # Use a lightweight but effective embedding model from HuggingFace
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Initializing ChromaDB...")
    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete collection if it exists (fresh start)
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(name=collection_name)
    print(f"Created new collection: {collection_name}")
    
    # Generate embeddings and add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in batch_chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for chunks {i} to {i + len(batch_chunks)}...")
        embeddings = model.encode(texts).tolist()
        
        # Prepare data for ChromaDB
        ids = [f"chunk_{chunk['chunk_id']}" for chunk in batch_chunks]
        metadatas = [
            {
                'chunk_id': chunk['chunk_id'],
                'start_word': chunk['start_word'],
                'end_word': chunk['end_word']
            }
            for chunk in batch_chunks
        ]
        
        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!")
    print(f"Collection '{collection_name}' is ready for querying.")


def main():
    """
    Main function to orchestrate the vector database generation process.
    """
    # Download Othello
    othello_text = download_othello()
    
    # Clean the text
    cleaned_text = clean_text(othello_text)
    
    # Chunk the text
    chunks = chunk_text(cleaned_text, chunk_size=500, overlap=50)
    
    # Generate embeddings and store in ChromaDB
    generate_embeddings_and_store(chunks)
    
    print("\n" + "="*50)
    print("Vector database generation complete!")
    print("You can now run the Streamlit app with: streamlit run app.py")
    print("="*50)


if __name__ == "__main__":
    main()
