Othello RAG Chatbot
Explore Shakespeare's Othello through AI-powered conversation.

This application is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit. It allows users to ask natural language questions about the play and receive answers grounded in the actual text. It combines semantic search (to find relevant passages) with OpenAI's GPT models (to synthesize answers).

Features
Context-Aware Discovery: Uses vector embeddings to find relevant passages based on the meaning of your query, not just keywords.

Dual Modes:

AI Mode: (Requires API Key) Generates natural language answers summarizing the text.

Basic Mode: (Free) Returns the raw text passages most relevant to your query.

Verifiable Evidence: Every response provides "Source Citations"—the exact excerpts from the play used to generate the answer.

Auto-Setup: Automatically downloads Othello from Project Gutenberg, cleans it, chunks it, and builds the vector database on the first run.

Robust Networking: Includes custom HTTP client handling to prevent proxy/firewall issues often found in corporate or cloud environments.

Tech Stack
Frontend: Streamlit

Vector Database: ChromaDB

Embeddings: all-MiniLM-L6-v2 via SentenceTransformers

LLM Integration: OpenAI GPT-3.5 Turbo

HTTP Client: httpx (for robust API connections)

Installation & Setup
Prerequisites
Python 3.8 or higher.

An OpenAI API Key (optional, but required for AI-generated text).

1. Clone or Download the Repository
Ensure you have the app_auto_setup.py file in your project directory.

2. Install Dependencies
Create a requirements.txt file with the following content, or install them directly:

Bash
pip install streamlit chromadb sentence-transformers openai requests httpx
3. Run the Application
Execute the following command in your terminal:

Bash
streamlit run app_auto_setup.py
Usage Guide
First Run (Initialization)
When you run the app for the first time, it will perform a one-time setup:

Download Othello from Project Gutenberg.

Clean and segment the text into chunks.

Download the embedding model.

Generate vectors and store them in a local chroma_db folder.

Note: This process may take 1-2 minutes depending on your internet connection and CPU.

The Chat Interface
Navigation: Use the sidebar to switch between Home and Chat.

Settings (Sidebar):

OpenAI Key: Enter your key here. The input is masked for security. If left blank, the app runs in "Basic Mode" (text retrieval only).

Context Chunks: Adjust how many text passages (1-5) are fed to the AI. More chunks = more context but higher token usage.

Show Sources: Toggle to hide/show the raw text citations below the answer.

Asking Questions: Type questions like "Why does Iago hate Othello?" or "What is the significance of the handkerchief?"

 Project Structure
Plaintext
.

├── app_auto_setup.py # Main application logic

├── chroma_db/     # (Generated on first run) Stores vector embeddings

└── README.md          # This documentation
 Troubleshooting
1. OpenAI API Error or AuthenticationError

Ensure your API key is correct and has active credits.

The app includes a fallback: if the API fails, it automatically switches to Basic Mode and shows the raw text chunks instead.

2. Network/Proxy Issues

This app uses httpx to handle network requests. If you are on a strict corporate network and cannot reach OpenAI, the app will gracefully degrade to Basic Mode.

3. "Model loading..." takes a long time

The first run requires downloading the SentenceTransformer model (approx. 80MB) and the book text. Subsequent runs will be instant as these resources are cached.

⚖️ License & Attribution
Text Source: Othello by William Shakespeare, provided by Project Gutenberg.

This project is for educational and demonstration purposes.
