import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import os
import openai

# --- CONFIGURATION INITIALE ---
st.set_page_config(page_title="Othello Expert RAG", layout="wide")

@st.cache_resource
def get_resources():
    """Charge ChromaDB et le modèle d'embeddings une seule fois."""
    client = chromadb.PersistentClient(path="./chroma_db")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return client, model

def load_and_clean_othello():
    """Charge le texte local d'Othello."""
    if not os.path.exists("othello.txt"):
        return ""
    with open("othello.txt", "r", encoding="utf-8") as f:
        text = f.read()
    start = text.find("* START")
    return text[start:] if start != -1 else text

def create_chunks(text, size=500, overlap=50):
    """Découpe le texte en morceaux pour la base de données."""
    words = text.split()
    return [' '.join(words[i:i + size]) for i in range(0, len(words), size - overlap)]

def get_collection():
    """Initialise ou récupère la collection ChromaDB."""
    client, model = get_resources()
    try:
        return client.get_collection("othello_docs")
    except:
        coll = client.create_collection("othello_docs")
        chunks = create_chunks(load_and_clean_othello())
        for i, chunk in enumerate(chunks):
            coll.add(ids=[f"id_{i}"], documents=[chunk], 
                     embeddings=model.encode([chunk]).tolist())
        return coll

def call_llm(query, context, api_key=None):
    """Appelle OpenAI ou LM Studio (Correction du bug proxies)."""
    if api_key:
        # Mode OpenAI Cloud
        client = openai.OpenAI(api_key=api_key)
        model_name = "gpt-3.5-turbo"
    else:
        # Mode LM Studio Local (Syntaxe corrigée sans proxies)
        client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        model_name = "local-model"

    prompt = f"Contexte d'Othello :\n{context}\n\nQuestion : {query}\nRéponse :"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": "Tu es un expert de Shakespeare."},
                  {"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Clé API OpenAI (laisser vide pour LM Studio)", type="password")
    if st.sidebar.button("🗑️ Effacer la discussion"):
        st.session_state.messages = []

    coll = get_collection()
    _, model_emb = get_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Posez une question sur Othello..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)

        # RAG : Recherche de contexte
        results = coll.query(query_embeddings=model_emb.encode([prompt]).tolist(), n_results=2)
        context = "\n".join(results['documents'][0])

        with st.chat_message("assistant"):
            try:
                answer = call_llm(prompt, context, api_key if api_key else None)
            except Exception as e:
                answer = f"Erreur : Vérifiez que LM Studio est lancé (port 1234) ou votre clé. \n\n{e}"
            
            st.write(answer)
            with st.expander("Sources consultées (RAG)"): st.info(context)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if _name_ == "_main_":
    main()
