import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama

client = chromadb.Client(Settings())
collection = client.get_or_create_collection("documents")

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("Interactive Ollama with ChromaDB")
st.subheader("Embed documents and use them as context for AI responses")

if st.button("Show All Documents in ChromaDB"):
    docs = collection.get()
    if docs and 'metadatas' in docs and isinstance(docs['metadatas'], list):
        st.write("### Stored Documents:")
        for i, metadata in enumerate(docs['metadatas']):
            st.write(f"**Document {i + 1}:** {metadata.get('content', 'No content')} ")
    else:
        st.write("No documents found in ChromaDB.")

st.write("### Add a New Document")
new_doc = st.text_area("Enter document text:", "")
if st.button("Add Document"):
    if new_doc.strip():
        embedding = model.encode([new_doc])[0]
        existing_docs = collection.get()
        existing_ids = existing_docs['ids'] if existing_docs and 'ids' in existing_docs else []
        doc_id = f"doc_{len(existing_ids)}"
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            metadatas=[{"content": new_doc}]
        )
        st.success("Document added successfully!")
        st.write(f"Added document with ID: {doc_id} and content: {new_doc}")
    else:
        st.error("Document cannot be empty.")

st.write("### Ask Ollama a Question")
question = st.text_input("Enter your question:", "")
if st.button("Ask Ollama"):
    if question.strip():
        query_embedding = model.encode([question])[0] 
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )

        if results and 'metadatas' in results and len(results['metadatas']) > 0:
            context = results['metadatas'][0][0].get('content', "No content")

            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            try:
                response = ollama.chat(
                    model="llama2",
                    messages=[{"role": "user", "content": prompt}]
                )
                st.write(f"### Ollama's Response:\n{response['message']['content']}")
            except Exception as e:
                st.error(f"Error interacting with Ollama: {e}")
        else:
            st.error("No relevant context found in ChromaDB. Make sure the document is added.")
    else:
        st.error("Question cannot be empty.")
