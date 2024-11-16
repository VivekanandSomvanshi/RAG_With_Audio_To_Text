import streamlit as st
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import List
# Groq Integration (if used)
from groq import Groq
from langchain_groq import ChatGroq
# External Libraries
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv('HF_TOKEN')

# Initialize Pinecone and index
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("earningscall")

# Initialize the models
model_embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-70b-versatile", temperature=0.3)

# Define a function to query Pinecone
def query_pinecone(query_text, top_k=5):
    query_embedding = model_embedding.encode(query_text).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

# Custom Pinecone Retriever
class CustomPineconeRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = query_pinecone(query)
        docs = []
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.pop('text', '')  # Remove 'text' from metadata and use it as the main content
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# Set up the custom retriever
custom_retriever = CustomPineconeRetriever()

# Define the LLM and QA chain Langchain
prompt_template = """
Based on the provided information, please answer the user's question accurately. 
Try to get key insights from output.
If the information is insufficient or the answer is unknown, simply respond with "I don't know."

Context: {context}
Question: {question}

Provide a clear and helpful answer below:
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(
    llm=groq_chat,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Streamlit UI
st.title("Earnings Call QA Chatbot")
user_input = st.text_input("Ask a question:")

if user_input:
    result = qa({"query": user_input})
    st.write("Response:", result["result"])
