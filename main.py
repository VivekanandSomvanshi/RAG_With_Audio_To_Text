from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document
from typing import List
from pinecone import Pinecone
import os
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
import logging

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QA API with Pinecone and Groq")

# Get API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv('HF_TOKEN')

class Query(BaseModel):
    question: str

# Initialize components
@lru_cache()
def init_pinecone():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        return pc.Index("earningscall")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {str(e)}")
        raise

@lru_cache()
def init_sentence_transformer():
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Failed to initialize SentenceTransformer: {str(e)}")
        raise

@lru_cache()
def init_chat_groq():
    try:
        modelgroq = "llama-3.1-70b-versatile"
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name=modelgroq, temperature=0.3)
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {str(e)}")
        raise

# Query Pinecone
def query_pinecone(query_text, top_k=5):
    try:
        model_embedding = init_sentence_transformer()
        index = init_pinecone()
        query_embedding = model_embedding.encode(query_text).tolist()
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        return results
    except Exception as e:
        logger.error(f"Failed to query Pinecone: {str(e)}")
        raise

# Custom Pinecone Retriever
class CustomPineconeRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = query_pinecone(query)
        docs = []
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.pop('text', '')
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# Set up prompt template
prompt_template = """
Based on the provided information, please answer the user's question accurately. 
If the information is insufficient or the answer is unknown, simply respond with "I don't know."

Context: {context}
Question: {question}

Provide a clear and helpful answer below:
Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create QA chain
@lru_cache()
def get_qa_chain():
    llm = init_chat_groq()
    custom_retriever = CustomPineconeRetriever()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=custom_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

# Dependency to check API keys
def check_api_keys():
    if not GROQ_API_KEY or not PINECONE_API_KEY:
        logger.error("API keys are not set")
        raise HTTPException(status_code=400, detail="API keys are not set")
    return True

@app.get("/")
async def root():
    return {"message": "Welcome to the QA API. Use POST /query to ask questions."}

@app.post("/query")
async def query(query: Query, api_keys: bool = Depends(check_api_keys)):
    try:
        logger.info(f"Received query: {query.question}")
        qa_chain = get_qa_chain()
        result = qa_chain({"query": query.question})
        logger.info("Query processed successfully")
        return {"response": result["result"]}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Perform a simple check on each component
        init_pinecone()
        init_sentence_transformer()
        init_chat_groq()
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Health check failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)