import os
import pandas as pd
from rag import clean_text, chunk_text, embed_chunks, retrieve_related_chunks, add_chunks_to_database
import utils.fileRetrieve as fr
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
from config import chunk_size, chunk_overlap, batch_size, dataset_name, chromadb_collection_name, top_k

load_dotenv() # Load environment variables from .env file

# Sets up logging to see messages in console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

def initialize_rag_pipeline():
    dataset = fr.retrieve_file(dataset_name=dataset_name)  # retrieve dataset
    all_chunks = []
    chunk_ids = []
    # Loop over rows and chunk if over set size
    for i, row in dataset.iterrows():
        # combine dataset
        text = clean_text(
            f"question: {row['Question']} answer: {row['Answer']} category: {row['Category']}")
        logger.info(f"Processing text: {text}")
        chunks = chunk_text(text, chunk_size, overlap=chunk_overlap)
        # note: its extend not append - causes flat list not list of lists
        all_chunks.extend(chunks)
        # assign unique id to each chunk eg:1-2
        chunk_ids.extend([f"{i}-{j}" for j in range(len(chunks))])

    chunk_embeddings = {}  # dictionary key: chunk_id, value: embedding
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start:start + batch_size]
        batch_ids = chunk_ids[start:start + batch_size]
        batch_embeddings = embed_chunks(batch)

        for cid, emb in zip(batch_ids, batch_embeddings):
            chunk_embeddings[cid] = emb
    add_chunks_to_database(chunk_ids, all_chunks, list(chunk_embeddings.values()))
    logger.info("all chunks embedded and mapped to IDs")


class QueryRequest(BaseModel):
    query: str
    top_k: int = top_k

#listening
@app.post("/retrieve")
def retrieve(request: QueryRequest):
    results = retrieve_related_chunks(request.query, top_k=request.top_k)
    return {"results": results}

#fetch file,process text and save chunks to database
try:
    initialize_rag_pipeline()
#error handling
except (RuntimeError) as e:
    logger.error(f"RAG pipeline init failed: {e}")
except (FileNotFoundError) as e:
    logger.error("file was not found")
except Exception as e:
    logger.error(f"An unexpected error occurred during RAG pipeline initialization: {e}")