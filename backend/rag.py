import tiktoken
import os
from openai import OpenAI
import random
import chromadb
import numpy as np
import re
import logging
from config import chromadb_collection_name, top_k, similarity_threshold


logger = logging.getLogger(__name__)

enc = tiktoken.get_encoding("cl100k_base")

def get_chroma_collection(collection_name=chromadb_collection_name):
    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.get_or_create_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Error getting or creating Chroma collection {collection_name}: {e}")
        raise
    return collection

collection = get_chroma_collection()

# Text utils
def clean_text(text):
    # remove special characters
    text = re.sub(r'[^a-zA-Z0-9\\s:]', '', text)
    # remove extra whitespace
    text = re.sub(r'\\s+', ' ', text)
    # convert to lowercase
    text = text.lower()
    # remove stopwords
    words = text.split()
    return text

# Token utils
def tokenize(text: str) -> list[int]:
    return enc.encode(text)

def detokenize(tokens: list[int]) -> str:
    return enc.decode(tokens)


def chunk_text(text: str, chunk_size=300, overlap=50) -> list[str]:
    tokens = tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(detokenize(chunk_tokens))
        start = end - overlap
    return chunks

#function to embed the chunks
def embed_chunks(chunks,retry_attempt=2):
    # Use mock embeddings if MOCK_EMBEDDINGS environment variable is set to 'true'
    if os.environ.get("MOCK_EMBEDDINGS", "false").lower() == "true":
        logger.info("Using mock embeddings.")
        return embed_chunks_mock(chunks)

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # Send all chunks in a single call
        logger.info(f"Trying to send {len(chunks)} chunks for embedding.")
        if not isinstance(chunks, list):
            chunks = [chunks]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks)
        embeddings = [e.embedding for e in resp.data]
        return embeddings
    except Exception as e:
        logger.warning(f"Retrying to send chunks... attempts left: {retry_attempt}. Error: {e}")
        if(retry_attempt>0):
            return embed_chunks(chunks,retry_attempt-1)
        else:
            logger.error("Failed to embed chunks after multiple retries.")
            raise Exception("failed to embed chunks")
            
def embed_chunks_mock(chunks):
    embeddings = []
    for chunk in chunks:
        # generate a fake vector of length 1536 (like text-embedding-3-small)
        embeddings.append([random.random() for _ in range(1536)])
    return embeddings

def add_chunks_to_database(chunk_ids, chunks, embeddings, collection_name=chromadb_collection_name):
    collection = get_chroma_collection(collection_name)
    
    collection.add(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings
    )
    
    logger.info(f"Successfully Added {len(chunks)} chunks to Chroma collection '{collection_name}'.")
    return collection

def retrieve_related_chunks(query, top_k=top_k, similarity_threshold=similarity_threshold, collection_name=chromadb_collection_name):
    logger.info(f"Retrieving related chunks for query: {query}")
    query_chunks = chunk_text(query)
    query_embeddings = embed_chunks(query_chunks) # Embed all query chunks in one go
    
    collection = get_chroma_collection(collection_name)
    
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k
    )

    # Filter results by similarity_threshold if needed, but ChromaDB's query already returns most similar
    # The 'distances' field in results can be used for this if filtering by an absolute threshold is required.
    # For now, we'll return the top_k directly.
    
    retrieved_documents = results['documents']
    retrieved_distances = results['distances']
    retrieved_ids = results['ids']

    # Flatten the list of lists returned by ChromaDB query
    flat_documents = [item for sublist in retrieved_documents for item in sublist]
    flat_distances = [item for sublist in retrieved_distances for item in sublist]
    flat_ids = [item for sublist in retrieved_ids for item in sublist]
    
    # Combine into a list of tuples (id, similarity, document)
    formatted_results = []
    for i in range(len(flat_documents)):
        if 1 - flat_distances[i] >= similarity_threshold: # Assuming distance is L2, 1-distance approximates cosine similarity
            formatted_results.append((flat_ids[i], 1 - flat_distances[i], flat_documents[i]))
    
    logger.info(f"Found {len(formatted_results)} related chunks.")
    return formatted_results
