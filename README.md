# RAG Retrieval Demo Program

This project implements a Retrieval-Augmented Generation (RAG) demo program using FastAPI for the backend and a simple HTML/JavaScript frontend.

## Project Structure

```
.editorconfig
.env (create this file, see Setup)
README.md
requirements.txt (create this file, see Setup)
backend/
  config.py
  main.py
  rag.py
  utils/
    fileRetrieve.py
frontend/
  index.html
```

## Features

- **FastAPI Backend**: Handles data processing, embedding generation, and retrieval logic.
- **ChromaDB**: Used as the vector store for storing and retrieving document chunks.
- **OpenAI Embeddings**: Generates embeddings for text chunks.
- **Configurable**: Key parameters like chunk size, overlap, dataset name, and ChromaDB collection name are centralized in `backend/config.py`.
- **Environment Variables**: API keys are securely loaded via `.env` file.
- **Logging**: Robust logging implemented across backend components.
- **Interactive Frontend**: A simple HTML/JavaScript interface to submit queries and display retrieved results.

## Setup

Follow these steps to set up and run the project:

### 1. Create `.env` File

Create a file named `.env` in the project's root directory and add your OpenAI API key:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

(Optional) For using mock embeddings during development, you can add:

```
MOCK_EMBEDDINGS="true"
```

### 2. Create `requirements.txt` File

Create a file named `requirements.txt` in the project's root directory with the following content:

```
openai
pandas
chromadb
tiktoken
fastapi
uvicorn
chardet
numpy
python-dotenv
nltk
```

### 3. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Stopwords (if re-enabled)

If you choose to re-enable stopword removal in `backend/rag.py` (which was removed in a previous step), you will need to download the NLTK stopwords corpus. You can do this by running a Python interpreter and executing:

```python
import nltk
nltk.download('stopwords')
```

### 5. Run the FastAPI Backend

Navigate to the `backend` directory in your terminal and start the FastAPI application:

```bash
cd backend
uvicorn main:app --reload
```

The backend server will typically run on `http://127.0.0.1:8000`.

### 6. Access the Frontend

Open the `frontend/index.html` file in your web browser. You can then type your queries into the input field and see the retrieved context chunks, their IDs, and similarity scores.

## Configuration

Key configuration parameters are located in `backend/config.py`:

- `chunk_size`: Size of text chunks.
- `chunk_overlap`: Overlap between text chunks.
- `batch_size`: Number of chunks to embed in a single batch.
- `dataset_name`: Name of the CSV dataset file.
- `chromadb_collection_name`: Name of the ChromaDB collection.
- `top_k`: Number of top similar chunks to retrieve.
- `similarity_threshold`: Minimum similarity score for a chunk to be considered relevant.

## Further Improvements

Refer to the `IMPROVEMENTS.md` file (if present) for a detailed list of potential improvements and suggestions for further enhancements.
