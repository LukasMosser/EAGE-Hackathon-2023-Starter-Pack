import os
import dotenv
import time
from pathlib import Path
from tqdm.auto import tqdm
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load local environment variables, keys, deployments, etc.
dotenv.load_dotenv()

# Set some paths for getting the data and storing the vector db
DATA_DIR = Path("./data/Annual 2023_proceedings for Hackathon")
CHUNK_VECTORDB_PATH = Path("data/vectordb/eage_annual_2023_chunks_basic_test")

# Text splitter for creating chunks - using some overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100, allowed_special="all"
)

# Create a base embeddings api object
embeddings = OpenAIEmbeddings(
    deployment=os.getenv("ADA_002_DEPLOYMENT_NAME", None), chunk_size=1
)

# Initialize a non-existant db to allow for the rate limit preventing pattern with sleep
db = None

# Iterate over all the files
for idx, path in enumerate(tqdm(os.listdir(DATA_DIR))):

    # Load the current PDF
    loader = PyPDFLoader(str(DATA_DIR / path))
    doc = loader.load()

    # Split the doc into chunks
    split_docs = text_splitter.split_documents(doc)

    if db is None:
        # If no db has been created yet, create the db
        db = FAISS.from_documents(split_docs, embedding=embeddings)
    else:
        # Else just add documents chunks
        db.add_documents(split_docs)

    # Sleep for a bit to prevent rate limiting
    time.sleep(2.0)

# Save Vector DB to local disk
db.save_local(CHUNK_VECTORDB_PATH)

# Do a check to ensure that we are able to reload from disk
db2 = FAISS.load_local(CHUNK_VECTORDB_PATH, embeddings)
print(db2.similarity_search("What are common issues with CO2 storage?"))
