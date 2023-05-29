import os
import dotenv
import openai
from pathlib import Path
from tqdm.auto import tqdm
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.schema import Document

# Load local environment variables, keys, deployments, etc.
dotenv.load_dotenv()

# Set some paths for getting the data and storing the vector db
DATA_DIR = Path("./data/Annual 2023_proceedings for Hackathon")
SUMMARY_DIR = Path("./data/summaries")
SUMMARY_VECTORDB_PATH = Path("data/vectordb/eage_annual_2023_summaries_basic_test")

llm = OpenAI(
    deployment_id=os.getenv("DAVINCI_003_DEPLOYMENT_NAME", None), temperature=0
)

summary_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
summarize_document_chain = AnalyzeDocumentChain(
    combine_docs_chain=summary_chain, verbose=False
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

    # Join all the pages of the doc into a single document
    raw_text = " ".join([d.page_content for d in doc])
    full_doc = Document(
        page_content=raw_text, metadata={"source": doc[0].metadata["source"]}
    )

    try:
        # Run the summarization chain
        summary = summarize_document_chain.run([full_doc.page_content])

        # Create a new summary doc
        summary_doc = Document(
            page_content=summary, metadata={"source": doc[0].metadata["source"]}
        )

        # Write the intermediary result to json (optional)
        with open(SUMMARY_DIR / str(path.split(".")[0] + ".json"), "w") as f:
            f.write(summary_doc.json())

        if db is None:
            # If no db has been created yet, create the db
            db = FAISS.from_documents([summary_doc], embedding=embeddings)
        else:
            # Else just add the new generated summary
            db.add_documents([summary_doc])

    except (openai.error.InvalidRequestError, openai.error.APIError) as error:
        print(error)
        print(path)
        continue


# Save Vector DB to local disk
db.save_local(SUMMARY_VECTORDB_PATH)

# Do a check to ensure that we are able to reload from disk
db2 = FAISS.load_local(SUMMARY_VECTORDB_PATH, embeddings)
print(db2.similarity_search("What are common issues with CO2 storage?"))
