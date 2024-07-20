from dotenv import load_dotenv
import os
from getpass import getpass
from google.cloud import aiplatform
from langchain_google_vertexai import ChatVertexAI
import fitz
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import chromadb
#from chromadb.utils import embedding_function_to_chroma_function
from langchain_community.vectorstores import Chroma
import pickle
# Load environment variables from .env file
load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
google_cloud_project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_token

# For langchain
LANGCHAIN_TRACING_V2 = True

# aiplatform
aiplatform.init(project=google_cloud_project_id)

#gemini
#os.environ["GOOGLE_API_KEY"] = "AIzaSyC86pcKxm7voDn8Sz9RMwspiFDV6Qtj2I4"
llm = ChatVertexAI(model="gemini-pro") #hope this works

#huggingface
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Vhf_BbtMKxDSCPMhbcUgNkOQNGvvGcYGKvywEF"

pdf_path = r"/Users/unnatisaraswat/Desktop/random_sentence_generator/Copy of Placement Chronicles 2023-24.pdf"

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    document = fitz.open(pdf_path)
    text = ""

    # Iterate through each page
    for page_num in range(len(document)):
        page = document.load_page(page_num)  # Load the page
        text += page.get_text("text")  # Extract text with 'text' layout

    return text

raw_text=extract_text_from_pdf(pdf_path)

# Create a Document object
document = Document(page_content=raw_text, metadata={'source': 'Placement Chronicles 2023-24'})

# List of documents
documents = [document]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(documents)

#for testing
#print(all_splits[100])

#create embeddings
embedding_function = HuggingFaceBgeEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",
    model_kwargs = {'device':'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)


# Create collection


# Add documents to the collection with embeddings

#db = Chroma.from_documents(all_splits, embedding_function)

db2 = Chroma.from_documents(all_splits, embedding_function,persist_directory="./chroma_db",  )
#db2.persist()
#docs = db2.similarity_search(query)
###embeddings save ho gye###




