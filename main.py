import os
import time

from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

os.environ["PINECONE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

def semantic_search(query, embeddings):
   llm = OpenAI()
   docs = embeddings.similarity_search(query)
   chain = load_qa_chain(llm, chain_type="stuff")
   return chain.invoke({"input_documents": docs, "question": query}, return_only_outputs=True)

def chunk_data(url):
    loader = OnlinePDFLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0
    )
    return text_splitter.split_documents(data)

def index_url(url):
   split_text = chunk_data(url)
   embeddings_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
   return PineconeVectorStore.from_documents(split_text, embeddings_model, index_name="langchain")

def pc_setup(index_name):
   pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
   existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
   if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

def setup():
  print("Bot: Welcome! Type 'exit' to quit. Please input a PDF url.")
  pdf_url = input()
  print("Bot: Now, please input an index name for your book.")
  index_name = input()

  pc_setup(index_name)
  split_text = chunk_data(pdf_url)
  embeddings_model = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
  embeddings_pc = PineconeVectorStore.from_documents(split_text, embeddings_model, index_name=index_name)
  
  print("Bot: I am all set up, ask away!")

  return embeddings_pc

def chatbot():
  
  embeddings = setup()

  while True:
    user_input = input().strip().lower()
    if user_input == 'exit':
        print("Bot: Bye!")
        break
    else:
       query = user_input
       response = semantic_search(query, embeddings)
       print(f"Bot: {response['output_text']}")       
  

if __name__ == "__main__":
   chatbot()
