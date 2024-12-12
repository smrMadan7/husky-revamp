import asyncio
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader,Docx2txtLoader,UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import re
import nest_asyncio
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from groq import Groq
from datetime import datetime
import requests
from langchain_qdrant import FastEmbedSparse, RetrievalMode

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("QDRANT_KEY")
url = os.getenv('QDRANT_URL')
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key,
)


def find_event_date(context):
    client = Groq()
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "Identify and provide only the event start date mentioned in the context. If an event date is explicitly mentioned, provide only that date. "
                       "If no event date is mentioned, extract and provide only the first occurring date in the context. "
                       "If there are multiple dates provide only the first occuring date"
                       "Respond with the date alone in 'Month Day' or 'Month Day' format without any additional text."
                       "If Month is in abbreviated from please expand and provide the response"
                       "Example Oct 5 should be October 5"
        },
        {
            "role": "user",
            "content": context,
        }
    ],
    model="gemma2-9b-it",
    temperature=0.01,
    max_tokens=7000,
    top_p=1,
    stop=None,
    stream=False,
)

    # Print the completion returned by the LLM.
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content



from datetime import datetime

def convert_to_yyyy_mm_dd(date_string):
    # Remove any leading/trailing whitespace from the date string
    date_string = date_string.strip()

    if date_string == 'February 29':
        date_string = "February 28 2024"
        parsed_date = datetime.strptime(date_string, "%B %d %Y")
        formatted_date = parsed_date.strftime("%Y-%m-%d")
        return formatted_date

    # Define possible date formats, including "30 April" format
    date_formats = ["%B %d, %Y", "%B %d", "%d %B, %Y", "%d %B", "%B", "%B %d %Y", "%B %Y"]

    # Attempt to parse the date string with each format
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_string, fmt)
            # If year is missing, assume current year
            if "%Y" not in fmt:
                try:
                    parsed_date = parsed_date.replace(year=datetime.now().year)
                    # Check if parsed date is valid for the current year (e.g., February 29)
                    parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    parsed_date = parsed_date.replace(year=2024)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Raise an error if no format matches
    raise ValueError(f"Date format for '{date_string}' not recognized.")



def read_urls(urls):
    nest_asyncio.apply()
    loader = WebBaseLoader(urls)
    loader.requests_per_second = 1
    data = loader.aload()  # Await the coroutine
    return data

def create_collection_if_not_exists(collection_name):
    if not qdrant_client.collection_exists(collection_name=collection_name):
        # Specify the dimensions of your embeddings
        dimensions = 384  # Update this to match the embedding model you're using
        qdrant_client.create_collection(
            collection_name=collection_name,
            vector_size=dimensions,  # Vector size based on your model
            distance="Cosine"  # Or another distance metric, e.g., "Euclidean"
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")


async def create_vector_qdrant(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    # embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en", model_kwargs={'device': 'cpu'})
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")

    qdrant = QdrantVectorStore.from_documents(
        texts,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        url=url,
        api_key=api_key,
        collection_name="Events_OpenAI",
        retrieval_mode=RetrievalMode.HYBRID,
        timeout=90
    )
    print("Dumped successfully...")
    return qdrant


def data_cleaning(docs):
    for doc in docs:
        print(doc.metadata['source'])
        print("oooooooooooooo")
        doc.page_content = doc.page_content.replace('\n', '')
        doc.page_content = re.sub(r'[",\\]', '', doc.page_content)
        doc.page_content = re.sub(r"[\"']", '', doc.page_content)
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()


        if "lu.ma" in doc.metadata['source']:
            print("enter")
            url_source=doc.metadata['source'].split('/')
            event=url_source[-1]
            url="https://api.lu.ma/url?url="+event
            response = requests.get(url)
            print(response.status_code)
            if response.status_code == 200:
                data = response.json()  # Parse the JSON response
                date=data["data"]["start_at"]
                formatted_date = datetime.fromisoformat(date.replace("Z", "")).strftime("%Y-%m-%d")
                print("Formatted date:", formatted_date)
                doc.metadata['event_date']=formatted_date

        elif "libp2p_meeting_notes" in doc.metadata['source']:
            doc.page_content = doc.page_content.replace('\n', '')
            doc.page_content = re.sub(r'[",\\]', '', doc.page_content)
            doc.page_content = re.sub(r"[\"']", '', doc.page_content)
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

        elif "Helia" in doc.metadata['source']:
            print("pppppppppp")
            doc.page_content = doc.page_content.replace('\n', '')
            doc.page_content = re.sub(r'[",\\]', '', doc.page_content)
            doc.page_content = re.sub(r"[\"']", '', doc.page_content)
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()

        #
        else:
            date=find_event_date(doc.page_content)
            print(date)
            date=(re.sub("The date mentioned in the context is:","",date)).strip()
            date = (re.sub("Here is the date mentioned in the context:", "", date)).strip()

            if "-" in date:
                date_string=date.split("-")

                date=date_string[0]
                print("hhhh--",date)
            if "to" in date:
                date_string = date.split("to")[0].strip()
                date_string = date_string[0]
            if "August" not in date:
                date=re.sub(r'nd|st|th|rd','',date)
            parts=date.split(" ")
            if len(parts[0])==3 and parts[0]!='May':
                month_full_mapping = {
                    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May",
                    "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October",
                    "Nov": "November", "Dec": "December"
                }

                abbreviated_month = parts[0]
                if abbreviated_month in month_full_mapping:
                    full_month_name = month_full_mapping[abbreviated_month]
                    date=full_month_name+" "+parts[1]
                    print(date)
            date=re.sub("'",'',date)
            date = re.sub(",", '', date)
            print(f'date before {date}')
            if doc.metadata['source']=="https://fil-brussels.io/":
                date='July 8'
            if doc.metadata['source']=="https://devcon.org/en/":
                date='November 12'

            date=convert_to_yyyy_mm_dd(date)
            print(date)
            doc.metadata['event_date']=date

    return docs

def read_libp2p_docs(folder_path):
    docx_files = [f for f in os.listdir(folder_path) if f.endswith('.docx')]
    documents = []
    for file in docx_files:
        file_path = os.path.join(folder_path, file)
        # Load each .docx file using Docx2TxtLoader
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    return documents

def read_ipfs_docs(folder_path):
    docx_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
    documents = []
    for file in docx_files:
        file_path = os.path.join(folder_path, file)
        loader = UnstructuredHTMLLoader(file_path)
        documents.extend(loader.load())
    print(documents[0].page_content)
    return documents




async def main():
    df = pd.read_csv('./Data/Events/Event_Urls_Without_Luma.csv')
    urls = df['Event_Urls'].values.tolist()
    urls=list(set(urls))
    luma_urls=[event for event in urls if "lu.ma" in event]
    filtered_urls = [event for event in urls if "docs.google" not in event and "lu.ma" not in event]
    docs = read_urls(filtered_urls)
    docs1=read_urls(luma_urls)
    documents1 = data_cleaning(docs1)
    documents = data_cleaning(docs)
    libp2p_meeting_docs=read_libp2p_docs('/home/ubuntu/Downloads/husky-be-v1/Data/Events/libp2p_meeting_notes')
    libp2p_docs=data_cleaning(libp2p_meeting_docs)
    ipfs_meeting_docs=read_ipfs_docs('/home/ubuntu/Downloads/husky-be-v1/Data/Events/ipfs_meeting_notes')
    ipfs_docs=data_cleaning(ipfs_meeting_docs)
    all_docs=ipfs_docs+libp2p_docs+documents1+documents
    qdrant = await create_vector_qdrant(all_docs)


if __name__ == "__main__":
    asyncio.run(main())
