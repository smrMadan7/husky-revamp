import asyncio
import pandas as pd
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, UnstructuredHTMLLoader
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
from logger_configuration import logger


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
openai_api_key = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("QDRANT_KEY")
url = os.getenv('QDRANT_URL')

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key,
)


def find_event_date(context):
    try:
        client = Groq()
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Identify and provide only the event start date mentioned in the context. "
                               "If an event date is explicitly mentioned, provide only that date. "
                               "If no event date is mentioned, extract and provide only the first occurring date in the context. "
                               "If there are multiple dates provide only the first occurring date. "
                               "Respond with the date alone in 'Month Day' or 'Month Day' format without any additional text. "
                               "Expand abbreviated months like Oct to October."
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
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in find_event_date: {e}")
        return None


def convert_to_yyyy_mm_dd(date_string):
    # Remove any leading/trailing whitespace from the date string
    date_string = date_string.strip()

    # Handle February 29 explicitly
    if date_string == 'February 29':
        date_string = "February 28 2024"
        parsed_date = datetime.strptime(date_string, "%B %d %Y")
        return parsed_date.strftime("%Y-%m-%d")

    # Remove ordinal indicators (st, nd, rd, th) using regex
    date_string = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_string)

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
    try:
        nest_asyncio.apply()
        loader = WebBaseLoader(urls,continue_on_failure=True)
        loader.requests_per_second = 1
        return loader.aload()
    except Exception as e:
        logger.error(f"Error in read_urls: {e}")
        return []


def create_collection_if_not_exists(collection_name, dimensions=384):
    try:
        if not qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.create_collection(
                collection_name=collection_name,
                vector_size=dimensions,
                distance="Cosine"
            )
            logger.info(f"Collection '{collection_name}' created.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Error in create_collection_if_not_exists: {e}")


async def create_vector_qdrant(docs):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        texts = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
        sparse_embeddings = FastEmbedSparse(model_name="Qdrant/BM25")

        QdrantVectorStore.from_documents(
            texts,
            embedding=embeddings,
            sparse_embedding=sparse_embeddings,
            url=url,
            api_key=api_key,
            collection_name="events",
            retrieval_mode=RetrievalMode.HYBRID,
            timeout=90
        )
        logger.info("Documents dumped to Qdrant successfully.")
    except Exception as e:
        logger.error(f"Error in create_vector_qdrant: {e}")


def data_cleaning(docs):
    try:
        for doc in docs:
            doc.page_content = re.sub(r'[",\\]', '', doc.page_content.replace('\n', ''))
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
    except Exception as e:
        logger.error(f"Error in data_cleaning: {e}")
        return []

def read_libp2p_docs(folder_path):
    try:
        docx_files = [f for f in os.listdir(folder_path) if f.endswith('.docx')]
        documents = []
        for file in docx_files:
            loader = Docx2txtLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
        return documents
    except Exception as e:
        logger.error(f"Error in read_libp2p_docs: {e}")
        return []


def read_ipfs_docs(folder_path):
    try:
        html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
        documents = []
        for file in html_files:
            loader = UnstructuredHTMLLoader(os.path.join(folder_path, file))
            documents.extend(loader.load())
        return documents
    except Exception as e:
        logger.error(f"Error in read_ipfs_docs: {e}")
        return []


async def main():
    try:
        df = pd.read_csv('./Data/Events/Event_Urls_Without_Luma.csv')
        urls = list(set(df['Event_Urls'].values.tolist()))
        luma_urls = [url for url in urls if "lu.ma" in url]
        filtered_urls = [url for url in urls if "docs.google" not in url and "lu.ma" not in url]

        docs = read_urls(filtered_urls)
        luma_docs = read_urls(luma_urls)

        all_docs = data_cleaning(docs + luma_docs)

        libp2p_docs = data_cleaning(read_libp2p_docs('/home/ubuntu/Downloads/husky-be-v1/Data/Events/libp2p_meeting_notes'))
        ipfs_docs = data_cleaning(read_ipfs_docs('/home/ubuntu/Downloads/husky-be-v1/Data/Events/ipfs_meeting_notes'))

        all_docs.extend(libp2p_docs + ipfs_docs)

        await create_vector_qdrant(all_docs)
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
