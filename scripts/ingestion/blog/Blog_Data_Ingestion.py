import time
import blog_url_reader
import json
import os
import re
import nest_asyncio
import asyncio
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import AsyncGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode

load_dotenv()
blog_rss_items = blog_url_reader.Extract_RSS_Blog_Links()
blog_items = blog_url_reader.Extract_Blog_Urls()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
api_key = os.getenv("QDRANT_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
url = os.getenv("QDRANT_URL")
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key,
)


# Timing decorator to measure function execution time
def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds to execute.")
        return result

    return wrapper


@time_function
def read_filecoin_blogs():
    filecoin_blogitems = blog_rss_items.Filecoin_Blog()
    filtered_filecoin_blogs = blog_rss_items.Filecoin_Blog_Filter(filecoin_blogitems)
    filecoin_metadata = json.loads(filtered_filecoin_blogs)
    filecoin_blog_urls = [data['Source'] for data in filecoin_metadata['Data']]
    print(filecoin_blog_urls)
    return filecoin_blog_urls, filecoin_metadata


@time_function
def read_gitcoin_blogs():
    return blog_items.Gitcoin_Blog()

@time_function
def read_email_gitcoin_blogs():
    return blog_items.Email_Gitcoin_Blogs()

@time_function
def read_huddle01_blogs():
    return blog_items.Huddle01_Blog()


@time_function
def read_funding_commons_blogs():
    return blog_items.FundingComoons_Blog()

@time_function
def read_email_funding_commons_blogs():
    return blog_items.Email_Funding_The_Commons_Blogs()

@time_function
def read_libp2p_blogs():
    return blog_items.Libp2p_Blog()


# @time_function
# def read_lighthouse_blogs():
#     return blog_items.LightHouse_Blog()


@time_function
def read_drand_blogs():
    return blog_items.Drand_Blog()


@time_function
def read_tableland_blogs():
    return blog_items.Tableland_Blog()


@time_function
def read_desci_blogs():
    return blog_items.Desci_Blog()


@time_function
def read_oso_blogs():
    return blog_items.OSO_Blog()


@time_function
def read_dsa_blogs():
    return blog_items.DSA_Blog()


@time_function
def read_molecule_blogs():
    return blog_items.Molecule_Blog()

@time_function
def read_email_molecule_blogs():
    return blog_items.Email_Molecule_Blogs()

@time_function
def read_textile_blogs():
    return blog_items.Textile_Blog()


@time_function
def read_ipfs_blogs():
    ipfs_blogitems = blog_rss_items.IPFS_Blog()
    filtered_ipfs_blogs = blog_rss_items.IPFS_Blog_Filter(ipfs_blogitems)
    ipfs_metadata = json.loads(filtered_ipfs_blogs)
    ipfs_blog_urls = [data['Source'] for data in ipfs_metadata['Data']]
    print(ipfs_blog_urls)
    return ipfs_blog_urls, ipfs_metadata


@time_function
# def read_protocol_labs_blogs():
#     protocol_blogitems=blog_rss_items.Protocol_Labs_Blog()
#     result = blog_rss_items.Protocol_Labs_Blog_Filter(protocol_blogitems)
#     print(result)
#     protocol_labs_metadata = json.loads(result)
#
#     if "Error" in protocol_labs_metadata:
#         print("Error in retrieving blogs:", protocol_labs_metadata["Error"])
#         return [], protocol_labs_metadata  # Return empty list and metadata with error
#
#     protocol_labs_blog_urls = [data['Source'] for data in protocol_labs_metadata['Data']]
#     print(protocol_labs_blog_urls)
#     return protocol_labs_blog_urls, protocol_labs_metadata



async def load_single_url(url):
    loader = WebBaseLoader(url)
    loader.requests_per_second = 1
    docs = loader.aload()
    return docs




def extract_date_from_string(input_string):
    match = re.search(r'(\b\w+ \d{1,2}, \d{4})', input_string)
    date=match.group(0)
    date=re.sub('Blog','',date)
    date_obj = datetime.strptime(date, '%B %d, %Y')
    formatted_date = date_obj.strftime('%a, %d %b %Y 04:00:00 +0000')
    return formatted_date if match else None


def extract_chars_before_share(context):
    share_index = context.find('Share')
    if share_index != -1:
        start_index = max(0, share_index - 12)
        substring_before_share = context[start_index:share_index]
        substring_before_share=str(substring_before_share)
        print(substring_before_share)
        cleaned_date_string = re.sub(r'^[^\d]*(\w{3} \d{1,2}, \d{4})', r'\1',substring_before_share)
        date_object = datetime.strptime(cleaned_date_string , '%b %d, %Y')
        formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
        return formatted_date
    else:
        return None

def extract_chars_before_mint(context):
    print("hurry burry...")
    share_index = context.find('Mint')
    if share_index != -1:
        start_index = max(0, share_index - 19)
        substring_before_share = context[start_index:share_index]
        substring_before_share=str(substring_before_share)
        print(substring_before_share)
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        parts=substring_before_share.split(" ")
        if parts[0] in months:
            cleaned_date_string = re.sub(r'^[^\d]*(\w{3} \d{1,2}, \d{4})', r'\1', substring_before_share)
            cleaned_date_string = re.sub(r'(\d{1,2})(st|nd|rd|th)', r'\1', cleaned_date_string)
            date_object = datetime.strptime(cleaned_date_string, '%B %d, %Y')
            formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
            return formatted_date



def extract_from_tablelan(text, keyword='tabledan.eth'):
    # Find the index of the keyword
    index = text.find(keyword)
    if index != -1:
        result = text[index:index + 36]
        result=re.sub(keyword,"",result)
        # date_pattern = r'(\b\w+\s+\d{1,2}(?:th|st|nd|rd)?,?\s+\d{4})'
        date_pattern = r'(\b\w+\s?\d{1,2}(?:th|st|nd|rd)?,?\s+\d{4})'
        match = re.search(date_pattern, result)
        if match:
            print("right out here-----",match.group(1))
            date_str = match.group(1).replace("th", "").replace("st", "").replace("nd", "").replace("rd", "").strip()
            date_object = datetime.strptime(date_str, '%B %d, %Y')
            formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
            print(formatted_date)
            return formatted_date



    else:
        formatted_date = extract_chars_before_mint(text)
        print(formatted_date)
        return formatted_date


def correct_clipped_month(date_string,requestor):
    # List of correct month names
    if requestor=='molecule' or requestor=='tableland':
        v=0
    else:
        v=1
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # Check if the month is clipped
    for month in months:
        if date_string.startswith(month[v:]):  # Compare the string without the first letter
            # Replace the clipped month with the full month name
            return month + date_string[len(month) - 1:]  # Append the rest of the string

    return date_string


def expand_month(date_string):
    month_full_mapping = {
        "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "May": "May",
        "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September", "Oct": "October",
        "Nov": "November", "Dec": "December"
    }
    date_string=date_string.strip()
    abbreviated_month = date_string[:3]
    if abbreviated_month in month_full_mapping:
        full_month_name = month_full_mapping[abbreviated_month]
        return date_string.replace(abbreviated_month, full_month_name, 1)
    return date_string


def format_date(expanded_date_string):
    try:
        date_object = datetime.strptime(expanded_date_string, '%B %d, %Y')
        formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
        return formatted_date
    except ValueError:
        raise ValueError(f"Date format does not match the expected formats for: {expanded_date_string}")


def extract_textile_date(context):
    share_index = context.find('Share')
    if share_index != -1:
        start_index = max(0, share_index - 12)
        substring_before_share = context[start_index:share_index]
        substring_before_share=str(substring_before_share)
        substring=correct_clipped_month(substring_before_share,"textile")
        date_object = datetime.strptime(substring, '%B %d, %Y')
        formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
        return formatted_date
    else:
        return None


def modify_date(doc):
    date_formats = ['%B %d, %Y', '%d %B %Y', '%b %d, %Y', '%d %b %Y']
    date_string = doc  # Assuming doc contains the extracted date string
    for date_format in date_formats:
        try:
            date_object = datetime.strptime(date_string, date_format)
            formatted_date = date_object.strftime('%a, %d %b %Y %H:%M:%S +0000')

            print("Date extracted:", formatted_date)

            return formatted_date
        except ValueError:
            continue
    raise ValueError(f"Date format not recognized: {date_string}")

def extract_date(doc):
    print(doc.metadata['source'])
    context = doc.page_content
    if "gitcoin.co" in doc.metadata['source']:
        print("gitcoin blog")
        # Regex pattern to match "Blog" followed by any characters until "by"
        match = re.search(r"(Blog.*?by)", context)
        if match:
            extracted_text = match.group(0)  # Extract the matching text
            print(f"Extracted text: '{extracted_text.strip()}'")  # Debugging: Print matched string
            date = extract_date_from_string(extracted_text.strip())
            print(f"Extracted date: '{date}'")  # Debugging: Print the extracted date
            doc.metadata['Publication_Date']=date
            print("*" * 120)
        else:
            print("No match found after 'Blog'")
    elif "huddle01.com" in doc.metadata['source']:
        date=extract_chars_before_share(doc.page_content)
        doc.metadata['Publication_Date'] = date
        print(date)
    elif "https://mirror.xyz/tableland.eth" in doc.metadata['source']:
        date= extract_from_tablelan(context)
        doc.metadata['Publication_Date'] = date
        print(date)
    elif "https://blog.textile.io" in doc.metadata['source']:
        date=extract_textile_date(context)
        doc.metadata['Publication_Date'] = date
        print(date)


    else:
        # Date patterns to search for
        date_patterns = [
            r'(?:\D|^)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',  # Matches dates like 12/05/2021, 05-12-2021
            r'(?:\D|^)(\d{1,2}[ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{4})\b',
            r'(?:\D|^)(\d{1,2}[ ](?:January|February|March|April|May|June|July|August|September|October|November|December)[ ]\d{4})\b',
            r'(?:\D|^)(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[ ]\d{1,2},[ ]\d{4})\b',
            r'(?:\D|^)(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)[ ]\d{1,2},[ ]\d{4})\b',
            r'(?:\D|^)(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)[ ]\d{1,2}(?: st|nd|rd|th)?[, ]\d{4})\b'
        ]

        # Search for dates that match any of the patterns
        for pattern in date_patterns:
            match = re.search(pattern, context)
            if match:
                # Get the matched date
                date_string = match.group(0)
                if "https://blog.libp2p.io" in doc.metadata['source']:
                    cleaned_date_string = date_string.strip()
                    parts=cleaned_date_string.split(" ")
                    months = [
                        "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December",
                    ]
                    temp = ""
                    if parts[1] in months:
                        print("great ur here...", date_string, parts[1])
                        temp = parts[1]
                        parts[1] = parts[0]
                        parts[0] = temp
                        cleaned_date_string = parts[0] + ' ' + parts[1] + ',' + ' ' + parts[2]
                        print(cleaned_date_string)

                    if (len(parts[0])==3) and (parts[0] not in months):
                        print('lllll----------',cleaned_date_string)
                        cleaned_date_string=expand_month(cleaned_date_string)

                    date_object = datetime.strptime(cleaned_date_string, '%B %d, %Y')
                    formatted_date_string = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
                    print(formatted_date_string)
                if "https://www.molecule.xyz/blog" in doc.metadata['source']:
                    date_string=str(date_string.strip())
                    cleaned_date_string=re.sub("•","",date_string)
                    cleaned_date_string=expand_month(cleaned_date_string)
                    formatted_date_string = format_date(cleaned_date_string)
                    doc.metadata['Publication_Date'] = formatted_date_string
                    print(formatted_date_string)
                if "https://desci.com/blog" in doc.metadata['source']:
                    date_string = date_string.strip()
                    formatted_date=modify_date(date_string)
                    doc.metadata['Publication_Date'] = formatted_date
                    print(formatted_date)


                if "https://docs.drand.love/blog" in doc.metadata['source']:
                    date_string = str(date_string.strip())
                    parts=date_string.split(" ")
                    months = [
                        "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December",
                        'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'
                    ]


                    if parts[0]  in months and len(parts[0])==3:
                        date_string=expand_month(date_string)
                    elif parts[0] in months:
                        date_string=date_string
                    else:
                        date_string=str(date_string[1:])
                    cleaned_date_string=correct_clipped_month(date_string,"drand")
                    date_object = datetime.strptime(cleaned_date_string, '%B %d, %Y')
                    formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
                    doc.metadata['Publication_Date'] = formatted_date
                    print(formatted_date)

                if "https://docs.opensource.observer/blog" in doc.metadata['source']:
                    date_string = str(date_string.strip())
                    print(f'date string is {date_string}')
                    parts = date_string.split(" ")
                    months = [
                        "January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December",
                         ]
                    temp = ""
                    if parts[1] in months:
                        print("great ur here...", date_string, parts[1])
                        temp=parts[1]
                        parts[1] = parts[0]
                        parts[0] = temp
                        date_string=parts[0]+' '+parts[1]+','+' '+parts[2]
                        print(date_string)

                    if parts[0] in months:
                        date_string=date_string
                    else:
                        date_string=str(date_string[1:])

                    date_object = datetime.strptime(date_string, '%B %d, %Y')
                    formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
                    doc.metadata['Publication_Date'] = formatted_date
                    print(formatted_date)

                if "https://dsalliance.io" in doc.metadata['source']:
                    date_string=expand_month(date_string)
                    print(date_string)
                    parts=date_string.split(' ')
                    # date_string=expand_month(date_string)
                    date_object = datetime.strptime(date_string, '%B %d, %Y')
                    formatted_date = date_object.strftime('%a, %d %b %Y 04:00:00 +0000')
                    doc.metadata['Publication_Date'] = formatted_date
                    print(formatted_date)
                return doc  #

    return doc


async def read_url(urls):
    nest_asyncio.apply()
    tasks = [load_single_url(url) for url in urls]
    docs_list = await asyncio.gather(*tasks)
    docs = [doc for docs in docs_list for doc in docs]
    return docs

@time_function
def data_cleaning(docs):
    for doc in docs:
        doc=extract_date(doc)
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content.replace('\n', '').strip())
        doc.page_content = re.sub(r'[",\\]', '', doc.page_content)
    return docs


@time_function
def replace_metadata(docs, metadata):
    for i, doc in enumerate(docs):
        if i < len(metadata['Data']):
            doc.metadata = metadata['Data'][i]
    return docs


@time_function
async def create_vector_qdrant(docs):
    collection_name = "Blogs_OpenAI_Dense"
    # embeddings = HuggingFaceEmbeddings(model_name="jinaai/jina-embeddings-v2-small-en", model_kwargs={'device': 'cpu'})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    qdrant = QdrantVectorStore.from_documents(
        texts,
        embedding=embeddings,
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        timeout=270
    )

    print("Dumped successfully with metadata schema...")
    return qdrant

@time_function
async def llm_response(context):
    client = AsyncGroq()
    stream = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in providing detailed and relevant responses based on the context and a citation expert. "
                    "Please provide detailed and elaborate responses using citations inside text wherever appropriate without fail. "
                    "Please provide all unique sources based on which you generated the response. "
                    "Sources should appear after the response is complete under the heading Sources. They should be unique "
                    "and should not repeat. If any source ends with 'None', remove it."
                )
            },
            {
                "role": "user",
                "content": context,
            }
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.01,
        max_tokens=4096,
        top_p=1,
        stop=None,
        stream=True,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content, end="")


async def read_urls_from_rss():
    filecoin_blog_urls, filecoin_metadata = read_filecoin_blogs()
    filecoin_docs = await read_url(filecoin_blog_urls)
    filecoin_docs = replace_metadata(filecoin_docs, filecoin_metadata)
    ipfs_blog_urls, ipfs_metadata = read_ipfs_blogs()
    ipfs_docs = await read_url(ipfs_blog_urls)
    ipfs_docs = replace_metadata(ipfs_docs, ipfs_metadata)
    # protocol_labs_blog_urls, protocol_labs_metadata = read_protocol_labs_blogs()
    # protocol_labs_docs = await read_url(protocol_labs_blog_urls)
    # protocol_labs_docs = replace_metadata(protocol_labs_docs, protocol_labs_metadata)
    return filecoin_docs, ipfs_docs


async def read_urls_from_csv():
    gitcoin_blog_urls = read_gitcoin_blogs()
    gitcoin_docs = await read_url(gitcoin_blog_urls)
    huddle01_blog_urls = list(set(read_huddle01_blogs()))
    huddle01_docs = await read_url(huddle01_blog_urls)
    libp2p_blog_urls = list(set(read_libp2p_blogs()))
    libp2p_docs = await read_url(libp2p_blog_urls)
    funding_commons_blog_urls = list(set(read_funding_commons_blogs()))
    funding_commons_docs = await read_url(funding_commons_blog_urls)
    tableland_blog_urls = list(set(read_tableland_blogs()))
    tableland_docs = await read_url(tableland_blog_urls)
    textile_blog_urls = list(set(read_textile_blogs()))
    textile_docs = await read_url(textile_blog_urls)
    molecule_blog_urls = list(set(read_molecule_blogs()))
    molecule_docs = await read_url(molecule_blog_urls)
    # light_house_blog_urls = list(set(read_lighthouse_blogs()))
    # light_house_docs = await read_url(light_house_blog_urls)
    desci_blog_urls = list(set(read_desci_blogs()))
    desci_docs = await read_url(desci_blog_urls)
    drand_blog_urls = list(set(read_drand_blogs()))
    drand_docs = await read_url(drand_blog_urls)
    oso_blog_urls = list(set(read_oso_blogs()))
    oso_docs = await read_url(oso_blog_urls)
    dsa_blog_urls = list(set(read_dsa_blogs()))
    dsa_docs = await read_url(dsa_blog_urls)
    return (gitcoin_docs, huddle01_docs, libp2p_docs, funding_commons_docs, tableland_docs, textile_docs,
            molecule_docs, desci_docs, drand_docs, oso_docs, dsa_docs)


async def run():
    all_docs = await read_urls_from_rss()
    # all_docs_csv = await read_urls_from_csv()
    # all_docs = [doc for sublist in all_docs for doc in sublist] + [doc for sublist in all_docs_csv for doc in sublist]
    all_docs=[doc for sublist in all_docs for doc in sublist]
    for doc in all_docs:
        doc.metadata['source'] = doc.metadata.pop('Source')
    cleaned_docs = data_cleaning(all_docs)
    print(cleaned_docs[0:10])
    qdrant = await create_vector_qdrant(cleaned_docs)


if __name__ == '__main__':
    asyncio.run(run())
