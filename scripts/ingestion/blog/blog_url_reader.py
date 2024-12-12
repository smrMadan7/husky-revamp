import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import json
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import re
import nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import os
load_dotenv()



class Extract_RSS_Blog_Links():
    def Filecoin_Blog(self):
        url = "https://filecoin.io/blog/feed/index.xml"
        response = requests.get(url)
        xml_content = response.text
        return xml_content

    def Filecoin_Blog_Filter(self,content_string):
        BASE_URL = "https://filecoin.io"
        root = ET.fromstring(content_string)
        channel = root.find('channel')
        today = datetime.utcnow().replace(tzinfo=pytz.UTC)
        three_months_ago = today - relativedelta(months=3)
        filtered_items=[]

        for item in channel.findall('item'):
            pub_date_str = item.find('pubDate').text

            pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
            if pub_date >= three_months_ago:
                Title = item.find('title').text
                Link = item.find('link').text
                Guid = item.find('guid').text
                Description = item.find('description').text
                if Link.startswith(BASE_URL):
                    Link = Link[len(BASE_URL):]

                filtered_items.append({
                    "Title": Title,
                    "Source": Link,
                    "Publication_Date": pub_date_str,
                    "Guid": Guid,
                    "Description": Description
                })

        return json.dumps({"Data": filtered_items}, indent=4)




    def IPFS_Blog(self):
        url = "https://blog.ipfs.tech/index.xml"
        response = requests.get(url)
        xml_content = response.text
        return xml_content

    def IPFS_Blog_Filter(self,content_string):
        root = ET.fromstring(content_string)
        channel = root.find('channel')
        today = datetime.utcnow().replace(tzinfo=pytz.UTC)
        five_months_ago = today - relativedelta(months=9)
        filtered_items=[]
        for item in channel.findall('item'):
            pub_date_str = item.find('pubDate').text
            pub_date_str = pub_date_str.replace('GMT', '+0000')
            pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
            if pub_date >= five_months_ago:
                Title = item.find('title').text
                Link = item.find('link').text
                Guid = item.find('guid').text
                Description = item.find('description').text
                filtered_items.append({
                    "Title": Title,
                    "Source": Link,
                    "Publication_Date": pub_date_str,
                    "Guid": Guid,
                    "Description": Description
                })
        return json.dumps({"Data": filtered_items}, indent=4)

    def Protocol_Labs_Blog(self):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        url = "https://www.protocol.ai/rss.xml"
        response = requests.get(url, headers=headers)
        xml_content = response.text
        print(xml_content)
        return xml_content

    def Protocol_Labs_Blog_Filter(self,content_string):
        try:
            # Parse the XML content
            root = ET.fromstring(content_string)

            # Set the date range (today and five months ago)
            today = datetime.utcnow().replace(tzinfo=pytz.UTC)
            five_months_ago = today - relativedelta(months=5)
            filtered_items = []

            # Find the channel in the XML
            channel = root.find('channel')
            if channel is None:
                print("Error: Channel element not found.")
                return json.dumps({"Data": [], "Error": "Channel element missing from XML"}, indent=4)

            # Process each item within the channel
            for item in channel.findall('item'):
                title = item.find('title').text.strip() if item.find('title') is not None else "No Title"
                link = item.find('link').text.strip() if item.find('link') is not None else "No Link"
                guid = item.find('guid').text.strip() if item.find('guid') is not None else "No GUID"
                description = item.find('description').text.strip() if item.find(
                    'description') is not None else "No Description"
                pub_date_str = item.find('pubDate').text.strip().replace('GMT', '+0000') if item.find(
                    'pubDate') is not None else None

                # Parse publication date
                if pub_date_str:
                    pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
                else:
                    print("Warning: Missing or invalid pubDate for item; skipping.")
                    continue  # Skip item if pubDate is invalid

                # Filter items by date
                if pub_date >= five_months_ago:
                    # Retrieve enclosure URL if it exists
                    enclosure = item.find('enclosure')
                    enclosure_url = enclosure.get('url') if enclosure is not None else "No Enclosure URL"

                    # Append item to filtered list
                    filtered_items.append({
                        "Title": title,
                        "Source": link,
                        "Publication_Date": pub_date_str,
                        "Guid": guid,
                        "Description": description,
                        "Enclosure_URL": enclosure_url
                    })

            # Return filtered items in JSON format
            return json.dumps({"Data": filtered_items}, indent=4)

        except ET.ParseError as e:
            # Detailed error in case of parsing failure
            print("XML parsing error at root level:", e)
            return json.dumps({"Data": [], "Error": "Failed to parse XML content due to invalid structure or encoding"},
                              indent=4)


class Email_Reader:

    def __init__(self, username, password):
        self.imap_server = imaplib.IMAP4_SSL("imap.gmail.com")
        self.imap_server.login(username, password)

    def extract_urls(self,text):
        url_pattern = re.compile(r'(https?://[^\s]+)')
        urls = url_pattern.findall(text)
        return urls

    def get_redirected_url(self,link):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service('/home/ubuntu/Documents/PLN/husky-revamp/utils/chromedriver')
        driver = webdriver.Chrome(service=service, options=chrome_options)
        original_urls = []
        try:
            for url in link:
                driver.get(url)
                time.sleep(5)
                final_url = driver.current_url
                original_urls.append(final_url)
            return original_urls
        finally:
            driver.quit()

    def fetch_team_emails(self,team_email):
        """Fetches emails from the specified team email address."""
        self.imap_server.select("inbox")
        status, messages = self.imap_server.search(None, f'(FROM "{team_email}")')
        email_ids = messages[0].split()
        emails = []

        for email_id in email_ids:
            res, msg = self.imap_server.fetch(email_id, "(RFC822)")
            for response_part in msg:
                if isinstance(response_part, tuple):
                    email_data = self.parse_email(response_part[1])  # Pass only the bytes content
                    emails.append(email_data)
        return emails

    def parse_email(self, email_content):
        msg = email.message_from_bytes(email_content)
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else "utf-8")
        received_date = msg.get("Date")
        parsed_date = None
        if received_date:
            cleaned_date = received_date.replace(" (UTC)", "")
            try:
                # Parse the cleaned date string into a datetime object
                parsed_date = datetime.strptime(cleaned_date, "%a, %d %b %Y %H:%M:%S %z")
                print(f"Received Date: {parsed_date}")
            except ValueError as e:
                print(f"Error parsing date: {e}")
        body, urls = self.extract_body_and_urls(msg)
        return {
            "subject": subject,
            "received_date": parsed_date,
            "body": body,
            "urls": urls
        }

    def extract_body_and_urls(self, msg):
        urls = []
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" not in content_disposition:
                    if content_type == "text/plain":
                        body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        urls.extend(self.extract_urls(body))
                    elif content_type == "text/html":
                        html_body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                        clean_body = BeautifulSoup(html_body, "html.parser").get_text()
                        urls.extend(self.extract_urls(clean_body))
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")
            urls.extend(self.extract_urls(body))
        urls = list(set(urls))
        return body, urls


class Extract_Blog_Urls:
    def __init__(self):
        mail=os.getenv("Mail_Box")
        pass_word=os.getenv("Pass")
        self.gmail_reader = Email_Reader(mail,pass_word)

    def Gitcoin_Blog(self):
        df = pd.read_csv('./Data/Blogs/Gitcoin.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Desci_Blog(self):
        df = pd.read_csv('./Data/Blogs/Desci.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Drand_Blog(self):
        df = pd.read_csv('./Data/Blogs/Drand.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def DSA_Blog(self):
        df = pd.read_csv('./Data/Blogs/DSA.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def FundingComoons_Blog(self):
        df = pd.read_csv('./Data/Blogs/FundingCommons.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Huddle01_Blog(self):
        df = pd.read_csv('./Data/Blogs/Huddle01.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Libp2p_Blog(self):
        df = pd.read_csv('./Data/Blogs/Libp2p.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def LightHouse_Blog(self):
        df = pd.read_csv('./Data/Blogs/LightHouse.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Molecule_Blog(self):
        df = pd.read_csv('./Data/Blogs/Molecule.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def OSO_Blog(self):
        df = pd.read_csv('./Data/Blogs/OSO.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Tableland_Blog(self):
        df = pd.read_csv('./Data/Blogs/Tableland.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Textile_Blog(self):
        df = pd.read_csv('./Data/Blogs/Textile.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Email_Gitcoin_Blogs(self):
        emails = self.gmail_reader.fetch_team_emails("team@gitcoin.co")
        gitcoin_urls = []
        for email_data in tqdm(emails):
            date = email_data["received_date"]
            urls = self.gmail_reader.get_redirected_url(email_data['urls'])
            print(date)
            original_urls = [u for u in urls if "https://www.gitcoin.co/blog" in u]
            print("URLs:", original_urls)
            gitcoin_urls.append(original_urls)
            print("*" * 50)
        gitcoin_blog_list = [item for sublist in gitcoin_urls for item in sublist]
        print(gitcoin_blog_list)
        return gitcoin_blog_list

    def Email_Molecule_Blogs(self):
        emails = self.gmail_reader.fetch_team_emails("athenadao@substack.com")
        molecule_urls = []
        for email_data in tqdm(emails):
            date = email_data["received_date"]
            urls = self.gmail_reader.get_redirected_url(email_data['urls'])
            print(date)
            original_urls = [u for u in urls if "https://www.molecule.xyz/blog" in u]
            print("URLs:", original_urls)
            molecule_urls.append(original_urls)
            print("*" * 50)
        molecule_blog_list = [item for sublist in molecule_urls for item in sublist]
        return molecule_blog_list

    def Email_Funding_The_Commons_Blogs(self):
        emails = self.gmail_reader.fetch_team_emails("contact@fundingthecommons.io")
        funding_the_commons_urls = []
        for email_data in tqdm(emails):
            date = email_data["received_date"]
            urls = self.gmail_reader.get_redirected_url(email_data['urls'])
            print(date)
            funding_the_commons_urls.append(urls)
            print("*" * 50)
        funding_the_commons_blog_list = [item for sublist in funding_the_commons_urls for item in sublist]
        print(funding_the_commons_blog_list)
        return funding_the_commons_blog_list



urls=Extract_Blog_Urls()
print(urls.Email_Gitcoin_Blogs())




