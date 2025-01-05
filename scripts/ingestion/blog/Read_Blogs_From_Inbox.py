import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
load_dotenv()


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
        chrome_options.add_argument("--disable-gpu")
        service = Service(ChromeDriverManager().install())
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

    def fetch_team_emails(self, team_email):
        """Fetches emails from the specified team email address within the last 4 months."""
        self.imap_server.select("inbox")

        # Calculate the date 4 months ago
        four_months_ago = (datetime.now() - timedelta(days=4 * 30)).strftime("%d-%b-%Y")

        # Search for emails from the team email within the last 4 months
        search_criteria = f'(FROM "{team_email}" SINCE {four_months_ago})'
        status, messages = self.imap_server.search(None, search_criteria)

        email_ids = messages[0].split()
        emails = []

        for email_id in email_ids:
            res, msg = self.imap_server.fetch(email_id, "(RFC822)")
            for response_part in msg:
                if isinstance(response_part, tuple):
                    email_data = self.parse_email(response_part[1])  # Pass only the bytes content
                    emails.append(email_data)
        return emails

    # def fetch_team_emails(self,team_email):
    #     """Fetches emails from the specified team email address."""
    #     self.imap_server.select("inbox")
    #     status, messages = self.imap_server.search(None, f'(FROM "{team_email}")')
    #     email_ids = messages[0].split()
    #     emails = []
    #
    #     for email_id in email_ids:
    #         res, msg = self.imap_server.fetch(email_id, "(RFC822)")
    #         for response_part in msg:
    #             if isinstance(response_part, tuple):
    #                 email_data = self.parse_email(response_part[1])  # Pass only the bytes content
    #                 emails.append(email_data)
    #     return emails

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
                # print(f"Received Date: {parsed_date}")
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


class Extract_Teams_Blogs():

    def __init__(self,username,pass_word):
        mail = os.getenv("Mail_Box")
        pass_word = os.getenv("Pass")
        self.gmail_reader = Email_Reader(mail, pass_word)

    def Email_Gitcoin_Blogs(self):
        emails = self.gmail_reader.fetch_team_emails("team@gitcoin.co")
        gitcoin_urls = []
        for email_data in tqdm(emails):
            date = email_data["received_date"]
            urls = self.gmail_reader.get_redirected_url(email_data['urls'])
            # print(date)
            original_urls = [u for u in urls if "https://www.gitcoin.co/blog" in u]
            # print("URLs:", original_urls)
            gitcoin_urls.append(original_urls)
            print("*" * 50)
        gitcoin_blog_list = [item for sublist in gitcoin_urls for item in sublist]
        print(f'gitcoin urls are {gitcoin_blog_list}')
        return gitcoin_blog_list

    def Email_Molecule_Blogs(self):
        print("welcome to molecule")
        emails = self.gmail_reader.fetch_team_emails("athenadao@substack.com")
        molecule_urls = []
        for email_data in tqdm(emails):
            date = email_data["received_date"]
            urls = self.gmail_reader.get_redirected_url(email_data['urls'])
            # print(date)
            original_urls = [u for u in urls if "https://www.molecule.xyz/blog" in u]
            # print("URLs:", original_urls)
            molecule_urls.append(original_urls)
            print("*" * 50)
        molecule_blog_list = [item for sublist in molecule_urls for item in sublist]
        return molecule_blog_list

    def Email_Funding_The_Commons_Blogs(self):
        print("welcome to funding")
        emails = self.gmail_reader.fetch_team_emails("contact@fundingthecommons.io")
        funding_the_commons_urls = []
        for email_data in tqdm(emails):
            date = email_data["received_date"]
            urls = self.gmail_reader.get_redirected_url(email_data['urls'])
            # print(date)
            funding_the_commons_urls.append(urls)
            print("*" * 50)
        funding_the_commons_blog_list = [item for sublist in funding_the_commons_urls for item in sublist]
        return funding_the_commons_blog_list










username=os.getenv("Mail_Box")
password=os.getenv("Pass")
inbox_reader=Extract_Teams_Blogs(username, password)

gitcoin_urls=inbox_reader.Email_Gitcoin_Blogs()
print(gitcoin_urls)

molecule_urls=inbox_reader.Email_Molecule_Blogs()
print(molecule_urls)

funding_common_urls=inbox_reader.Email_Funding_The_Commons_Blogs()
print(funding_common_urls)


