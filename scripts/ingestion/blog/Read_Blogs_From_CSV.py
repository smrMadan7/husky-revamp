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


class Read_Blogs_CSV():
    def Gitcoin_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Gitcoin.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Desci_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Desci.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Drand_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Drand.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def DSA_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/DSA.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def FundingComoons_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/FundingCommons.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Huddle01_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Huddle01.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Libp2p_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Libp2p.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def LightHouse_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/LightHouse.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Molecule_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Molecule.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def OSO_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/OSO.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Tableland_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Tableland.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls

    def Textile_Blog(self):
        df = pd.read_csv('/home/ubuntu/Documents/PLN/husky-revamp/Data/Blogs/Textile.csv')
        urls = df['Blog_Urls'].values.tolist()
        urls = list(set(urls))
        return urls



