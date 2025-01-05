import xml.etree.ElementTree as ET
from dateutil.relativedelta import relativedelta
import pytz
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
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
        months_ago = today - relativedelta(months=3)
        filtered_items=[]

        for item in channel.findall('item'):
            pub_date_str = item.find('pubDate').text

            pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
            if pub_date >= months_ago:
                Title = item.find('title').text
                Link = item.find('link').text
                Guid = item.find('guid').text
                Description = item.find('description').text
                if Link.startswith(BASE_URL):
                    Link = Link[len(BASE_URL):]

                filtered_items.append({
                    "Title": Title,
                    "source": f"https://filecoin.io{Link}",
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
        months_ago = today - relativedelta(months=9)
        filtered_items=[]
        for item in channel.findall('item'):
            pub_date_str = item.find('pubDate').text
            pub_date_str = pub_date_str.replace('GMT', '+0000')
            pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
            if pub_date >= months_ago:
                Title = item.find('title').text
                Link = item.find('link').text
                Guid = item.find('guid').text
                Description = item.find('description').text
                filtered_items.append({
                    "Title": Title,
                    "source": Link,
                    "Publication_Date": pub_date_str,
                    "Guid": Guid,
                    "Description": Description
                })
        return json.dumps({"Data": filtered_items}, indent=4)

    def Protocol_Labs_Blog(self):
        url = "https://www.protocol.ai/rss.xml"
        response = requests.get(url)
        xml_content = response.text
        return xml_content


    def Protocol_Labs_Blog_Filter(self,content_string):
        try:
            root = ET.fromstring(content_string)

            # Set the date range (today and five months ago)
            today = datetime.utcnow().replace(tzinfo=pytz.UTC)
            months_ago = today - relativedelta(months=8)
            filtered_items = []

            # Find the channel in the XML
            channel = root.find('channel')
            if channel is None:
                return json.dumps({"Data": [], "Error": "Channel element missing from XML"}, indent=4)

            # Process each item within the channel
            for item in channel.findall('item'):
                # Helper function to extract and clean text
                def extract_text(element, tag, default=""):
                    elem = element.find(tag)
                    return elem.text.strip() if elem is not None and elem.text else default

                title = extract_text(item, 'title', "No Title")
                link = extract_text(item, 'link', "No Link")
                guid = extract_text(item, 'guid', "No GUID")
                description = extract_text(item, 'description', "No Description")
                pub_date_str = extract_text(item, 'pubDate', None)

                if pub_date_str:
                    pub_date_str = pub_date_str.replace('GMT', '+0000')
                    try:
                        pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %z')
                    except ValueError:
                        continue  # Skip items with invalid pubDate format
                else:
                    continue  # Skip items without pubDate
        #         # Filter items by date
                if pub_date >= months_ago:
                    # Add to filtered items
                    filtered_items.append({
                        "Title": title,
                        "source": link,
                        "Publication_Date": pub_date.isoformat(),
                        "GUID": guid,
                        "Description": description,

                    })
            return json.dumps({"Data": filtered_items}, indent=4)
        except ET.ParseError as e:
            return json.dumps({"Data": [], "Error": f"Failed to parse XML. Details: {e}"}, indent=4)
        except Exception as e:
            return json.dumps({"Data": [], "Error": str(e)}, indent=4)


rss_reader=Extract_RSS_Blog_Links()
filecoin_xml=rss_reader.Filecoin_Blog()
filecoin_urls=rss_reader.Filecoin_Blog_Filter(filecoin_xml)
print(filecoin_urls)
ipfs_xml=rss_reader.IPFS_Blog()
ipfs_urls=rss_reader.IPFS_Blog_Filter(ipfs_xml)
print(ipfs_urls)
pl_xml=rss_reader.Protocol_Labs_Blog()
pl_urls=rss_reader.Protocol_Labs_Blog_Filter(pl_xml)
print(pl_urls)
