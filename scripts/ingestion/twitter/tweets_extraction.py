import json
import logging
import os
import time
from argparse import ArgumentParser
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
load_dotenv()
TWITTER_AUTH_TOKEN = os.getenv("TWITTER_AUTH_TOKEN")


class TwitterExtractor:
    def __init__(self, member_name, folder_path, headless=True):
        """
        Initializes the TwitterExtractor instance.

        Parameters:
            member_name (str): The name of the Twitter member to extract data from.
            folder_path (str): The folder path where the extracted data will be saved.
            headless (bool): If True, runs the browser in headless mode. Default is True.
        """
        self.driver = self._start_chrome(headless)
        self.set_token()
        self.member_name = member_name
        self.folder_path = folder_path

    def _start_chrome(self, headless):
        """
        Starts the Chrome WebDriver with the specified options.

        Parameters:
            headless (bool): If True, runs Chrome in headless mode.

        Returns:
            webdriver.Chrome: The Chrome WebDriver instance.
        """
        options = Options()
        options.headless = headless
        driver = webdriver.Chrome(options=options)
        driver.get("https://twitter.com")
        return driver

    def set_token(self, auth_token=TWITTER_AUTH_TOKEN):
        """
        Sets the Twitter authentication token as a cookie in the browser.

        Parameters:
            auth_token (str): The Twitter authentication token to be set.
                              Default is fetched from the environment variable.

        Raises:
            ValueError: If the auth_token is missing or improperly configured.
        """
        if not auth_token or auth_token == "YOUR_TWITTER_AUTH_TOKEN_HERE":
            raise ValueError("Access token is missing. Please configure it properly.")
        expiration = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        cookie_script = f"document.cookie = 'auth_token={auth_token}; expires={expiration}; path=/';"
        self.driver.execute_script(cookie_script)

    def _wait_for_tweets_to_load(self):
        """Waits for tweets to be loaded on the page."""
        WebDriverWait(self.driver, 15).until(EC.presence_of_element_located((By.XPATH, "//article[@data-testid='tweet']")))

    def fetch_tweets(self, page_url, start_date, end_date):
        """
        Fetches tweets from the specified Twitter page within a date range.

        Parameters:
            page_url (str): The URL of the Twitter page to fetch tweets from.
            start_date (str): The start date for filtering tweets (format: YYYY-MM-DD).
            end_date (str): The end date for filtering tweets (format: YYYY-MM-DD).
        """
        count = 0
        try:
            self.driver.get(page_url)
            cur_filename = f"{self.folder_path}/{self.member_name}_tweets_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            print("cur file name------------------->", cur_filename)
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            while True:
                tweet_element = self._get_first_ttweet()
                if not tweet_element:
                    continue
                row = self._process_tweet(tweet_element)
                if row["date"]:
                    try:
                        date = datetime.strptime(row["date"], "%Y-%m-%d")
                    except ValueError as e:
                        # infer date format
                        logger.info(
                            f"Value error on date format, trying another format.{row['date']}",
                            e,
                        )
                        date = datetime.strptime(row["date"], "%d/%m/%Y")
                    if date < start_date:
                        count = count + 1
                        if count > 10:
                            break
                    elif date > end_date:
                        self._delete_first_tweet()
                        continue
                print(f"The extracted row is ------------>{row}")
                self._save_to_json(row, filename=f"{cur_filename}.json")
                logger.info(f"Saving tweets...\n{row['date']},  {row['author_name']} -- {row['text'][:50]}...\n\n")
                self._delete_first_tweet()
            self._save_to_excel(json_filename=f"{cur_filename}.json", output_filename=f"{cur_filename}.xlsx")
            self.fix_json_file(filename=f"{cur_filename}.json", output_filename=f"{cur_filename}.json")
        except TimeoutException:
            logger.error(f"Timeout while trying to load the page {page_url}.")
        except NoSuchElementException:
            logger.error(f"Page {page_url} does not exist or elements could not be found.")
        except Exception as e:
            logger.error(f"An error occurred while fetching tweets from {page_url}: {e}")
        finally:
            self.close()

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(TimeoutException),
    )
    def _get_first_ttweet(self, timeout=5, use_hacky_workaround_for_reloading_issue=True):
        """
        Retrieves the first tweet element on the page.

        Parameters:
            timeout (int): The maximum time to wait for the tweet to load.

        Returns:
            tuple: A tuple containing the tweet element and its stats element.

        Raises:
            TimeoutException: If the tweet does not load within the specified time.
            NoSuchElementException: If the tweet or 'Retry' button is not found.
        """
        try:
            WebDriverWait(self.driver, timeout).until(EC.presence_of_element_located((By.XPATH, "//article[@data-testid='tweet']")))
            tweet_element = self.driver.find_element(By.XPATH, "//article[@data-testid='tweet']")
            stats_element = tweet_element.find_element(By.XPATH, ".//div[@role='group']")
            return tweet_element, stats_element
        except TimeoutException:
            logger.error("Timeout waiting for tweet or after clicking 'Retry'")
            raise
        except NoSuchElementException:
            logger.error("Could not find tweet or 'Retry' button")
            raise

    def _process_tweet(self, tweet_data):
        """
        Processes the tweet data to extract relevant information.

        Parameters:
            tweet_data (tuple): A tuple containing the tweet element and its stats element.

        Returns:
            dict: A dictionary containing the processed tweet data.
        """
        tweet_element, stats_element = tweet_data
        author_name, author_handle = self._extract_author_details(tweet_element)
        try:
            data = {
                "text": self._get_element_text(tweet_element, ".//div[@data-testid='tweetText']"),
                "author_name": author_name,
                "author_handle": author_handle,
                "date": self._get_element_attribute(tweet_element, "time", "datetime")[:10],
                "lang": self._get_element_attribute(tweet_element, "div[data-testid='tweetText']", "lang"),
                "url": self._get_tweet_url(tweet_element),
                "mentioned_urls": self._get_mentioned_urls(tweet_element),
                "is_retweet": self.is_retweet(tweet_element),
                "media_type": self._get_media_type(tweet_element),
                "images_urls": (self._get_images_urls(tweet_element) if self._get_media_type(tweet_element) == "Image" else None),
            }
        except Exception as e:
            logger.error(f"Error processing tweet: {e}")
            logger.info(f"Tweet: {tweet_element}")
            raise
        interaction_counts = self._extract_interaction_counts(stats_element)
        data.update(interaction_counts)
        if data["date"]:
            data["date"] = datetime.strptime(data["date"], "%Y-%m-%d").strftime("%Y-%m-%d")
        return data

    def _extract_interaction_counts(self, stat_element):
        """
        Extracts interaction counts (likes, retweets, replies) from the stats element.

        Parameters:
            stat_element: The stats element of the tweet.

        Returns:
            dict: A dictionary containing counts of replies, retweets, and likes.
        """
        try:
            aria_label = stat_element.get_attribute("aria-label")
            counts = {"replies": 0, "reposts": 0, "likes": 0}
            print("aria label----------------->", aria_label)

            for label in aria_label.split(","):
                if "likes" in label.lower() or "like" in label.lower():
                    counts["likes"] = int(label.split()[0].replace(",", ""))
                elif "replies" in label.lower() or "reply" in label.lower():
                    counts["replies"] = int(label.split()[0].replace(",", ""))
                elif "retweet" in label.lower() or "reposts" in label.lower() or "repost" in label.lower():
                    counts["reposts"] = int(label.split()[0].replace(",", ""))
            return {"num_reply": counts["replies"], "num_retweet": counts["reposts"], "num_like": counts["likes"]}
        except (NoSuchElementException, ValueError):
            return {"num_reply": 0, "num_retweet": 0, "num_like": 0}

    def _get_element_text(self, parent, selector):
        """
        Retrieves the text content from an element identified by an XPath expression.

        Parameters:
            parent_element: The parent element to search within.
            xpath (str): The XPath expression to locate the target element.

        Returns:
            str: The text content of the identified element.
        """
        try:
            return parent.find_element(By.XPATH, selector).text
        except NoSuchElementException:
            return ""

    def _get_element_attribute(self, parent, selector, attribute):
        """
        Retrieves an attribute's value from an element identified by an XPath expression.

        Parameters:
            parent_element: The parent element to search within.
            xpath (str): The XPath expression to locate the target element.
            attr (str): The attribute whose value needs to be retrieved.

        Returns:
            str: The value of the specified attribute.
        """
        try:
            return parent.find_element(By.CSS_SELECTOR, selector).get_attribute(attribute)
        except NoSuchElementException:
            return ""

    def _get_mentioned_urls(self, tweet):
        """
        Retrieves URLs mentioned in the tweet.

        Parameters:
            tweet_element: The tweet element from which to extract URLs.

        Returns:
            list: A list of mentioned URLs in the tweet.
        """
        try:
            link_elements = tweet.find_elements(By.XPATH, ".//a[contains(@href, 'http')]")
            urls = [elem.get_attribute("href") for elem in link_elements]
            return urls
        except NoSuchElementException:
            return []

    def is_retweet(self, tweet):
        """
        Determines whether the tweet is a retweet.

        Parameters:
            tweet_element: The tweet element to be checked.

        Returns:
            bool: True if the tweet is a retweet, otherwise False.
        """
        try:
            retweet_indicator = tweet.find_element(By.XPATH, ".//div[contains(text(), 'Retweeted')]")
            if retweet_indicator:
                return True
        except NoSuchElementException:
            return False

    def _get_tweet_url(self, tweet):
        """
        Constructs the URL for the tweet.

        Parameters:
            tweet_element: The tweet element from which to construct the URL.

        Returns:
            str: The constructed URL of the tweet.
        """
        try:
            link_element = tweet.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
            return link_element.get_attribute("href")
        except NoSuchElementException:
            return ""

    def _extract_author_details(self, tweet):
        """
        Extracts the author name and handle from a given tweet element.

        This method retrieves the text of the author from a tweet and splits it
        into the author's name and handle. The expected format is that the
        author's name and handle are separated by a newline character.

        Parameters:
            tweet (WebElement): The tweet element from which to extract author details.

        Returns:
            tuple: A tuple containing:
                - author_name (str): The name of the author.
                - author_handle (str): The Twitter handle of the author. If not available, it returns an empty string.

        Example:
            author_name, author_handle = self._extract_author_details(tweet)
        """
        author_details = self._get_element_text(tweet, ".//div[@data-testid='User-Name']")
        parts = author_details.split("\n")
        if len(parts) >= 2:
            author_name = parts[0]
            author_handle = parts[1]
        else:
            author_name = author_details
            author_handle = ""

        return author_name, author_handle

    def _get_media_type(self, tweet):
        """
        Determines the media type of the tweet (e.g., image, video).

        Parameters:
            tweet_element: The tweet element to analyze.

        Returns:
            str: The media type, if any, associated with the tweet.
        """
        if tweet.find_elements(By.CSS_SELECTOR, "div[data-testid='videoPlayer']"):
            return "Video"
        if tweet.find_elements(By.CSS_SELECTOR, "div[data-testid='tweetPhoto']"):
            return "Image"
        return "No media"

    def _get_images_urls(self, tweet):
        """
        Retrieves URLs of images in the tweet.

        Parameters:
            tweet_element: The tweet element to analyze.

        Returns:
            list: A list of image URLs associated with the tweet.
        """
        images_urls = []
        images_elements = tweet.find_elements(By.XPATH, ".//img[@alt and not(@alt='Image')]")
        for img_element in images_elements:
            images_urls.append(img_element.get_attribute("src"))
        return images_urls

    def _delete_first_tweet(self):
        """
        Deletes the first tweet element from the page.
        """
        try:
            first_tweet = self.driver.find_element(By.XPATH, "//article[@data-testid='tweet']")
            self.driver.execute_script("arguments[0].remove();", first_tweet)
        except NoSuchElementException:
            logger.error("Error deleting first tweet element")

    @staticmethod
    def _save_to_json(data, filename="data.json"):
        """
        Saves the extracted data to a JSON file.

        Parameters:
            data (dict): The data to save.
            filename (str): The name of the file where the data will be saved.
        """
        with open(filename, "a", encoding="utf-8") as file:
            json.dump(data, file)
            file.write("\n")

    @staticmethod
    def fix_json_file(filename="data.json", output_filename="fixed_data.json"):
        """
        Fixes the JSON file by ensuring it is properly formatted.

        Parameters:
            filename (str): The name of the input JSON file.
            output_filename (str): The name of the output JSON file.
        """
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
        data_list = []
        for line in lines:
            data_list.append(json.loads(line))
        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump({"twitter_data": data_list}, file, ensure_ascii=False, indent=4)

    @staticmethod
    def _save_to_excel(json_filename, output_filename="data/data.xlsx"):
        """
        Converts the JSON data to an Excel file.

        Parameters:
            json_filename (str): The name of the input JSON file.
            output_filename (str): The name of the output Excel file.
        """
        cur_df = pd.read_json(json_filename, lines=True)
        cur_df.drop_duplicates(subset=["url"], inplace=True)
        cur_df.to_excel(output_filename, index=False)
        logger.info(f"\n\nDone saving to {output_filename}. Total of {len(cur_df)} unique tweets.")

    def extract_followers(self, url):
        """
        Extracts the followers of a Twitter account.

        Parameters:
            url (str): The URL of the Twitter profile from which to extract followers.

        Returns:
            list: A list of usernames of the followers.

        Raises:
            TimeoutException: If the followers timeline does not load within the wait time.
        """
        follower_url = f"{url}/followers"
        self.driver.get(follower_url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Timeline: Followers"]')))
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        followers = []
        elements = self.driver.find_elements(By.XPATH, '//div[@aria-label="Timeline: Followers"]//a[@role="link"]')
        for a in elements:
            username = a.text
            if username and "search" not in username:
                followers.append(username)
                print(username)
        return followers

    def extract_following(self, url):
        """
        Extracts the accounts that a Twitter user is following.

        Parameters:
            url (str): The URL of the Twitter profile from which to extract the following accounts.

        Returns:
            list: A list of usernames of the accounts being followed.

        Raises:
            TimeoutException: If the following timeline does not load within the wait time.
        """
        following_url = f"{url}/following"
        self.driver.get(following_url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Timeline: Following"]')))
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        following = []
        elements = self.driver.find_elements(By.XPATH, '//div[@aria-label="Timeline: Following"]//a[@role="link"]')
        for a in elements:
            username = a.text
            if username and "search" not in username:
                following.append(username)
                print(username)
        return following

    def extract_verified_followers(self, url):
        """
        Extracts the verified followers of a Twitter account.

        Parameters:
            url (str): The URL of the Twitter profile from which to extract verified followers.

        Returns:
            list: A list of usernames of the verified followers.

        Raises:
            TimeoutException: If the verified followers timeline does not load within the wait time.
        """
        following_url = f"{url}/verified_followers"
        self.driver.get(following_url)
        WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//div[@aria-label="Timeline: Verified Followers"]')))
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            time.sleep(2)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        verified_followers = []
        elements = self.driver.find_elements(By.XPATH, '//div[@aria-label="Timeline: Verified Followers"]//a[@role="link"]')
        for a in elements:
            username = a.text
            if username and "search" not in username:
                verified_followers.append(username)
                print(username)
        return verified_followers

    def close(self):
        """
        Closes the browser and cleans up resources.
        """
        self.driver.quit()


def process_team(team_name, team_handle, page_url, folder_path, start_date, end_date):
    """
    Processes Twitter data for a given team by fetching tweets, followers, and following information.

    Parameters:
        team_name (str): The name of the team.
        team_handle (str): The Twitter handle of the team.
        page_url (str): The URL of the team's Twitter profile page.
        folder_path (str): The path to the folder where output files will be saved.
        start_date (str): The start date for fetching tweets in YYYY-MM-DD format.
        end_date (str): The end date for fetching tweets in YYYY-MM-DD format.

    Returns:
        None: This function saves the results in a CSV file and does not return any value.

    Prints:
        - The list of followers for the team.
        - The list of accounts the team is following.
        - The list of verified followers for the team.
        - The total connections for the team, which combines strong connections and verified followers.
    """
    extractor = TwitterExtractor(member_name=team_name, folder_path=folder_path)
    extractor.fetch_tweets(page_url=page_url, start_date=start_date, end_date=end_date)

    followers = extractor.extract_followers(page_url)
    print(f"Followers for {team_name}: {followers}")

    following = extractor.extract_following(page_url)
    print(f"Following for {team_name}: {following}")

    strong_connections = list(set(followers).intersection(set(following)))
    verified_followers = extractor.extract_verified_followers(page_url)
    print(f"Verified Followers for {team_name}: {verified_followers}")

    filter_connections = verified_followers + strong_connections
    print(f"Total Connections for {team_name}: {filter_connections}")

    df = pd.DataFrame(filter_connections)
    filename = f"{team_name}.csv"
    df.to_csv(os.path.join(folder_path, filename), index=False)


def main(args):
    """
    Main function to execute the Twitter data extraction process for teams.

    This function parses command line arguments, reads a CSV file containing team Twitter handles,
    and processes each team by calling the `process_team` function.

    Command Line Arguments:
        file_path (str): Path to the CSV file containing team Twitter handles.
        output_folder (str): Output folder to save team CSV files.
        --start_date (str): Optional; Start date for fetching tweets in YYYY-MM-DD format (default: "2024-07-01").
        --end_date (str): Optional; End date for fetching tweets in YYYY-MM-DD format (default: "2024-10-24").

    Returns:
        None: This function does not return any value but triggers the data extraction process.
    """



    df = pd.read_csv(args.input_file_path)
    folder_path = args.output_folder
    start_date = args.start_date
    end_date = args.end_date

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        team_name = row[0]
        team_handle = row[1]
        page_url = row[2]

        process_team(team_name, team_handle, page_url, folder_path, start_date, end_date)


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Twitter data for teams")
    parser.add_argument("--input_file_path", help="Path to the CSV file containing team twitter handles")
    parser.add_argument("--output_folder", help="Output folder to save team CSV files")
    parser.add_argument("--start_date", type=str, default="2024-07-01",
                        help="Start date for fetching tweets (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-10-24", help="End date for fetching tweets (YYYY-MM-DD)")
    args = parser.parse_args()
    main(args)

#run the code using the similar command as shown below. python script.py input file path output folder path start date and end date
# python tweets_extraction.py --input_file_path /home/ubuntu/Downloads/husky-be-v1/Data/Twitter_Teams.csv --output_folder /home/ubuntu/Downloads/husky-be-v1/Data/temp_twitter --start_date 2024-07-01 --end_date 2024-10-24
