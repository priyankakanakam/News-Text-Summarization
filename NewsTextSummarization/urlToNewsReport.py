from newspaper import Article

import requests
from bs4 import BeautifulSoup
import time


def extract_fromTimesofIndia(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text
    '''print("Title:", article.title)
    print("Publish Date:", article.publish_date)
    print("\nArticle Content:\n")
    print(article.text)
    print("\n")'''


def extract_news_other(url):
    try:
        # Add a user-agent header to mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Send a GET request to the URL with headers
        response = requests.get(url, headers=headers)

        # Check if the request was successful (status code 200)
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract news content based on the HTML structure of the page
        # Adjust the following code according to the structure of the webpage you're scraping
        news_content = ""
        news_elements = soup.find_all('p')  # Adjust this based on the HTML structure of the news content

        for element in news_elements:
            news_content += element.get_text() + "\n"

        return news_content
    except requests.exceptions.RequestException as e:
        print(f"Error: Unable to fetch content from {url}")
        print(e)
        return None

def extract_news(url):
    #print("hello")
    # Example usage:
    # url="https://www.ndtv.com/india-news/pm-modi-to-launch-developmental-projects-in-rajasthan-madhya-pradesh-today-4451200"
    # not working url="https://www.msn.com/en-in/sports/cricket/rohit-kohli-bumrah-return-12-players-out-complete-list-of-changes-in-team-india-for-test-series-against-sa/ar-AA1lS63g?ocid=msedgntp&pc=HCTS&cvid=e9af290a5c7d4517b58f0d3a466f31fe&ei=8"
    # url="https://timesofindia.indiatimes.com/india/22-cases-of-covids-jn-1-variant-detected-in-country-till-thursday/articleshow/106214926.cms"
    #url = "https://www.hindustantimes.com/india-news/delhi-rouse-avenue-court-grants-ed-custody-of-aap-mp-sanjay-singh-till-oct-10-101696510055187.html"
    if (url.find("timesofindia")):
        news_text = extract_fromTimesofIndia(url)
        if news_text:
            print(news_text)
            return news_text
    else:
        news_text = extract_news_other(url)
        if news_text:
            print(news_text)
            return news_text

