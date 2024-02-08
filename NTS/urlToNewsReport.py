from newspaper import Article
import requests
from bs4 import BeautifulSoup

def extract_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if(len(article.text)==0):
            return None
        return article.text
    except requests.exceptions.RequestException as e:
        return Exception

def extract_contentbbc(url):

        # Fetch HTML content from the URL
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

            # Find and extract text content from the HTML
        paragraphs = soup.find_all('p')  # Assuming the main content is in <p> tags
        content = ' '.join([paragraph.get_text() for paragraph in paragraphs])
        if(len(content)==0):
            return None
        return content
    except requests.exceptions.RequestException as e:
        return Exception

# Example usage:

def extract_news(url):
    if(url.find('bbc.com')>-1):
        news_text=extract_contentbbc(url)
    else:
        news_text=extract_content(url)
    return news_text
