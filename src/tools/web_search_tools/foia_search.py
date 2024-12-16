import requests
from bs4 import BeautifulSoup
import random
from typing import List
from search_manager import WebContentExtractor
import logging

logger = logging.getLogger(__name__)

def foia_search(query: str) -> List[str]:
    """Searches FOIA.gov for the given query and returns a list of relevant content."""
    url = f"https://search.foia.gov/search?utf8=%E2%9C%93&m=true&affiliate=foia.gov&query={query.replace(' ', '+')}"
    web_content_extractor = WebContentExtractor()
    headers = {
        'User-Agent': random.choice(web_content_extractor.USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'DNT': '1',
    }
    try:
        response = requests.get(url, headers=headers, timeout=web_content_extractor.TIMEOUT)
        response.raise_for_status()
        html_content = response.content.decode('utf-8')
        soup = BeautifulSoup(html_content, 'html.parser')

        result_links = [a['href'] for a in soup.select('.result-title a') if a.has_attr('href')]

        content = []
        for link in result_links:
            try:
                if extracted_content := WebContentExtractor.extract_content(link):
                    content.append(extracted_content)
            except Exception as e:
                logger.error(f"Error extracting content from {link}: {e}")

        return content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching FOIA.gov: {e}")
        return []