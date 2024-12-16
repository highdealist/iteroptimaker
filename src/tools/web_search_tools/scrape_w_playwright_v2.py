import asyncio
import logging
from typing import List, Dict, Optional, NamedTuple
from tenacity import retry, stop_after_attempt, wait_exponential
from playwright.async_api import async_playwright, Error as PlaywrightError
import csv
from dataclasses import dataclass
import yaml
from aiohttp import ClientSession, ClientError
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import random
from functools import wraps
import time
import rate_limiter
from rate_limiter import AsyncLimiter

retries=3
timeout=30
results_per_page=5  # Add this if not present
pages_to_scrape=5
rate_limit=1.0

from urllib.parse import urlparse





logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProxyConfig:
    server: str
    port: int
    type: str
    failures: int = 0
    last_used: float = 0

@dataclass
class ScraperConfig:
    query: str
    retries: int
    timeout: int
    proxy_configs: List[ProxyConfig]
    pages_to_scrape: int
    results_per_page: int
    rate_limit: float

class SearchResult(NamedTuple):
    title: str
    url: str

def singleton(cls):
    instances = {}
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def load_config(file_path: str) -> ScraperConfig:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        proxies = []
        for p in config['proxy_configs']:
            parsed_proxy = urlparse(p)
            if parsed_proxy.scheme and parsed_proxy.hostname and parsed_proxy.port:
                proxies.append(ProxyConfig(server=parsed_proxy.hostname, port=parsed_proxy.port, type=parsed_proxy.scheme))
            else:
                logger.warning(f"Invalid proxy format: {p}. Skipping.")
        return ScraperConfig(
            query=config['query'],
            retries=config['retries'],
            timeout=config['timeout'],
            proxy_configs=proxies,
            pages_to_scrape=config.get('pages_to_scrape', 5),
            results_per_page=config.get('results_per_page', 5),
            rate_limit=config.get('rate_limit', 1.0)
        )
    except FileNotFoundError:
        logger.error(f"Config file not found: {file_path}")
        raise
    except yaml.YAMLError:
        logger.error(f"Invalid YAML in config file: {file_path}")
        raise
    except KeyError as e:
        logger.error(f"Missing required key in config: {str(e)}")
        raise
     
@singleton
class UserAgentRotator:
    def __init__(self):
        self.ua = UserAgent()
        self.user_agents = [self.ua.chrome, self.ua.firefox, self.ua.safari, self.ua.edge]

    def get_random_user_agent(self):
        return random.choice(self.user_agents)

ua_rotator = UserAgentRotator()

class ProxyManager:
    def __init__(self, proxies: List[ProxyConfig]):
        self.proxies = proxies
        self.current_index = 0

    def get_next_proxy(self) -> ProxyConfig:
        proxy = self.proxies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.proxies)
        return proxy

    def mark_proxy_failure(self, proxy: ProxyConfig):
        proxy.failures += 1
        if proxy.failures > 3:
            self.proxies.remove(proxy)
            logger.warning(f"Removed failing proxy: {proxy.server}:{proxy.port}")

    def mark_proxy_success(self, proxy: ProxyConfig):
        proxy.failures = 0
        proxy.last_used = time.time()

def fetch_with_proxy(ua_rotator: UserAgentRotator, proxy_manager: ProxyManager, rate_limiter: AsyncLimiter, **kwargs) -> Optional[str]:
    url = kwargs['url']
    timeout = kwargs['timeout']

    for _ in range(3):  # Try up to 3 different proxies
        proxy = proxy_manager.get_next_proxy()
        try:
            with rate_limiter:
                with async_playwright() as p:
                    browser = p.chromium.launch(
                        proxy={"server": f"{proxy.type}://{proxy.server}:{proxy.port}"},
                        headless=random.choice([True, False])
                    )
                    with browser.new_context(
                        user_agent=ua_rotator.get_random_user_agent(),
                        viewport={'width': random.randint(1024, 1920), 'height': random.randint(768, 1080)}
                    ) as context:
                        with context.new_page() as page:
                            simulate_human_behavior(page)
                            page.goto(url, timeout=timeout, wait_until='networkidle')
                            simulate_browsing(page)
                            content = page.content()
                            proxy_manager.mark_proxy_success(proxy)
                            return content
        except PlaywrightError as e:
            logger.error(f"Playwright error with proxy {proxy.server}: {str(e)}")
        except TimeoutError as e:
            logger.error(f"Timeout error with proxy {proxy.server}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with proxy {proxy.server}: {str(e)}")
        
        proxy_manager.mark_proxy_failure(proxy)

    logger.error(f"Failed to fetch {url} after trying multiple proxies")
    return None


def simulate_human_behavior(page) -> None:
    try:
        page.mouse.move(random.randint(0, 1920), random.randint(0, 1080))
        asyncio.sleep(random.uniform(2, 5))
    except Exception as e:
        logger.error(f"Error simulating human behavior: {str(e)}")

def simulate_browsing(page) -> None:
    try:
        asyncio.sleep(random.uniform(3, 7))
        for _ in range(random.randint(2, 5)):
            page.evaluate("window.scrollBy(0, Math.floor(Math.random() * window.innerHeight))")
            asyncio.sleep(random.uniform(1, 3))

            if random.random() < 0.3:
                page.mouse.click(random.randint(0, 1920), random.randint(0, 1080))
    except Exception as e:
        logger.error(f"Error simulating browsing: {str(e)}")

def parse_search_results(html_content: str) -> List[SearchResult]:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        for item in soup.select('div.g'):
            title_elem = item.select_one('h3')
            url_elem = item.select_one('div.yuRUbf > a')
            if title_elem and url_elem:
                title = title_elem.get_text()
                url = url_elem.get('href')
                if title and url:
                    results.append(SearchResult(title=title, url=url))
        return results
    except Exception as e:
        logger.error(f"Error parsing search results: {str(e)}")
        return []

def scrape_url(url: str, context) -> str:
    try:
        page = context.new_page()
        page.goto(url, wait_until='networkidle')
        simulate_browsing(page)
        content = page.content()
        page.close()
        return content
    except Exception as e:
        logger.error(f"Error scraping result page {url}: {str(e)}")
        return ""

def scrape_urls(content: str, context, results_per_page: int) -> List[Dict[str, str]]:
    results = parse_search_results(content)
    scraped_results = []
    # Use the results_per_page parameter instead of hardcoded 5
    for result in results[:10]:
        try:
            page_content = scrape_url(result.url, context)
            scraped_text = extract_text_from_html(page_content)
            scraped_results.append({
                'title': result.title,
                'url': result.url,
                'content': scraped_text
            })
        except Exception as e:
            logger.error(f"Error extracting and scraping result {result.url}: {str(e)}")
    return scraped_results


def extract_text_from_html(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        logger.error(f"Error extracting text from HTML: {str(e)}")
        return ""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda _: None
)
def fetch_with_retry(url: str, proxy_config: ProxyConfig, timeout: float) -> Optional[str]:
    proxy_url = f"{proxy_config.type}://{proxy_config.server}:{proxy_config.port}"
    headers = {
        'User-Agent': ua_rotator.get_random_user_agent(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
    }

    try:
        with ClientSession() as session:
            with session.get(url, proxy=proxy_url, timeout=timeout, headers=headers, allow_redirects=True) as response:
                logger.info(f"Response status: {response.status}")
                logger.info(f"Response headers: {response.headers}")
                asyncio.sleep(random.uniform(1, 3))
                response.raise_for_status()
                return response.text()
    except ClientError as e:
        logger.error(f"Client error during fetch: {str(e)}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout error during fetch for URL: {url}")
    except Exception as e:
        logger.error(f"Unexpected error during fetch: {str(e)}")
    return None


def scrape_pages(config: ScraperConfig, context, proxy_manager: ProxyManager, rate_limiter: AsyncLimiter):
    all_results = []
    for page_num in range(config.pages_to_scrape):
        url = f"https://www.google.com/search?q={config.query}&start={page_num * 10}"
        content = fetch_with_proxy(ua_rotator, proxy_manager, rate_limiter, url=url, timeout=config.timeout)
        if content:
            results = scrape_urls(content, context, config.results_per_page)
            all_results.extend(results)
    return all_results

def save_results_to_csv(results: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['title', 'url', 'content'])
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Scraping completed. {len(results)} results saved to {filename}")
    
    
def main():
    try:
        config = load_config('scraper_config.yaml')
        proxy_manager = ProxyManager(config.proxy_configs)
        
        with async_playwright() as p:
            browser = p.chromium.launch(headless=True)
            with browser.new_context(user_agent=ua_rotator.get_random_user_agent()) as context:
                rate_limiter = rate_limiter.RateLimiter(1, config.rate_limit)
                all_results = scrape_pages(config, context, proxy_manager, rate_limiter)
        
        save_results_to_csv(all_results, 'search_results_with_content.csv')
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
