
import requests
import arxiv
from typing import List
from web_search_tool_v2 import SearchResult
import datetime
from datetime import datetime, timedelta
import logging
import re
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # Get a logger instance

class ArXivSearchProvider:
    """Provider for searching arXiv papers."""
    
    def __init__(self):
        self.client = arxiv.Client()
    
    def get_latest_papers(self, category: str = None, max_results: int = 10,
                              days: int = 7) -> List[SearchResult]:
        """Get the latest arXiv papers, optionally filtered by category.
        
        Args:
            category (str, optional): arXiv category (e.g., 'cs.AI', 'physics')
            max_results (int): Maximum number of results to return
            days (int): Number of past days to search
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Build search query
            search_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]"
            if category:
                search_query += f" AND cat:{category}"
            
            # Create search
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            results = []
            for paper in self.client.results(search):
                # Extract authors
                authors = ", ".join([author.name for author in paper.authors])
                
                # Create content combining abstract and metadata
                content = f"Title: {paper.title}\n\n"
                content += f"Authors: {authors}\n\n"
                content += f"Published: {paper.published}\n"
                content += f"Updated: {paper.updated}\n"
                content += f"DOI: {paper.doi}\n" if paper.doi else ""
                content += f"Primary Category: {paper.primary_category}\n"
                content += f"Categories: {', '.join(paper.categories)}\n\n"
                content += f"Abstract:\n{paper.summary}\n\n"
                content += f"PDF URL: {paper.pdf_url}\n"
                
                results.append(SearchResult(
                    title=paper.title,
                    url=paper.entry_id,
                    snippet=paper.summary[:2000] + "...",
                    content=content
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")
            return []
    def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search arXiv papers by query.
        
        Args:
            query (str): The search query
            max_results (int): Maximum number of results to return
            
        Returns:
            List[SearchResult]: List of search results
        """
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in self.client.results(search):
                # Extract authors
                authors = ", ".join([author.name for author in paper.authors])
                
                # Create content combining abstract and metadata
                content = f"Title: {paper.title}\n\n"
                content += f"Authors: {authors}\n\n"
                content += f"Published: {paper.published}\n"
                content += f"Updated: {paper.updated}\n"
                content += f"DOI: {paper.doi}\n" if paper.doi else ""
                content += f"Primary Category: {paper.primary_category}\n"
                content += f"Categories: {', '.join(paper.categories)}\n\n"
                content += f"Abstract:\n{paper.summary}\n\n"
                content += f"PDF URL: {paper.pdf_url}\n"
                
                results.append(SearchResult(
                    title=paper.title,
                    url=paper.entry_id,
                    snippet=paper.summary[:200] + "...",
                    content=content
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
        
    def scrape_pdf_url(self, url: str) -> str:
        """Scrape the PDF URL from an arXiv paper page.
        
        Args:
            url (str): The URL of the arXiv paper page
            
        Returns:
            str: The URL of the PDF
        """
        try:
            # Fetch the page
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the page
            page = response.text
            pdf_url = re.search(r'href="([^"]+\.pdf)"', page).group(1)
            
            return pdf_url
        
        except Exception as e:
            logger.error(f"Error scraping PDF URL: {e}")
            return ""
            
    def scrape_pdf_content(self, pdf_url: str) -> str:
        """Scrape the content from a PDF URL.
        
        Args:
            pdf_url (str): The URL of the PDF
            
        Returns:
            str: The content of the PDF
        """
        try:
            # Fetch the PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            
            # Parse the PDF content
            with open ('temp.pdf', 'wb') as f:
                f.write(response.content) # Save the PDF to a temporary file
            return "PDF content"
        
        except Exception as e:
            logger.error(f"Error fetching PDF content: {e}")
            return ""
            




    


#Usage
if __name__ == '__main__':
    option = input("Do you want to search for the latest papers or a specific query? (1=latest / 2=query): ")
    if option not in ['1', '2']:
        print("Invalid option. Please select 1 or 2.")
    toggle_scrape = input("Do you want to scrape the full papers from arXiv? (y/n): ")
    search_provider = ArXivSearchProvider()
    if option == '1':
        category = input("Enter the category of papers you want to search: ")
        latest_papers = search_provider.get_latest_papers(category, max_results=5, days=7)
        for i, result in enumerate(latest_papers):
            print(f"Result {i+1}:\n{result.content}\n")
            if toggle_scrape.lower() == 'y':
                page_url = result.url
                pdf_url = result.paper.pdf_url
                print(f"PDF URL for {page_url}: {pdf_url}")
            
    if option == '2':
        query = input("Enter the search query: ")
        search_results = search_provider.search(query, max_results=5)
        for i, result in enumerate(search_results):
            print(f"Result {i+1}:\n{result.content}\n")
            if toggle_scrape.lower() == 'y':
                page_url = f"https://arxiv.org/abs/{result.url}"
                pdf_url = search_provider.scrape_pdf_url(page_url)
                print(f"PDF URL for {page_url}: {pdf_url}")
    
        
        