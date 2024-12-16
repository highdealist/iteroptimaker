
import requests
import feedparser
def fetch_latest_arxiv_papers(topic: str) -> list:
    """
    Fetch the latest arXiv results for the given topic.

    Parameters
    ----------
    topic : str
        The topic to search for.

    Returns
    -------
    list
        A list of dictionaries containing the title, authors, summary and
        published date of each paper.
    """
    url = f'http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=11lan&sortBy=submittedDate&sortOrder=descending'
    response = requests.get(url)
    feed = feedparser.parse(response.content)

    results = []
    for entry in feed.entries:
        paper = {
            'title': entry.title,
            'authors': [author.name for author in entry.authors],
            'summary': entry.summary,
            'published': entry.published,
            'link': entry.link
        }
        results.append(paper)
    return results

if __name__ == '__main__':
    topic = input("Enter a topic to search for: ")
    results = fetch_latest_arxiv_papers(topic)
    for paper in results:
        print(f"Title: {paper['title']}\n")
        print(f"Authors: {', '.join(paper['authors'])}\n")
        print(f"Published: {paper['published']}\n")
        print(f"Summary: {paper['summary']}\n")
        print(f"Link: {paper['link']}\n")
        print('-' * 80)
