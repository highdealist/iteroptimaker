"""Module for defining and managing external tools for the AI assistant."""
from typing import List, Annotated, Optional
from src.tools.search_manager import SearchManager
from src.tools.fetch_latest_arxiv_papers import fetch_latest_arxiv_papers
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL
import signal
from contextlib import contextmanager
import time

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """Context manager for timing out operations after specified seconds."""
    def signal_handler(signum, frame):
        raise TimeoutError("Operation timed out")
    
    # Register a function to raise a TimeoutError on the signal
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

repl = PythonREPL()

@tool
def web_search(
    search_manager: SearchManager, 
    query: str, 
    num_results: int = 10,
    timeout_seconds: int = 30
) -> str:
    """Performs a web search using the provided SearchManager.
            
    Args:
        search_manager: The SearchManager instance to use.
        query (str): The search query.
        num_results (int, optional): The maximum number of results to return. Defaults to 10.
        timeout_seconds (int, optional): Maximum time to wait for search results. Defaults to 30.

    Returns:
        str: A formatted string containing the search results.

    Raises:
        ValueError: If num_results is less than 1 or query is empty.
        TimeoutError: If the search operation times out.
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty")
    
    if num_results < 1:
        raise ValueError("num_results must be at least 1")

    try:
        with timeout(timeout_seconds):
            results = search_manager.search(query, num_results)
            if not results:
                return "No results found for the given query."
            
            return "\n\n".join(
                [
                    f"**{result['title']}** ({result['url']})\n{result['snippet']}\n{result['content'][:50000]}"
                    for result in results
                ]
            )
    except TimeoutError:
        return "Search operation timed out. Please try again or refine your query."
    except Exception as e:
        return f"Search failed: {str(e)}"

@tool
def fetch_recent_arxiv_papers_by_topic(
    topic: str,
    timeout_seconds: int = 30
) -> List[str]:
    """Fetches recent arXiv papers based on a given topic.
    
    Args:
        topic (str): The topic to search for papers.
        timeout_seconds (int, optional): Maximum time to wait for results. Defaults to 30.
        
    Returns:
        List[str]: List of paper information.
        
    Raises:
        ValueError: If topic is empty.
        TimeoutError: If the operation times out.
    """
    if not topic.strip():
        raise ValueError("Topic cannot be empty")

    try:
        with timeout(timeout_seconds):
            return fetch_latest_arxiv_papers(topic)
    except TimeoutError:
        return ["Operation timed out while fetching arXiv papers. Please try again."]
    except Exception as e:
        return [f"Failed to fetch arXiv papers: {str(e)}"]

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
    timeout_seconds: int = 10,
    max_output_length: int = 10000
) -> str:
    """Executes Python code and returns the output.
    
    Args:
        code (str): The Python code to execute.
        timeout_seconds (int, optional): Maximum execution time in seconds. Defaults to 10.
        max_output_length (int, optional): Maximum length of output to return. Defaults to 10000.
        
    Returns:
        str: The execution result or error message.
        
    Raises:
        TimeoutError: If code execution exceeds timeout_seconds.
    """
    if not code.strip():
        return "No code provided to execute."

    try:
        with timeout(timeout_seconds):
            start_time = time.time()
            result = repl.run(code)
            execution_time = time.time() - start_time
            
            if len(str(result)) > max_output_length:
                result = str(result)[:max_output_length] + "... (output truncated)"
            
            return (
                f"Successfully executed in {execution_time:.2f}s:\n"
                f"```python\n{code}\n```\n"
                f"Stdout: {result}"
            )
    except TimeoutError:
        return f"Code execution timed out after {timeout_seconds} seconds"
    except Exception as e:
        return f"Failed to execute. Error: {repr(e)}"
