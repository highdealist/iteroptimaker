import logging
import re
from tools.web_search_tools.web_search_tool_v2 import SearchAPI, SearchManager, SearchProvider, SearchResult, initialize_search_manager
from models.gemini import GeminiModel
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('researcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


    
def extract_tool_call(text: str) -> Dict[str, str]:
    logger.debug(f"Attempting to extract tool call from text: {text}")
    match = re.search(r"Tool:\s*(.*?),\s*Query:\s*(.*)", text)
    if match:
        result = {"tool": match.group(1).strip(), "query": match.group(2).strip()}
        logger.debug(f"Successfully extracted tool call: {result}")
        return result
    logger.debug("No tool call found in text")
    return None

def process_user_input(user_input: str) -> str:
    logger.info("Starting process_user_input")
    logger.info(f"Initializing search manager")
    search_manager = initialize_search_manager()
    if not search_manager:
        logger.error("Search manager initialization failed")
        return "Search functionality is disabled."

    researcher_system_message = (
        "You are the researcher for a think tank team of content publishers, you serve as the liason between the internet and the team.  "
        "Your primary goal is to extract information from provided search results that is relevant to the team's inquiry and present it in a comprehensive and detailed report for the requesting team member in order to provide missing or additional information and any relevant supporting information that may be useful. "
        "based upon the context of the provided conversation / message log. If the initial search results content is not sufficient to answer the original query, then do not write the report, instead request a new query in the format 'Tool: Web Search, Query: | Your Search Query |'.  "
        "The quotes are unnecessary, but you MUST adhere to this format including both the pipe characters '|', the colon ':' and the 'Tool: Web Search' flag  otherwise the search request will be ignored.  "
        "This new query should be more specific, more detailed, and more effective query keywords to improve the quality and specificity of the search results and get the needed information. !!! NOTE: DO NOT ASSUME OR GUESS ANY ASSERTION OR STATEMENT OF FACT IN YOUR REPORTS.  IF YOU ARE NOT PROVIDED WITH THE NECESSARY INFORMATION, DO NOT WRITE THE REPORT UNTIL YOU OBTAIN IT FROM AS MANY SEARCH REQUESTS AS REQUIRED. RESULTS FROM THIS SEARCH QUERY IN THE FOLLOWING MESSAGE !!!  "
        "To request an additional search if needed, use the following format explicitly and include nothing else in your response: "
        "Tool: Web Search, Query: | Your Search Query | "
    )

    logger.info("Initializing Gemini model")
    researcher_model = GeminiModel(
        model_config={"model_name": "gemini-2.0-flash-exp", "temperature": 0.7},
        system_message=researcher_system_message
    )

    search_count = 0
    search_results = ""
    conversation_history = ""
    max_searches = 3

    while search_count < max_searches:
        logger.info(f"Starting search iteration {search_count + 1}/{max_searches}")
        prompt = f"""
        ## Conversation History:
        {conversation_history}

        ## Web Search Results:
        {search_results}

        ## User Input:
        {user_input}

        ## Your Task:
        Based on the provided information, determine if you need additional information to fulfill the user's request.
        If YES, provide a specific web search query in the format 'Tool: Web Search, Query: | Your Search Query |'.
        If NO, synthesize a report based on the available information.
        """

        logger.info("Sending prompt to researcher model")
        researcher_response = researcher_model.chat([{"role": "user", "content": prompt}])
        logger.info(f"Received response from researcher model: {researcher_response[:200]}...")
        conversation_history += f"User: {user_input}\nAI: {researcher_response}\n"

        tool_call = extract_tool_call(researcher_response)

        if tool_call:
            logger.info(f"Tool call detected: {tool_call}")
            if tool_call["tool"] == "Web Search":
                logger.info(f"Executing web search with query: {tool_call['query']}")
                search_results_list = search_manager.search(tool_call['query'], num_results=5)
                search_results = f"Web search results for '{tool_call['query']}':\n"
                for i, result in enumerate(search_results_list):
                    search_results += f"Result {i+1}:\nTitle: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\nContent: {result['content'][:500]}...\n\n"
                search_count += 1
                logger.info(f"Search completed. Total searches: {search_count}")
            else:
                logger.warning(f"Unsupported tool requested: {tool_call['tool']}")
                return "Unsupported tool."
        else:
            logger.info("No tool call detected, returning researcher response")
            return researcher_response

    logger.info("Maximum searches reached, generating final response")
    final_prompt = f"""
    ## Conversation History:
    {conversation_history}

    ## Web Search Results:
    {search_results}

    ## User Input:
    {user_input}

    ## Your Task:
    You have reached the maximum number of allowed searches.
    Synthesize a research report based on the available information, addressing the user's request.
    """
    
    logger.info("Sending final prompt to researcher model")
    final_response = researcher_model.chat([{"role": "user", "content": final_prompt}])
    logger.info("Process completed successfully")
    return final_response

if __name__ == "__main__":
    user_input = "What are the benefits of using renewable energy sources?"
    response = process_user_input(user_input)
    print(response)