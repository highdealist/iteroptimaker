import logging
import re
from .tools.web_search_tools.search_manager_LATESTWORKING import SearchAPI, SearchManager, SearchProvider, SearchResult, initialize_search_manager
from .models.gemini import GeminiModel  # Import the GeminiModel class
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_tool_call(text: str) -> Dict[str, str]:
    """Extract tool call and query from Gemini response."""
    match = re.search(r"Tool:\s*(.*?),\s*Query:\s*(.*)", text)
    if match:
        return {"tool": match.group(1).strip(), "query": match.group(2).strip()}
    return None

def process_user_input(user_input: str) -> str:
    """Process user input using Gemini and potentially other tools."""
    search_manager = initialize_search_manager()
    if not search_manager:
        return "Search functionality is disabled."

    researcher_system_message = (
        "You are the researcher for a think tank team of content publishers, you serve as the liason between the internet and the team.  "
        "Your primary goal is to extract information from provided search results that is relevant to the team's inquiry and present it in a comprehensive and detailed report for the requesting team member in order to provide missing or additional information and any relevant supporting information that may be useful. "
        "based upon the context of the provided conversation / message log. If the initial search results content is not sufficient to answer the original query, then do not write the report, instead request a new query in the format 'Tool: Web Search, Query: | Your Search Query |'.  "
        "The quotes are unnecessary, but you MUST adhere to this format including both the pipe characters '|', the colon ':' and the 'Tool: Web Search' flag  otherwise the search request will be ignored.  "
        "This new query should be more specific, more detailed, and more effective query keywords to improve the quality and specificity of the search results and get the needed information. !!! NOTE: DO NOT ASSUME OR GUESS ANY ASSERTION OR STATEMENT OF FACT IN YOUR REPORTS.  IF YOU ARE NOT PROVIDED WITH THE NECESSARY INFORMATION, DO NOT WRITE THE REPORT UNTIL YOU OBTAIN IT FROM AS MANY SEARCH REQUESTS AS REQUIRED. RESULTS FROM THIS SEARCH QUERY IN THE FOLLOWING MESSAGE !!!  "
        "To request an additional search if needed, use the following format explicitly and include nothing else in your response: "
        "Tool: Web Search, Query: | Your Search Query | "
        "NOTE: "
        "*REPLACE the text 'Your Search Query' with your actual query. "
        "*DO INCLUDE: 'Tool: Web Search', THE PIPE CHARACTERS, THE COLON (:). "
        "*DO NOT INCLUDE ANY EXTRA CHARACTERS NOR ALTER THE FORMATTING IN ANY WAY! "
        "*DO NOT WRITE THE REPORT UNTIL YOU ARE PROVIDED WITH THE RESULTS FROM THIS SEARCH QUERY IN THE FOLLOWING        MESSAGE. "
        "*IF AND ONLY IF you decide to request an additional search, following the above format is important,        otherwise your search request will not be seen. "
        "If and when you have the necessary information to sufficiently answer the original query made by the team/      writer OR you have reached the maximum limit on additional searches (3), synthesize a relevant and helpful      report from the search results already gathered and provide all needed information and details so the team        can continue their work. "
        "Include: "
        ". Implications/Actionable Information/Applicable Steps relevant to the context "
        ". Sources "
        "Prioritize synthesizing available information before requesting additional searches."
    )
    
    researcher_model = GeminiModel(
        model_config={"model_name": "gemini-pro", "temperature": 0.7},
        system_message=researcher_system_message
    )

    search_count = 0
    search_results = ""
    conversation_history = ""
    max_searches = 3

    while search_count < max_searches:
        # Construct the prompt
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

        # Send user input to the researcher model
        logging.info(f"User Input: {user_input}")
        researcher_response = researcher_model.chat([{"role": "user", "content": prompt}])
        logging.info(f"Researcher Response: {researcher_response}")
        conversation_history += f"User: {user_input}\nAI: {researcher_response}\n"

        # Check if a tool call is present
        tool_call = extract_tool_call(researcher_response)

        if tool_call:
            if tool_call["tool"] == "Web Search":
                # Perform the web search
                logging.info(f"Performing Web Search with query: {tool_call['query']}")
                search_results_list = search_manager.search(tool_call['query'], num_results=5)
                search_results = f"Web search results for '{tool_call['query']}':\n"
                for i, result in enumerate(search_results_list):
                    search_results += f"Result {i+1}:\nTitle: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\nContent: {result['content'][:500]}...\n\n"
                search_count += 1
            else:
                return "Unsupported tool."
        else:
            # If no tool call, return the response
            return researcher_response

    # If we reach here, we've hit the max searches
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
    final_response = researcher_model.chat([{"role": "user", "content": final_prompt}])
    return final_response

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    final_response = process_user_input(user_input)
    print(f"Final Response: {final_response}")
