import logging
import re
from .tools.web_search_tools.search_manager_LATESTWORKING import SearchAPI, SearchManager, SearchProvider, SearchResult
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
    # Initialize the first Gemini model with a system message for calling web searches
    #implement something like this :  AI?
#    As the researcher for a think tank team of content publishers, you serve as the liason between the internet and the team.  Your primary goal is to extract information from provided search results that is relevant to the team's inquiry and present it in a comprehensive and detailed report for the requesting team member in order to provide missing or additional information and any relevant supporting information that may be useful. based upon the context of the provided conversation / message log. If the initial search results content is not sufficient to answer the original query, then do not write the report, instead request a new query in the format 'WEB_SEARCH: | Your Search Query |'.  The quotes are unnecessary, but you MUST adhere to this format including both the pipe characters '|', the colon ':' and the 'WEB_SEARCH' flag  otherwise the search request will be ignored.  This new query should be more specific, more detailed, and more effective query keywords to improve the quality and specificity of the search results and get the needed information. !!! NOTE: DO NOT ASSUME OR GUESS ANY ASSERTION OR STATEMENT OF FACT IN YOUR REPORTS.  IF YOU ARE NOT PROVIDED WITH THE NECESSARY INFORMATION, DO NOT WRITE THE REPORT UNTIL YOU OBTAIN IT FROM AS MANY SEARCH REQUESTS AS REQUIRED. RESULTS FROM THIS SEARCH QUERY IN THE FOLLOWING MESSAGE !!!  To request an additional search if needed, use the following format explicitly and include nothing else in your response: AI?
# AI?
#    ADDITIONAL_WEB_SEARCH: | Your Search Query | AI?
# AI?
#    NOTE: AI?
#        *REPLACE the text 'Your Search Query' with your actual query. AI?
#    *DO INCLUDE: 'ADDITIONAL_WEB_SEARCH', THE PIPE CHARACTERS, THE COLON (:). AI?
#    *DO NOT INCLUDE ANY EXTRA CHARACTERS NOR ALTER THE FORMATTING IN ANY WAY! AI?
#    *DO NOT WRITE THE REPORT UNTIL YOU ARE PROVIDED WITH THE RESULTS FROM THIS SEARCH QUERY IN THE FOLLOWING        MESSAGE. AI?
#        *IF AND ONLY IF you decide to request an additional search, following the above format is important,        otherwise your search request will not be seen. AI?
# AI?
#    If and when you have the necessary information to sufficiently answer the original query made by the team/      writer OR you have reached the maximum limit on additional searches (3), synthesize a relevant and helpful      report from the search results already gathered and provide all needed information and details so the team        can continue their work. AI?
#        Include: AI?
#    . Implications/Actionable Information/Applicable Steps relevant to the context AI?
#        . Sources AI?
#        Prioritize synthesizing available information before requesting additional searches. AI?
# AI?
### Code Example: AI?
# AI?
#import re AI?
# AI?
#def analyze_information_needs(conversation_history, objective, max_searches=3). : AI?
#    """ AI?
#    Analyzes conversation history and search results to determine if more information is needed. AI?
# AI?
#    Args: AI?
#        conversation_history (str). : The conversation history of the team. AI?
#        objective (str). : The objective of the team. AI?
#        max_searches (int). : The maximum number of searches allowed. AI?
# AI?
#    Returns: AI?
#        str: The LLM's response, either requesting more information or providing a summary. AI?
#    """ AI?
# AI?
#    search_count = 0 AI?
#    search_results = "" AI?
# AI?
#    while search_count < max_searches: AI?
#        # Construct the prompt AI?
#        prompt = f""" AI?
#        ## Conversation History: AI?
#        {conversation_history} AI?
# AI?
#    ## Web Search Results: AI?
#        {search_results} AI?
# AI?
#    ## Team Objective: AI?
#        {objective} AI?
# AI?
#    ## Your Task: AI?
#        Based on the provided information, determine if you need additional information to fulfill the team's objective. AI?
# AI?
#    *   If YES, provide up to 3 specific web search queries. AI?
#        *   If NO, state that you have enough information. AI?
#        """ AI?
# AI?
#    # Call the LLM with the prompt (replace with your actual LLM call). AI?
#        llm_response = call_your_llm(prompt). AI?
# AI?
#    # Check if the LLM requests more information AI?
#        if"WEB_SEARCH:"in llm_response: AI?
#            search_queries = re.findall(r"WEB_SEARCH:\s*(.+). ", llm_response). [0].split("|"). AI?
#            search_results += perform_search(search_queries). AI?
#            search_count += 1 AI?
#        else: AI?
#            # LLM has enough information AI?
#            return llm_response AI?
# AI?
#    # If we reach here, we've hit the max searches AI?
#    final_prompt = f""" AI?
#    ## Conversation History: AI?
#    {conversation_history} AI?
# AI?
#    ## Web Search Results: AI?
#    {search_results} AI?
# AI?
#    ## Team Objective: AI?
#    {objective} AI?
# AI?
#    ## Your Task: AI?
#    You have reached the maximum number of allowed searches. AI?
#    Synthesize a research report based on the available information, addressing the team's objective. AI?
#    """ AI?
#    return call_your_llm(final_prompt). AI?
# AI?
####### Example usage: AI?
# AI?
#conversation_history = "Team Member 1: What were the key economic indicators leading up to the 2008 financial crisis? AI?
#    Team Member 2: I'm not sure, we should probably research that." AI?
#objective = "Analyze the key economic indicators that contributed to the 2008 financial crisis." AI?
# AI?
#response = analyze_information_needs(conversation_history, objective). AI?
#print(response).
    
    
    
    researcher_system_message = (
        "You are a tool selection specialist. You will analyze the user's input and determine if a tool is needed. "
        "If a tool is needed, respond with 'Tool: <tool_name>, Query: <query>'. "
        "If no tool is needed, respond directly to the user's input."
    )
    tool_selector_model = GeminiModel(
        model_config={"model_name": "gemini-pro", "temperature": 0.7},
        system_message=tool_selector_system_message
    )

    # Send user input to the tool selector model
    logging.info(f"User Input: {user_input}")
    tool_selector_response = tool_selector_model.chat([{"role": "user", "content": user_input}])
    logging.info(f"Tool Selector Response: {tool_selector_response}")

    # Check if a tool call is present
    tool_call = extract_tool_call(tool_selector_response)

    if tool_call:
        if tool_call["tool"] == "Web Search":
            # Simulate a web search (replace with actual web search logic)
            logging.info(f"Simulating Web Search with query: {tool_call['query']}")
            web_search_results = f"Web search results for '{tool_call['query']}': This is a simulated search result."

            # Initialize the summarizer Gemini model with a system message for summarization
            summarizer_system_message = (
                "You are a summarization specialist. You will summarize the given text into a concise summary."
            )
            summarizer_model = GeminiModel(
                model_config={"model_name": "gemini-pro", "temperature": 0.7},
                system_message=summarizer_system_message
            )

            # Send web search results to the summarizer model
            summarizer_response = summarizer_model.chat([{"role": "user", "content": web_search_results}])
            logging.info(f"Summarizer Response: {summarizer_response}")
            return summarizer_response
        else:
            return "Unsupported tool."
    else:
        return tool_selector_response

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    final_response = process_user_input(user_input)
    print(f"Final Response: {final_response}")