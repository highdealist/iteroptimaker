import pyperclip
from ..agent_tools.search_manager import search_manager, WebContentExtractor    

def main():
    url = pyperclip.paste()
    content = WebContentExtractor.extract_content(url)
    # Print the URL to verify
    print(f"The content from: {url}\n{content}")
def scrape_url_from_clipboard():
    """
    Scrapes content from URL in clipboard and returns the extracted content.
    Returns:
        tuple: (url, content) where url is the URL that was scraped and content is the extracted text
    """
    url = pyperclip.paste()
    content = WebContentExtractor.extract_content(url)
    return url, content

# Update main to use the new function
def main():
    url, content = scrape_url_from_clipboard()
    # Print the URL to verify  
    print(f"The content from: {url}\n{content}")
    input("Press Enter to continue...")
    if input("Do you want to save the content to a file? (y/n): ") == "y":
        file_name = input("Enter the name of the file: ")
        with open(file_name, "w") as file:
            file.write(content)
        print(f"Content saved to {file_name}")  
        
if __name__ == "__main__":
    main()
            
    #Usage:
    #1. Copy the URL to the clipboard
    #2. Run the script
    #3. The script will print the content of the URL to the console
    
    #Call from another script:
    #url, content = scrape_url_from_clipboard()
    #print(f"The content from: {url}\n{content}")
