import pygetwindow as gw
import pyperclip
import keyboard
import sys
import os

# Get the directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)
# Add the project root to the Python path
sys.path.insert(0, parent_dir)

from search_manager import WebContentExtractor, SearchManager, SearchProvider, SearchAPI, DuckDuckGoSearchProvider
import datetime
import os
import psutil

def is_browser_window(window_title):
    """Check if the current window is a browser window."""
    browsers = ["Microsoft Edge", "Google Chrome", "Firefox", "Safari", "Opera"]
    return any(browser in window_title for browser in browsers)

def setup_hotkey():
    """Set up a hotkey (Ctrl+Alt+C) to capture and scrape the current browser URL."""
    def on_hotkey():
        url = capture_url()
        if url:
            content_extractor = WebContentExtractor()
            content = content_extractor.extract_content(url)
            save_to_file(url, content)
            print(f"Captured and saved content from: {url}")
        else:
            print("No valid URL found in current window")

    keyboard.add_hotkey('ctrl+alt+c', on_hotkey)
    print("Hotkey (Ctrl+Alt+C) registered for URL capture")


def get_edge_url_with_psutil():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['name'] == 'msedge.exe':
            cmdline = proc.info['cmdline']
            for arg in cmdline:
                if arg.startswith('https://'):
                    return arg

def capture_url():
    # Get the active Edge window
    try:
        edge_window = gw.getActiveWindow()

        if "Microsoft Edge" not in edge_window.title:
            return None
        # Get the URL from the clipboard
        url = pyperclip.paste()

        if not url.startswith("http"):
            # If the clipboard doesn't contain a URL, try to get the URL from the Edge window title
            url = edge_window.title.split(" - ")[0]
        if not url.startswith("https://"):
            url = get_edge_url_with_psutil()
            
        return url
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def save_to_file(url, content):
    """Save the URL and content to a new text file with a timestamp on the desktop."""
    if url:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
        with open(os.path.join(desktop, f"Scraped_URL_{timestamp}.txt"), "w") as f:
            f.write(f"{url}\n{content}")


url = capture_url()
if url:
    content_extractor = WebContentExtractor()
    content = content_extractor.extract_content(url)
    save_to_file(url, content)
    print(f"Captured URL: {url}")
    print(f"Captured content: {content}")
input("Press Enter to exit...")
