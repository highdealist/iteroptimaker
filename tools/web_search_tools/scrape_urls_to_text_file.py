import requests
from bs4 import BeautifulSoup
import os

def extract_main_content(html_content):
  soup = BeautifulSoup(html_content, 'html.parser')

  # Some common patterns for main content
  potential_containers = [
      soup.find('article'),
      soup.find('main'),
      soup.find('div', class_=['content', 'article-body', 'post-content']),
      soup.find('section', id='content')
  ]

  for container in potential_containers:
    if container:
      text = container.get_text(separator='\n').strip()
      return text
  
  # Fallback to get text from the whole body (less accurate)
  return soup.body.get_text(separator='\n').strip()


def process_urls(urls):
  all_text = ""
  for url in urls:
    try:
      response = requests.get(url)
      response.raise_for_status()  # Raise an exception for bad status codes

      main_text = extract_main_content(response.text)
      all_text += main_text + "\n\n"
    except requests.exceptions.RequestException as e:
      print(f"Error fetching or processing {url}: {e}")

  return all_text

if __name__ == "__main__":
  urls = input("Enter the URLs separated by spaces: ").split()
  concatenated_text = process_urls(urls)

  output_filename = "concatenated_content.txt"
  with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(concatenated_text)
  
  os.startfile(output_filename) #Opens the text file, windows only