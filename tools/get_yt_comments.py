import os
import sys
import importlib
from youtube_comment_downloader import YoutubeCommentDownloader

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
parent_dir = os.path.dirname(current_dir)  # Get the parent directory
sys.path.insert(0, parent_dir)

import models  # Now import models

# Get the ModelManager class
ModelManager = models.ModelManager

# Rest of the code...
model_manager = ModelManager()  # Initialize *after* defining ModelManager
from youtube_comment_downloader import YoutubeCommentDownloader
def fetch_comments(video_url):
    downloader = YoutubeCommentDownloader()
    return [
        comment['text']
        for comment in downloader.get_comments_from_url(video_url)
    ]

def save_comments_to_file(comments, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for comment in comments:
            file.write(comment + '\n')

def main():
    url = input("Enter youtube URL: ")
    comments = fetch_comments(url)
    save_comments_to_file(comments, 'comments.txt')
    model_manager = ModelManager()
    prompt = f"Summarize and extract all insights from the following comment sections for a Youtube Video: \n {comments}"
    response = model_manager.generate_response(prompt)
    for comment in comments:
        print(comment)
    print(response.text)

if __name__ == '__main__':
    main()
