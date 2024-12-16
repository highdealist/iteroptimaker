import pyttsx3
import pyperclip
import time
import re

def clean_text(text):
    # Remove URLs and email addresses with more comprehensive patterns
    # Remove URLs with or without protocol
    text = re.sub(r'(?:https?:\/\/)?(?:[\w-]+\.)+[\w-]+(?:\/[^\s]*)?', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

if __name__ == "__main__":
    # Get text from clipboard
    text = pyperclip.paste()
    
    if text:
        # Clean the text
        cleaned_text = clean_text(text)
        
        if cleaned_text.strip():
            # Initialize the text-to-speech engine
            engine = pyttsx3.init()
            voice = engine.getProperty('voices')
            engine.setProperty('voice', voice[1].id)  # Use a female voice
            
            # Convert text to speech
            engine.say(cleaned_text)
            
            # Start the engine
            engine.startLoop(True)
            
            # Wait for the user to interrupt or cancel the playback
            while engine.isBusy():
                time.sleep(0.1)
            
            # Stop the engine
            engine.stop()