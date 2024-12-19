import pyperclip
import google.generativeai as genai
import dotenv
import os
from models.gemini import GeminiModel
import agents
from agents.researcher_agent import ResearcherAgent
from agents.writer_agent import WriterAgent

# Import the GenerateContentResponse class from the genai module
from google.generativeai.types import GenerateContentResponse

# Create a GenerateContentResponse object using the WriterAgent
response = genai.GenerateContentResponse(agent=agents.WriterAgent())

# Print the response object
print(response)


# Load environment variables from .env file
dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Function to get text from clipboard
def get_clipboard_text():
    return pyperclip.paste()

def generate_prompt(clipboard_text, extra_instructions):
    template = f"""Please infer based on the following text and instructions what I may need help with and please help figure out / resolve it.
    Instructions: {extra_instructions}
    Current Work:  {clipboard_text}

    Provide clear, concise guidance and explain any important considerations."""
    return template

# Function to call the Gemini API, setting system_instructions according to the agent_name (the agent is in large part defined by its system_instructions)
def call_gemini(prompt):
    model = GeminiModel({"model_name": "gemini-2.0-flash-exp", "temperature": 0.7})
    response = model.generate_content(prompt)
    return response.text




def main():
    clipboard_text = get_clipboard_text()

    options = {
        "1": "Improve the structure, wording, conciseness, comprehensiveness, and clarity of the input text below.",
        "2": "Review the following code for potential bugs, performance issues, and best practice violations.",
        "3": "Translate the following text to [Target Language].",  # User will specify target language
        "4": "Summarize the following text.",
        "5": "Generate creative content based on the following input (e.g., a story, poem, or article).",
        "6": "Answer the following question based on the provided context.",
        "7": "Extract key information and entities from the following text.",
        "8": "Rewrite the following text in a different style (e.g., formal, informal, technical).", # User will specify the style
        "9": "Brainstorm ideas related to the following topic."
    }

    for key, value in options.items():
        print(f"{key}: {value}")

    while True:  # Loop until a valid option is chosen
        extra_instructions_key = input("Enter the corresponding number (1-9): ")
        if extra_instructions_key in options:
            extra_instructions = options[extra_instructions_key]

            if extra_instructions_key == "3":
                target_language = input("Enter the target language: ")
                extra_instructions += f" Translate to {target_language}."
            elif extra_instructions_key == "8":
                style = input("Enter the target style: ")
                extra_instructions += f" Rewrite in {style} style."


            prompt = generate_prompt(clipboard_text, extra_instructions)
            response = call_gemini(prompt)
            print(response)
            break  # Exit loop after successful processing
        else:
            print("Invalid input. Please enter a number between 1 and 9.")


if __name__ == "__main__":
    main()