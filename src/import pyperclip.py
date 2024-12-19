import pyperclip
import google.generativeai as genai
import dotenv

load.

def get_clipboard_text():
    return pyperclip.paste()

def generate_prompt(clipboard_text, extra_instructions):
    template = f"""Please infer based on the following text and instructions what I may need help with and please help figure out / resolve it.
    Instructions: {extra_instructions}
    Current Work:  {clipboard_text}

    Provide clear, concise guidance and explain any important considerations."""
    return template

def call_gemini(prompt):
    # Ensure API key is set - ideally, don't hardcode, use environment variables
    api_key = "YOUR_API_KEY"  # Or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set GOOGLE_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
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