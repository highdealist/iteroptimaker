import time
import google.generativeai as genai
from collections import deque
from google.api_core import retry
from datetime import datetime

# Constants
API_KEY = "AIzaSyC5eWR1ZfRIdni39hzOlfjEIYQFtfJkkUk"

#Prompt user to get the max number of iterations
def set_max_iterations():
    max_iterations = int(input("Enter the maximum number of iterations: "))
    return max_iterations

# Get the user prompt
def get_prompt():
    """
    Prompts the user to enter a task prompt and returns it.

    Returns:
    str: The user-provided task prompt.
    """
    while True:
        prompt = input("Enter your prompt: ").strip()
        if prompt == "1":
            # Create a collection of five unique, creative, interesting, and hilarious T-shirt design
            return "Create a collection of five unique, creative, interesting, and hilarious T-shirt designs. The ultimate final objective is to end up with 5 paragraphs that perfectly describe the designs such that the paragraphs could be simply input into Midjourney or Stable Diffusion to result in the graphical designs perfectly generated. Don't try to complete this task all in one go, start off with some brainstorming and iteratively work the feedback you get to develop and refine the 5 ideas. Remember, you're creating prompts for text-to-image generative ai models, so we're not working with real physical artwork here; the whole creative process will lie within our writing and thoughtful ekphrasis."
        elif prompt == "2":
            # Create a concept for a song from a new type of genre, combining elements of hip-hop, rock, jazz, electronic dance music, dubstep, pop, and metal funk.
            return "Create a concept for a song from a new type of genre, combining elements of hip-hop, rock, jazz, electronic dance music, dubstep, pop, and metal funk. The ultimate final objective is to end up with 5 paragraphs that perfectly describe the song such that the paragraphs could be simply input into a song generative ai model, such as Sudo AI and a high quality, unique song will result. Don't try to complete this task all in one go, start off with some brainstorming and iteratively work the feedback you get to develop and refine the concept. Remember, you're brainstorming, honing in on a unique idea, and ultimately creating a prompt for a text-to-audio generative ai model."
        elif prompt == "3":
            # Create postcard ideas
            return """Craft a prompt for Stable Diffusion that will guide it to design captivating postcards with a unique and trendy aesthetic, specifically tailored to appeal to buyers on Etsy and similar print-on-demand shops. The prompt you create should serve as a clear and comprehensive guideline, with the goal of generating desirable and marketable postcard designs. Keep in mind the target demographic of print-on-demand shoppers who appreciate handmade, personalized, and distinctive items. The prompt should aim for a design that resonates with this audience and stands out from the competition. Here are the key elements to include in the prompt for Stable Diffusion:
Theme and Concept: Emphasize the importance of a cohesive design centered around a specific theme. This could be an overarching concept or a specific aesthetic that ties all the visual elements together.
Image: Instruct Stable Diffusion to meticulously describe the visual composition, including layout, vibe, and style. Prompt it to consider the overall arrangement of elements, the choice of colors, and the inclusion of any specific illustrative styles or textures to create a unique look and feel.
Text: Guide Stable Diffusion to generate a concise and captivating message, quote, or saying that complements the image. Ensure it aligns with the theme and resonates with the intended audience. The text should enhance the visual appeal and could range from a playful pun to an inspiring quote.
Format and Size: Remind Stable Diffusion to adhere to standard postcard dimensions and provide the specific measurements and aspect ratios to ensure the designs are print-ready. Include any relevant information about leaving adequate space for messages, addresses, and postage.
Your prompt should provide a clear framework for Stable Diffusion to create appealing postcard designs with strong sales potential on print-on-demand platforms. Feel free to add any further details or constraints to tailor the prompt to your specific requirements, or leave creative freedom for Stable Diffusion to offer unexpected yet delightful interpretations!"""
            ##Checks if prompt contains the word joke, funny, or humor and calls the set_topic_and_format function if so
        elif prompt:
            return prompt
        else:
            print("Prompt cannot be empty. Please provide a valid input.")

def set_topic_and_format():
    topic = input("Enter the topic: ")
    format = input("Enter the format: ")
    return topic, format

# Generation configurations for various models
generation_configs = [
    {
        "temperature": 0.6,
        "top_p": 0.7,
        "top_k": 20,
        "max_output_tokens": 32000,
    },
    {
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 30,
        "max_output_tokens": 32000,
    },
    {
        "temperature": 0.9,
        "top_p": 0.9,
        "top_k": 30,
        "max_output_tokens": 32000,
    },
    {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
        "max_output_tokens": 32000,
    },
    {
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": 50,
        "max_output_tokens": 32000,
    },
    {
        "temperature": 1.2,
        "top_p": 0.95,
        "top_k": 50,
        "max_output_tokens": 32000,
    },
]

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Helper function to initialize a model
def initialize_model(system_instruction, generation_config):
    """
    Initializes a generative AI model with the given system instruction and generation configuration.

    Parameters:
    system_instruction (str): The instruction that defines the role and behavior of the model.
    generation_config (dict): The configuration settings for the model's generation process.

    Returns:
    genai.GenerativeModel: The initialized generative AI model.
    """
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        system_instruction=system_instruction,
        model_name="models/gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    return model

# Initialize team of creative content worker models
def initialize_models():
    """
    Initializes all predefined models with their respective instructions and generation configurations.
    """
    instructions = [
        ("Evaluative and Constructive Feedback Model", "You are an astute and outspoken Evaluative Constructive Critique Assistant, dedicated to enhancing written content of all kinds, from lyrics to blog posts and even text-to-image prompts for sticker designs. Your direct and no-nonsense approach is designed to provide efficient and effective feedback. You dive straight into the specifics, offering thorough and respectful critiques supported by concrete examples. Your feedback is tailored to elevate the content's originality, impact, and engagement while ensuring it remains cliché-free. You also consider the potential virality of the content, offering suggestions to enhance its shareability. Your role is to identify areas where the content could be strengthened, providing precise and actionable recommendations for improvement. You maintain a balanced tone, offering frank feedback without sugarcoating, while also refraining from unnecessary harshness. Your goal is to help creators refine their work, ensuring it resonates with their intended audience and achieves its full potential."),
        ("Linguistic Enhancer / Writer Model", "Elevate written content through meticulous refinement and creative input. Enhance clarity, coherence, and readability while preserving the original voice and intent. Employ your linguistic expertise, eye for detail, and literary device prowess to transform drafts. Work closely with content creators, providing suggestions and ensuring consistency in tone, style, and messaging. As the lead writer, you have final say in strategies and are responsible for shaping compelling narratives and persuasive communication. Prerequisite: deep understanding of the audience and a creative, detail-oriented mindset."),
        ("Legendary Comic", f"As a legendary comedian, inject your skills and humor into a new revision of the current content.  Utilize your razor-sharp wit, honed from the legends of comedy, to tackle sensitive or controversial premises that are often avoided. Your jokes should be hilarious, unique, and aim to provoke both laughter and thought. Focus on mining the rich veins of human absurdity and societal foibles to create content that resonates deeply with the audience."),
        ("Visual Artist / Text to Image Prompt Articulator Model",  "As a Visual Artist and Text-to-Image Prompt Articulator, your role is to craft the most effective and evocative prompts for MidJourney and Stable Diffusion to generate stunning visual art. Leverage your deep understanding of visual aesthetics, art history, and creative language to articulate detailed and inspiring prompts that guide AI models in producing high-quality, visually captivating artwork. Collaborate closely with artists, designers, and other creative professionals to understand their vision and translate it into precise, descriptive prompts that capture the essence of the desired output. Your tasks include researching visual trends, experimenting with different prompt structures, and continuously refining your approach to achieve the best possible results. This position requires a blend of artistic insight, linguistic precision, and a keen eye for detail to ensure that the generated visuals not only meet but exceed expectations, resonating deeply with the intended audience."),
        ("Lyricist Model",  "As a Lyricist Model, your role is to craft compelling and emotionally resonant lyrics for songs across various genres. Utilize your deep understanding of poetic devices, rhythm, and storytelling to create lyrics that capture the essence of the given theme or concept. Your lyrics should be original, engaging, and capable of evoking strong emotions in the listener. Collaborate closely with composers and musicians to ensure that the lyrics complement the musical composition and enhance the overall impact of the song. Focus on maintaining coherence, originality, and emotional depth in your lyrics to create a powerful and memorable musical experience."),
        ("Creative Content Producer",  "As a Creative Content Producer, your role is to generate unique and novel concepts for various types of content and formats. Utilize a range of ideation techniques such as brainstorming, forced connections, and subverting expectations to produce dynamic and comprehensive creative output. Your goal is to create content that is original, engaging, and aligns with the specified objectives. Collaborate with other team members to refine and enhance ideas, ensuring that the final output is both innovative and impactful. Focus on maintaining a high level of creativity and originality throughout the process."),
         ]
    models = [(name, initialize_model(instruction, config)) for (name, instruction), config in zip(instructions, generation_configs)]
    model_index_and_name = [(i, model[0]) for i, model in enumerate(models)]
    models_list_str = "\n".join([f"{index}: {name}" for index, name in model_index_and_name])
    return models, models_list_str

# Iterate through all models and return the index and name of model in a string to be used in the coordinator's prompt
def get_model_index_and_name(models):
    try:
        return [(i, model[0]) for i, model in enumerate(models)]
    except Exception as e:
        print(f"Error retrieving model index and name: {e}")
        return []
    
def get_model_indices(models):
    try:
        return [model[0] for model in models]
    except Exception as e:
        print(f"Error retrieving model indices: {e}")
        return []
    
# Initialize the director of the creative team / the coordinator model
# Initialize the director of the creative team / the coordinator model
def initialize_coordinator_model(models_list_str):
    instruction = (
        f"""You are a project coordinator responsible for keeping the team focused and on track. Your role includes tracking progress and maintaining team alignment by providing brief periodic status reports that recap the latest significant contributions to the content being deveveloped, and offering direction by recommending the next steps to take or aspect to focus on that will progress the team towards the final output goal.  Your primary goal is to achieve the objective specified in the initial prompt.

        You will:
        (a) Maintain context and track progress by briefly recapping the latest progress related to the user's objective.
        (b) Dynamically direct team member interactions by selecting the most appropriate team member to work on the content in each round.

        Always respond with only two (2) sections:
        1. Your status report, which should be a concise summary of the latest progress and the next steps to take to achieve the objective.
        2. The single digit number corresponding to the member of the creative team that is most appropriate or most logical to next provide their input/work next considering the current context, progress, content type, and available experts on our team: {models_list_str} \n
        IT IS IMPORTANT that the number you write is delimited by percent signs, like so: %x% (replace x with the numerical number).
        Now, respond with your status report and guidance related to the content, followed by your chosen single digit number."""
    )
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 30,
        "max_output_tokens": 32000,
    }
    return initialize_model(instruction, generation_config)



# Generate a response from a model
def generate_response(model, prompt):
    try:
        response = model.generate_content(prompt, request_options={"retry": retry.Retry()})
        return response.text if response and response.text else ""
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

# Have the coordinator model recap with a status report and decide the next model for the next iteration
def decide_next_model(coordinator_model, chat_history, prompt, models, models_list_str):
    model_indices = get_model_indices(models)
    try:
        chat_history_str = "\n".join(chat_history)
        decision_prompt = (
            f"{chat_history_str}\nBased on the progress up to this point and what the next steps ought to be, choose the number corresponding to the next most appropriate expert ( {models_list_str} ) to work on/provide input for/improve the quality of the content to en to completing the initial objective: '{prompt}'.  Ensure your response always consists of your feedback for the team with respect to the initial objective followed by, in a new paragraph, your chosen single digit number ENCLOSED IN PERCENT SIGNS, (SUCH AS %<numerical number>%) out of {', '.join(model_indices)}, which corresponds to your choice for the best expert to work on / provit input next."
        )
        
        decision_response = generate_response(coordinator_model, decision_prompt)
        feedback, model_choice = decision_response.rsplit('%', 2)[0], decision_response.rsplit('%', 2)[1]
        chat_history.append(f"Coordinator Feedback: {feedback.strip()}")
        print(f"\nCoordinator Feedback: {feedback.strip()}")
        return int(model_choice.strip())
    except ValueError:
        print(f"Unable to parse the response: '{decision_response}'. Defaulting to the first model.")
        return 0  # Default to the first model if parsing fails
    except Exception as e:
        print(f"Error deciding next model: {e}")
        return 0
    
# Process iterations with the models for max_iterations number of times, saving the chat history to a text file after each iteration, prompting the user for input after each round of iterations, and continuing with another round of iterations with or without user input or simply ending the program based on the user's choice.
def process_iterations(models, coordinator_model, chat_history, prompt, max_iterations, models_list_str):
    total_iterations = 0
    while total_iterations < max_iterations:
        current_model_index = decide_next_model(coordinator_model, chat_history, prompt, models, models_list_str)
        current_model_name, current_model = models[current_model_index]

        # Generate response with the current model
        prompt_text = "\n".join(chat_history)
        model_response = generate_response(current_model, prompt_text)
        chat_history.append(f"Model: {current_model_name} Response: {model_response}\n")

        total_iterations += 1
        print(f"\nIteration {total_iterations} with {current_model_name}:\n{model_response}\n")

    # Ask for user input after completing the iterations
    while True:
        user_choice = input("Do you want to continue iterations (y), save to text file and end (n), or provide feedback and continue (f)? (y/n/f/s): ").strip().lower()
        if user_choice == 'n':
            save_chat_history(chat_history[-max_iterations*2])
            break
        elif user_choice == 'f':
            prompt = get_prompt()
            chat_history.append(f"Objective: {prompt}")
            save_chat_history(chat_history[-max_iterations*2])
        elif user_choice == 'y':
            save_chat_history(chat_history[-max_iterations*2])
            process_iterations(models, coordinator_model, chat_history, prompt, max_iterations, models_list_str)
            break
        else:
            print("Invalid choice. Please enter y, n, or f.")
        


# Save the chat history to a text file titled "chat_history_<timestamp formatted as DD_MM_YYYY>.txt" in the current directory.  If nonexisting, create a new file for the day.
def save_chat_history(chat_history):
    try:
        timestamp = datetime.now().strftime("%d_%m_%Y")
        file_name = f"chat_history_{timestamp}.txt"
        with open(file_name, "w") as f:
            f.write("\n".join(chat_history))
    except Exception as e:
        print(f"Error saving chat history: {e}")
        return
        

def main():
    max_iterations = set_max_iterations()
    models, models_list_str = initialize_models()
    coordinator_model = initialize_coordinator_model(models_list_str)
    chat_history = deque(maxlen=12)
    prompt = get_prompt()
    chat_history.append(f"Objective: {prompt}")
    process_iterations(models, coordinator_model, chat_history, prompt, max_iterations, models_list_str)
    save_chat_history(chat_history)  # Ensure chat history is saved when the program terminates


if __name__ == "__main__":
    main()