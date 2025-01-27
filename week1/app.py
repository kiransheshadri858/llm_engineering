import os
from dotenv import load_dotenv
from webscraper import Website
from client_open_ai import OpenAIClient
from client_ollama import OllamaClient
from llm_client import LLMClient


load_dotenv()  # This loads the environment variables from the .env file.

def summarize(url: str, client: LLMClient, model: str) -> str:
    """
    Fetches content from a URL and generates a summary using the specified LLM client.
    
    Args:
        url (str): The URL of the website to summarize
        client (LLMClient): The initialized LLM client to use
        model (str): The model to use for generation
        
    Returns:
        str: The generated summary
        
    Raises:
        Exception: If there's an error with website fetching or API communication
    """
    # Fetch website content
    website = Website(url)
    
    system_prompt = """
    You are an assistant that analyzes the content of a website and provides a summary of the content. 
    Ignore text that is not relevant to the website's content and may be navigation or other non-content text.
    """

    user_prompt_template = """
    I am looking at a website:

    URL: {url}
    Title: {title}

    Content:
    {text}

    Please provide a comprehensive summary of this content, focusing on the main topics and key information.
    """
    
    # Generate the prompt using website data
    user_prompt = user_prompt_template.format(
        url=website.url,
        title=website.title,
        text=website.text
    )
    
    # Get response from AI client
    return client.get_completion(system_prompt, user_prompt, model=model) 

# Example usage
if __name__ == "__main__":
    print("Website Summarizer")
    print("-----------------")
    url_name = input("Enter the website URL to summarize: ")
    
    # Display available clients
    clients = ["openai", "ollama"]
    for i, client in enumerate(clients, 1):
        print(f"{i}. {client}")
    
    choice = input(f"Enter number (1-{len(clients)}) [default: 1]: ").strip() or "1"
    client_type = clients[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(clients) else "openai"
    
    try:
        # Initialize AI client and get available models
        if client_type.lower() == "openai":
            client = OpenAIClient()
        elif client_type.lower() == "ollama":
            client = OllamaClient()
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
        
        # Get and display available models
        print(f"\nFetching available models for {client_type}...")
        available_models = client.list_models()
        
        print("\nAvailable models:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
        
        model_choice = input(f"Enter number (1-{len(available_models)}) [default: 1]: ").strip() or "1"
        model = (available_models[int(model_choice) - 1] 
                if model_choice.isdigit() and 1 <= int(model_choice) <= len(available_models) 
                else available_models[0])
        
        print(f"\nSelected model: {model}")
        print("\nFetching and analyzing content...")
        summary = summarize(url_name, client, model)
        print(f"\nWebsite Summary for {url_name}:")
        print("-" * (20 + len(url_name)))
        print(summary)
    except Exception as e:
        print(f"\nError: {str(e)}")

