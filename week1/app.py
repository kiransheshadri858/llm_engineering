import os
from dotenv import load_dotenv
from webscraper import Website
from client_open_ai import OpenAIClient
from client_ollama import OllamaClient
from client_anthropic import AnthropicClient
from llm_client import LLMClient
import json
from pathlib import Path
from urllib.parse import urlparse
import markdown  # Add this to the imports at the top


load_dotenv()  # This loads the environment variables from the .env file.

def load_use_cases() -> dict:
    """Load use cases from JSON file and attach handler functions."""
    json_path = Path(__file__).parent / 'use_cases.json'
    with open(json_path) as f:
        use_cases = json.load(f)
    
    # Map handler strings to actual function references
    handler_map = {
        'website_summarizer': website_summarizer,
        'company_brochure': company_brochure,
        'joke_generator': joke_generator
    }
    
    # Replace handler strings with actual function references
    for use_case in use_cases.values():
        use_case['handler'] = handler_map[use_case['handler']]
    
    return use_cases

def website_summarizer(url: str, client: LLMClient, model: str) -> str:
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
    try:
        website = Website(url)
        use_case = USE_CASES["website_summarizer"]
        user_prompt = use_case["user_prompt_template"].format(
            url=website.url,
            title=website.title,
            text=website.text
        )
        return client.get_completion(use_case["system_prompt"], user_prompt, model=model)
    except Exception as e:
        raise Exception(f"Error in summarization: {str(e)}")

def company_brochure(url: str, client: LLMClient, model: str) -> str:
    """
    Fetches content from a company website and its relevant subpages to generate a professional brochure.
    Saves the result as both markdown and HTML.
    
    Args:
        url (str): The URL of the company website
        client (LLMClient): The initialized LLM client to use
        model (str): The model to use for generation
        
    Returns:
        str: The generated brochure
        
    Raises:
        Exception: If there's an error with website fetching or API communication
    """
    try:
        # Fetch main website content
        main_site = Website(url)
        
        # Gather content from all available links
        additional_content = []
        print("\nGathering content from internal pages...")
        for link in main_site.links:
            try:
                subpage = Website(link)
                additional_content.append(
                    f"\n=== PAGE ===\n"
                    f"URL: {link}\n"
                    f"Title: {subpage.title}\n"
                    f"Content:\n{subpage.text[:1000]}\n"
                    f"============="
                )
            except Exception:
                continue
    
        # Combine all content with clear URL sections
        combined_content = (
            f"=== MAIN PAGE ===\n"
            f"URL: {main_site.url}\n"
            f"Title: {main_site.title}\n"
            f"Content:\n{main_site.text}\n"
            f"===============\n"
            f"\n=== SUBPAGES ===\n"
            f"{''.join(additional_content)}"
        )
        
        # Print parse summary before LLM processing
        print("\nParse Summary:")
        print(f"Title: {main_site.title}")
        print(f"URL: {main_site.url}")
        print(f"Characters extracted: {len(main_site.text)}")
        
        for link in main_site.links:
            try:
                subpage = Website(link)
                print(f"\nTitle: {subpage.title}")
                print(f"URL: {link}")
                print(f"Characters extracted: {len(subpage.text)}")
            except Exception:
                continue
                
        use_case = USE_CASES["company_brochure"]
        user_prompt = use_case["user_prompt_template"].format(
            url=main_site.url,
            title=main_site.title,
            text=combined_content
        )
        result = client.get_completion(use_case["system_prompt"], user_prompt, model=model)
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate safe filename from URL
        domain = urlparse(url).netloc
        safe_filename = domain.replace(".", "_")
        
        # Save markdown file
        md_path = output_dir / f"{safe_filename}_brochure.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(result)
        
        # Convert to HTML and save
        html_content = markdown.markdown(result)
        html_path = output_dir / f"{safe_filename}_brochure.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Company Brochure - {domain}</title>
    <style>
        body {{
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
""")
        
        print(f"\nFiles saved:")
        print(f"Markdown: {md_path}")
        print(f"HTML: {html_path}")
        
        return result
    
    except Exception as e:
        raise Exception(f"Error in brochure generation: {str(e)}")

def joke_generator(topic: str, client: LLMClient, model: str) -> str:
    """
    Generates a joke based on the given topic using the specified LLM client.
    
    Args:
        topic (str): The topic to generate a joke about
        client (LLMClient): The initialized LLM client to use
        model (str): The model to use for generation
        
    Returns:
        str: The generated joke
        
    Raises:
        Exception: If there's an error with API communication
    """
    try:
        use_case = USE_CASES["joke_generator"]
        user_prompt = use_case["user_prompt_template"].format(topic=topic)
        return client.get_completion(use_case["system_prompt"], user_prompt, model=model)
    except Exception as e:
        raise Exception(f"Error in joke generation: {str(e)}")

# Load use cases at module level
USE_CASES = load_use_cases()

if __name__ == "__main__":
    # Display available use cases
    print("Available Use Cases:")
    print("-----------------")
    for i, (key, use_case) in enumerate(USE_CASES.items(), 1):
        print(f"{i}. {use_case['name']}")
    
    # Get use case selection
    choice = input(f"\nSelect use case (1-{len(USE_CASES)}) [default: 1]: ").strip() or "1"
    use_case_key = list(USE_CASES.keys())[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(USE_CASES) else "website_summarizer"
    use_case = USE_CASES[use_case_key]
    
    print(f"\n{use_case['name']}")
    print("-" * len(use_case['name']))
    
    # Get use case specific input
    input_value = input(use_case["input_prompt"])
    
    # Display available clients
    clients = ["openai", "ollama", "anthropic"]
    print("\nSelect the LLM client to use:")
    for i, client in enumerate(clients, 1):
        print(f"{i}. {client}")
    
    choice = input(f"\nEnter number (1-{len(clients)}) [default: 1]: ").strip() or "1"
    client_type = clients[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(clients) else "openai"
    
    try:
        # Initialize AI client and get available models
        if client_type.lower() == "openai":
            client = OpenAIClient()
        elif client_type.lower() == "ollama":
            client = OllamaClient()
        elif client_type.lower() == "anthropic":
            client = AnthropicClient()
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
        print("\nProcessing...")
        # Call the handler function for the selected use case
        result = use_case["handler"](input_value, client, model)
        print(f"\nResults for {use_case['name']}:")
        print("-" * (20 + len(use_case['name'])))
        print(result)
    except Exception as e:
        print(f"\nError: {str(e)}")