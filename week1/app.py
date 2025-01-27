import os
from dotenv import load_dotenv
from openai import OpenAI
from website import Website

load_dotenv()  # This loads the environment variables from the .env file.

def summarize(url: str) -> str:
    """
    Fetches content from a URL and generates a summary using OpenAI.
    
    Args:
        url (str): The URL of the website to summarize
        
    Returns:
        str: The generated summary
        
    Raises:
        Exception: If there's an error with website fetching or API communication
    """
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    
    if (api_key is None or 
        not api_key.strip() or 
        ' ' in api_key or 
        '\t' in api_key or
        not api_key.startswith('sk-')):
        raise ValueError("Invalid API key: Key is either None, empty, contains spaces/tabs, or has invalid format.")
    
    client = OpenAI(api_key=api_key)
    
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
    
    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    print("Website Summarizer")
    print("-----------------")
    url_name = input("Enter the website URL to summarize: ")
    
    try:
        print("\nFetching and analyzing content...")
        summary = summarize(url_name)
        print(f"\nWebsite Summary for {url_name}:")
        print("-" * (20 + len(url_name)))
        print(summary)
    except Exception as e:
        print(f"\nError: {str(e)}")

