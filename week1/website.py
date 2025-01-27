import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

class Website:
    def __init__(self, url: str):
        self.url = self._normalize_url(url)
        self.title = ""
        self.text = ""
        self._fetch_and_parse()
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize the URL by ensuring it has a proper scheme and format.
        
        Args:
            url (str): The input URL
            
        Returns:
            str: Normalized URL with proper scheme
            
        Raises:
            ValueError: If URL is invalid or empty
        """
        if not url:
            raise ValueError("URL cannot be empty")
        
        url = url.strip()
        
        # Add https if no scheme is present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Parse the URL to validate its format
        parsed = urlparse(url)
        if not parsed.netloc:
            raise ValueError("Invalid URL format")
            
        return url
    
    def _fetch_and_parse(self):
        try:
            # Send GET request to the URL
            response = requests.get(self.url)
            response.raise_for_status()
            
            # Create BeautifulSoup object
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            self.title = title_tag.text.strip() if title_tag else ""
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'head', 'header', 'footer', 'nav']):
                element.decompose()
            
            # Get text content and clean it
            text = soup.get_text()
            
            # Clean up the text: remove extra whitespace and empty lines
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            self.text = ' '.join(chunk for chunk in chunks if chunk)
            
        except requests.RequestException as e:
            raise Exception(f"Error fetching website content: {str(e)}") 