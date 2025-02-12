import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from typing import List

class Website:
    def __init__(self, url: str):
        """
        Initialize Website with URL and fetch its content.
        
        Args:
            url (str): The URL to fetch
            
        Raises:
            ValueError: If URL is invalid
            ConnectionError: If website cannot be reached
            Exception: For other errors during fetching
        """
        # Ensure URL has proper scheme
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        try:
            response = requests.get(url, timeout=10)  # Add timeout
            response.raise_for_status()  # Raise error for bad status codes
            
            self.url = url
            soup = BeautifulSoup(response.text, 'html.parser')
            self.title = soup.title.string if soup.title else "No title found"
            self.text = self._extract_text(soup)
            self.links = self._extract_links(soup)
            
        except requests.exceptions.MissingSchema:
            raise ValueError(f"Invalid URL format: {url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to website: {url}. Please check the URL and your internet connection.")
        except requests.exceptions.Timeout:
            raise ConnectionError(f"Connection timed out while trying to reach {url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching website content: {str(e)}")
    
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

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extract readable text from BeautifulSoup object.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            str: Extracted text content
        """
        # Remove script and style elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
            
        # Get text and clean it up
        text = soup.get_text()
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text 

    def _extract_links(self, soup: BeautifulSoup) -> List[str]:
        """
        Extract all hyperlinks from the webpage.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            List[str]: List of absolute URLs found in the page
        """
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip javascript and mailto links
            if href.startswith(('javascript:', 'mailto:', '#')):
                continue
            
            # Convert to absolute URL if relative
            absolute_url = urljoin(self.url, href)
            
            # Remove fragments
            absolute_url = absolute_url.split('#')[0]
            
            if absolute_url not in links:  # Avoid duplicates
                links.append(absolute_url)
                
        return links 