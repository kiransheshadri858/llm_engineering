import requests
from typing import List, Dict, Any
from llm_client import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url (str, optional): Ollama API base URL. Defaults to local instance.
        """
        self.base_url = base_url.rstrip('/')
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """
        Validate connection to Ollama server.
        
        Raises:
            ConnectionError: If cannot connect to Ollama server
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama server: {str(e)}")
    
    def _ensure_model(self, model: str) -> None:
        """
        Ensure the requested model is available, pull if not.
        
        Args:
            model (str): Name of the model to check/pull
        """
        try:
            # Check if model exists
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = [tag['name'] for tag in response.json()['models']]
            
            if model not in models:
                print(f"Model {model} not found locally. Pulling from repository...")
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model}
                )
                pull_response.raise_for_status()
                print(f"Successfully pulled {model}")
        except requests.RequestException as e:
            raise Exception(f"Error ensuring model availability: {str(e)}")
    
    def get_completion(self, 
                      system_prompt: str,
                      user_prompt: str,
                      model: str,
                      temperature: float = 0.7) -> str:
        """
        Get completion from Ollama API.
        
        Args:
            system_prompt (str): The system prompt to set context
            user_prompt (str): The user prompt for completion
            model (str): Model to use
            temperature (float, optional): Sampling temperature. Defaults to 0.7
            
        Returns:
            str: The generated completion text
            
        Raises:
            Exception: If there's an error in API communication
        """
        try:
            # Ensure model is available
            self._ensure_model(model)
            
            # Combine system and user prompts for Ollama
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Prepare the request payload
            payload = {
                "model": model,
                "prompt": combined_prompt,
                "temperature": temperature,
                "stream": False
            }
            
            # Make the request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            # Extract the response
            result = response.json()
            return result['response']
            
        except requests.RequestException as e:
            raise Exception(f"Error getting completion from Ollama: {str(e)}")
    
    def list_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List[str]: List of model identifiers available locally
            
        Raises:
            Exception: If there's an error in API communication
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [tag['name'] for tag in response.json()['models']]
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch Ollama models: {str(e)}") 