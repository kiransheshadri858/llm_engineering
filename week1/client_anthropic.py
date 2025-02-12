import anthropic
import os
from typing import List
from llm_client import LLMClient

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str = None):
        """
        Initialize Anthropic client.
        
        Args:
            api_key (str, optional): Anthropic API key. If not provided, will look for ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("No API key provided. Set ANTHROPIC_API_KEY environment variable or pass key to constructor.")
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """
        Validate connection to Anthropic API.
        
        Raises:
            ConnectionError: If cannot connect to Anthropic API
        """
        try:
            # Simple validation by listing models
            self.client.models.list()
        except Exception as e:
            raise ConnectionError(f"Cannot connect to Anthropic API: {str(e)}")
    
    def get_completion(self, 
                      system_prompt: str,
                      user_prompt: str,
                      model: str,
                      temperature: float = 0.7) -> str:
        """
        Get completion from Anthropic API.
        
        Args:
            system_prompt (str): The system prompt to set context
            user_prompt (str): The user prompt for completion
            model (str): Model to use (e.g., "claude-3-opus-20240229")
            temperature (float, optional): Sampling temperature. Defaults to 0.7
            
        Returns:
            str: The generated completion text
            
        Raises:
            Exception: If there's an error in API communication
        """
        try:
            message = self.client.messages.create(
                model=model,
                max_tokens=1000,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": user_prompt
                }],
                temperature=temperature
            )
            return message.content[0].text
            
        except Exception as e:
            raise Exception(f"Error getting completion from Anthropic: {str(e)}")
    
    def list_models(self) -> List[str]:
        """
        Get list of available models from Anthropic.
        
        Returns:
            List[str]: List of model identifiers available for use
            
        Raises:
            Exception: If there's an error in API communication
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            raise Exception(f"Failed to fetch Anthropic models: {str(e)}") 