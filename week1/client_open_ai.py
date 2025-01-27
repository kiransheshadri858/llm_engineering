import os
from openai import OpenAI
from typing import List, Dict, Any
from llm_client import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI client with API key.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will try to get from environment.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self._validate_connection()
        self.client = OpenAI(api_key=self.api_key)
    
    def _validate_connection(self) -> None:
        """
        Validate the API key format.
        
        Raises:
            ValueError: If API key is invalid
        """
        if (self.api_key is None or 
            not self.api_key.strip() or 
            ' ' in self.api_key or 
            '\t' in self.api_key or
            not self.api_key.startswith('sk-')):
            raise ValueError("Invalid API key: Key is either None, empty, contains spaces/tabs, or has invalid format.")
    
    def get_completion(self, 
                      system_prompt: str,
                      user_prompt: str,
                      model: str,
                      temperature: float = 0.7) -> str:
        """
        Get completion from OpenAI API.
        
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
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error getting completion from OpenAI: {str(e)}")

    def list_models(self) -> List[str]:
        """
        Get list of available models from OpenAI.
        
        Returns:
            List[str]: List of model identifiers available for use
            
        Raises:
            Exception: If there's an error in API communication
        """
        try:
            models = self.client.models.list()
            # Filter to only get the GPT models since those are the ones
            # that work with chat completions
            gpt_models = [
                model.id for model in models 
                if "gpt" in model.id.lower()
            ]
            return gpt_models
        except Exception as e:
            raise Exception(f"Failed to fetch OpenAI models: {str(e)}") 