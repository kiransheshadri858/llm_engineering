from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMClient(ABC):
    """Base class for LLM clients."""
    
    @abstractmethod
    def _validate_connection(self) -> None:
        """Validate connection/credentials to the LLM service."""
        pass
    
    @abstractmethod
    def get_completion(self,
                      system_prompt: str,
                      user_prompt: str,
                      model: str,
                      temperature: float = 0.7) -> str:
        """
        Get completion from LLM API.
        
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
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        Get list of available models from the LLM provider.
        
        Returns:
            List[str]: List of model identifiers available for use
            
        Raises:
            Exception: If there's an error in API communication
        """
        pass 