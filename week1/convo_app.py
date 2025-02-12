import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import json
from pathlib import Path
from client_open_ai import OpenAIClient
from client_ollama import OllamaClient
from client_anthropic import AnthropicClient
from llm_client import LLMClient

load_dotenv()

class ModelConversation:
    def __init__(
        self, 
        model1: Tuple[LLMClient, str, str, str],  # (client, model_name, system_prompt, character_name)
        model2: Tuple[LLMClient, str, str, str],
        initial_prompt: str,
        turns: int = 5,
        max_chars: int = 500
    ):
        self.model1 = model1
        self.model2 = model2
        self.initial_prompt = initial_prompt
        self.turns = turns
        self.max_chars = max_chars
        self.conversation_history: List[Dict[str, str]] = []
    
    def _format_conversation_history(self) -> str:
        """Format the conversation history for prompt injection."""
        formatted = ""
        for entry in self.conversation_history:
            formatted += f"{entry['speaker']}: {entry['message']}\n"
        return formatted
    
    def run_conversation(self) -> List[Dict[str, str]]:
        """Run the conversation between the two models."""
        # Start with the initial prompt
        self.conversation_history.append({
            "speaker": "Moderator",
            "message": self.initial_prompt
        })
        print(f"\nModerator: {self.initial_prompt}")
        
        # Alternate between models for the specified number of turns
        for turn in range(self.turns):
            # Determine current model
            current_model = self.model1 if turn % 2 == 0 else self.model2
            speaker = current_model[3]  # Use character name
            
            # Create prompt with conversation history
            conversation_text = self._format_conversation_history()
            user_prompt = (
                f"Here's the conversation so far:\n{conversation_text}\n"
                f"Please provide your response. Keep it under {self.max_chars} characters:"
            )
            
            # Get response from current model
            try:
                response = current_model[0].get_completion(
                    system_prompt=current_model[2],
                    user_prompt=user_prompt,
                    model=current_model[1]
                )
                
                # Truncate response if it exceeds max_chars
                if len(response) > self.max_chars:
                    response = response[:self.max_chars] + "..."
                
                # Add response to history
                self.conversation_history.append({
                    "speaker": speaker,
                    "message": response
                })
                
                # Print real-time updates (simplified format)
                print(f"\n{speaker}: {response}")
                
            except Exception as e:
                print(f"\nError getting response from {speaker}: {str(e)}")
                break
        
        return self.conversation_history

def select_client_and_model() -> Tuple[LLMClient, str]:
    """Helper function to select a client and model."""
    # Available clients
    clients = {
        "1": ("OpenAI", OpenAIClient),
        "2": ("Ollama", OllamaClient),
        "3": ("Anthropic", AnthropicClient)
    }
    
    # Display client options
    print("\nSelect client:")
    for key, (name, _) in clients.items():
        print(f"{key}. {name}")
    
    # Get client selection
    client_choice = input("Enter choice (1-3): ").strip()
    client_name, client_class = clients.get(client_choice, clients["1"])
    
    try:
        # Initialize client
        client = client_class()
        
        # Get available models
        available_models = client.list_models()
        
        # Display model options
        print(f"\nAvailable {client_name} models:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
        
        # Get model selection
        model_choice = input(f"Select model (1-{len(available_models)}): ").strip()
        model = available_models[int(model_choice) - 1]
        
        return client, model
        
    except Exception as e:
        print(f"Error initializing {client_name} client: {str(e)}")
        raise

def main():
    print("\n=== AI Debate Simulator ===")
    print("Create a debate between two AI models with unique personalities!")
    
    # Select first model
    print("\n--- First Participant Setup ---")
    client1, model1 = select_client_and_model()
    character_name1 = input("\nGive this participant a name (e.g., 'Chef Gordon'): ").strip() or "Participant 1"
    system_prompt1 = input(
        f"\nDescribe {character_name1}'s personality (e.g., 'You are a passionate food critic who's very opinionated'): "
    )
    
    # Select second model
    print("\n--- Second Participant Setup ---")
    client2, model2 = select_client_and_model()
    character_name2 = input("\nGive this participant a name (e.g., 'Professor Cuisine'): ").strip() or "Participant 2"
    system_prompt2 = input(
        f"\nDescribe {character_name2}'s personality (e.g., 'You are a traditional culinary purist who values proper categorization'): "
    )
    
    # Get debate parameters with more descriptive prompts
    print("\n--- Debate Setup ---")
    debate_topic = input(
        "\nWhat should they debate? (e.g., 'Is a hot dog a sandwich?'): "
    )
    
    # Select who starts
    print("\nWho should start the debate?")
    print(f"1. {character_name1} ({model1})")
    print(f"2. {character_name2} ({model2})")
    starter = input("Enter 1 or 2 (default: 1): ").strip() or "1"
    
    turns = int(input(
        "\nHow many exchanges should the debate have? (default: 5): "
    ) or "5")
    
    max_chars = int(input(
        "\nMaximum characters per response to keep arguments concise (default: 500): "
    ) or "500")
    
    # Create initial prompt based on who starts
    if starter == "2":
        model1, model2 = model2, model1
        client1, client2 = client2, client1
        system_prompt1, system_prompt2 = system_prompt2, system_prompt1
        character_name1, character_name2 = character_name2, character_name1
    
    # Create and run conversation
    print("\n=== Starting Debate ===")
    print(f"Topic: {debate_topic}")
    print(f"First Speaker: {character_name1} ({model1})")
    print(f"Second Speaker: {character_name2} ({model2})")
    print(f"Number of exchanges: {turns}")
    print(f"Character limit: {max_chars}")
    print("\n--- Begin Debate ---")
    
    conversation = ModelConversation(
        model1=(client1, model1, system_prompt1, character_name1),
        model2=(client2, model2, system_prompt2, character_name2),
        initial_prompt=f"Let's debate this topic: {debate_topic}. Please provide your opening argument.",
        turns=turns,
        max_chars=max_chars
    )
    
    history = conversation.run_conversation()
    
    # Save conversation with debate-specific structure
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"debate_{len(list(output_dir.glob('debate_*.json'))) + 1}.json"
    with open(output_file, "w") as f:
        json.dump({
            "debate_topic": debate_topic,
            "participants": {
                "first_speaker": {
                    "name": character_name1,
                    "model": model1,
                    "personality": system_prompt1
                },
                "second_speaker": {
                    "name": character_name2,
                    "model": model2,
                    "personality": system_prompt2
                }
            },
            "exchanges": turns,
            "char_limit": max_chars,
            "debate_log": history
        }, f, indent=2)
    
    print(f"\n=== Debate Saved ===")
    print(f"File: {output_file}")

if __name__ == "__main__":
    main() 