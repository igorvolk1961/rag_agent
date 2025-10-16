"""
Large Language Model client
"""

import logging
from typing import Optional, Dict, Any
import openai


class LLMClient:
    """Client for interacting with Large Language Models"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.logger = logging.getLogger(__name__)
        
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get from environment
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
            else:
                self.logger.warning("No OpenAI API key provided")
        
        self.logger.info(f"Initialized LLM client with model: {model}")
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """Generate response using LLM"""
        try:
            # Prepare prompt
            if context:
                prompt = f"""Context:
{context}

Question: {query}

Answer:"""
            else:
                prompt = f"Question: {query}\n\nAnswer:"
            
            # Make API call
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content.strip()
            self.logger.info(f"Generated response for query: {query[:50]}...")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat_completion(
        self,
        messages: list,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7
    ) -> str:
        """Generic chat completion"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error in chat completion: {e}")
            raise
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            models = openai.Model.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Error getting models: {e}")
            return []
