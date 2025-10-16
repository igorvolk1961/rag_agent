"""
Large Language Model client using LangChain with DeepSeek
"""

import logging
from typing import Optional, Dict, Any, List
import os

from langchain_deepseek import ChatDeepSeek
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class LLMClient:
    """Client for interacting with DeepSeek using LangChain"""

    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None,
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

        # Get API key
        if not api_key:
            api_key = os.getenv("DEEPSEEK_API_KEY")

        if not api_key:
            self.logger.warning("No DeepSeek API key provided")

        try:
            self.llm = ChatDeepSeek(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Create default prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain enough information to answer the question, say so."),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ])

            self.logger.info(f"Initialized DeepSeek LLM client with model: {model}")
        except Exception as e:
            self.logger.error(f"Error initializing DeepSeek LLM client: {e}")
            raise

    def generate_response(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate response using LLM"""
        try:
            # Use provided parameters or defaults
            current_temperature = temperature if temperature is not None else self.temperature
            current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

            # Create LLM with custom parameters if needed
            if temperature is not None or max_tokens is not None:
                llm = ChatDeepSeek(
                    model=self.model,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    temperature=current_temperature,
                    max_tokens=current_max_tokens
                )
            else:
                llm = self.llm

            # Prepare context
            context_text = context if context else "No context provided."

            # Create prompt
            prompt = self.prompt_template.format_messages(
                context=context_text,
                question=query
            )

            # Generate response
            response = llm.invoke(prompt)
            answer = response.content.strip()

            self.logger.info(f"Generated response for query: {query[:50]}...")
            return answer

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generic chat completion"""
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:  # user
                    langchain_messages.append(HumanMessage(content=content))

            # Use custom parameters if provided
            if temperature is not None or max_tokens is not None:
                llm = ChatDeepSeek(
                    model=self.model,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens
                )
            else:
                llm = self.llm

            response = llm.invoke(langchain_messages)
            return response.content.strip()

        except Exception as e:
            self.logger.error(f"Error in chat completion: {e}")
            raise

    def get_langchain_llm(self) -> BaseLanguageModel:
        """Get the underlying LangChain LLM object"""
        return self.llm

    def get_prompt_template(self) -> ChatPromptTemplate:
        """Get the prompt template"""
        return self.prompt_template

    def create_custom_prompt(self, system_message: str, human_template: str) -> ChatPromptTemplate:
        """Create a custom prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_template)
        ])

    def stream_response(
        self,
        query: str,
        context: Optional[str] = None
    ):
        """Stream response for real-time output"""
        try:
            context_text = context if context else "No context provided."

            prompt = self.prompt_template.format_messages(
                context=context_text,
                question=query
            )

            for chunk in self.llm.stream(prompt):
                yield chunk.content

        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            yield f"Error streaming response: {str(e)}"
