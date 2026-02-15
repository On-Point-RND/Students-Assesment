"""
AI Agents for document evaluation.

This module provides different AI agent classes for evaluating documents.
Currently supports:
1. DeepSeek API (via OpenAI client)
2. OpenRouter API
3. Placeholder agent for testing
"""

import time
from typing import Optional, Dict, Any
import json


class BaseInferer:
    """Base class for all inference agents."""
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, 
                 model: Optional[str] = None, **kwargs):
        """
        Initialize the base inferer.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API (optional)
            model: Model name to use (optional)
            **kwargs: Additional model-specific parameters
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.kwargs = kwargs
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the API client. Override in subclasses."""
        pass
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """
        Send a request to the AI model and get response.
        
        Args:
            system_prompt: System prompt/instructions
            user_input: User input/message
            
        Returns:
            Response text from the model
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model})"


class DeepSeekInferer(BaseInferer):
    """Inferer for DeepSeek API using OpenAI client."""
    
    def _initialize_client(self):
        """Initialize OpenAI client for DeepSeek."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key.strip(),
                base_url=self.base_url or "https://api.deepseek.com/v1"
            )
            self.model = self.model or "deepseek-chat"
        except ImportError:
            raise ImportError(
                "OpenAI package is required for DeepSeekInferer. "
                "Install with: pip install openai"
            )
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """Forward request to DeepSeek API."""
        if not self.client:
            self._initialize_client()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.3,
                stream=False,
                **self.kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"DeepSeek API error: {e}")


class OpenRouterInferer(BaseInferer):
    """Inferer for OpenRouter API."""
    
    def _initialize_client(self):
        """Initialize OpenAI client for OpenRouter."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key.strip(),
                base_url=self.base_url or "https://openrouter.ai/api/v1"
            )
            self.model = self.model or "openai/gpt-3.5-turbo"
        except ImportError:
            raise ImportError(
                "OpenAI package is required for OpenRouterInferer. "
                "Install with: pip install openai"
            )
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """Forward request to OpenRouter API."""
        if not self.client:
            self._initialize_client()
        
        try:
            # OpenRouter requires specific headers
            extra_headers = {
                "HTTP-Referer": "https://github.com/your-repo",  # Update with your repo
                "X-Title": "Document Evaluator",  # Your app name
            }
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.3,
                stream=False,
                extra_headers=extra_headers,
                **self.kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise Exception(f"OpenRouter API error: {e}")


class AnthropicInferer(BaseInferer):
    """Inferer for Anthropic Claude API."""
    
    def _initialize_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key.strip())
            self.model = self.model or "claude-3-haiku-20240307"
        except ImportError:
            raise ImportError(
                "Anthropic package is required for AnthropicInferer. "
                "Install with: pip install anthropic"
            )
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """Forward request to Anthropic API."""
        if not self.client:
            self._initialize_client()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_input}
                ],
                **self.kwargs
            )
            return response.content[0].text.strip()
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}")


class GoogleInferer(BaseInferer):
    """Inferer for Google Gemini API."""
    
    def _initialize_client(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key.strip())
            self.client = genai
            self.model = self.model or "gemini-pro"
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required for GoogleInferer. "
                "Install with: pip install google-generativeai"
            )
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """Forward request to Google Gemini API."""
        if not self.client:
            self._initialize_client()
        
        try:
            # Combine system prompt and user input for Gemini
            full_prompt = f"{system_prompt}\n\n{user_input}"
            
            model = self.client.GenerativeModel(self.model)
            response = model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1024,
                },
                **self.kwargs
            )
            return response.text.strip()
        except Exception as e:
            raise Exception(f"Google Gemini API error: {e}")


class PlaceholderInferer(BaseInferer):
    """Placeholder inferer for testing without API calls."""
    
    def __init__(self, api_key: str = "placeholder", **kwargs):
        """Initialize placeholder inferer."""
        super().__init__(api_key, **kwargs)
        self.call_count = 0
        self.responses = kwargs.get('responses', {})
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """Return placeholder response."""
        self.call_count += 1
        
        # Check if we have a predefined response for this input
        for key, response in self.responses.items():
            if key in user_input:
                return response
        
        # Default response - simulate different scores
        import random
        scores = [0, 1, 2, 3, 4, 5]
        weights = [0.1, 0.15, 0.2, 0.25, 0.2, 0.1]  # Weighted toward middle scores
        score = random.choices(scores, weights=weights)[0]
        
        # Simulate API delay
        time.sleep(0.05)
        
        return str(score)


class CachedInferer(BaseInferer):
    """
    Cached inferer that wraps another inferer with caching.
    
    Useful for development and testing to avoid repeated API calls.
    """
    
    def __init__(self, wrapped_inferer: BaseInferer, cache_file: Optional[str] = None):
        """
        Initialize cached inferer.
        
        Args:
            wrapped_inferer: The actual inferer to wrap
            cache_file: Optional file to persist cache (JSON format)
        """
        self.wrapped_inferer = wrapped_inferer
        self.cache_file = cache_file
        self.cache: Dict[str, str] = {}
        
        # Load cache from file if exists
        if cache_file:
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self.cache = {}
    
    def forward(self, system_prompt: str, user_input: str) -> str:
        """Forward request with caching."""
        # Create cache key
        import hashlib
        cache_key = hashlib.md5(f"{system_prompt}{user_input}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Call wrapped inferer
        response = self.wrapped_inferer.forward(system_prompt, user_input)
        
        # Cache the response
        self.cache[cache_key] = response
        
        # Save cache to file if specified
        if self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f)
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
        
        return response
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}
        if self.cache_file and os.path.exists(self.cache_file):
            os.remove(self.cache_file)


# Factory function to create inferers
def create_inferer(inferer_type: str, api_key: str, **kwargs) -> BaseInferer:
    """
    Factory function to create inferers.
    
    Args:
        inferer_type: Type of inferer ('deepseek', 'openrouter', 'anthropic', 
                       'google', 'placeholder', or 'cached')
        api_key: API key for the service
        **kwargs: Additional parameters for the inferer
    
    Returns:
        Configured inferer instance
    """
    inferer_type = inferer_type.lower()
    
    if inferer_type == 'deepseek':
        return DeepSeekInferer(api_key, **kwargs)
    elif inferer_type == 'openrouter':
        return OpenRouterInferer(api_key, **kwargs)
    elif inferer_type == 'anthropic':
        return AnthropicInferer(api_key, **kwargs)
    elif inferer_type == 'google':
        return GoogleInferer(api_key, **kwargs)
    elif inferer_type == 'placeholder':
        return PlaceholderInferer(api_key, **kwargs)
    elif inferer_type == 'cached':
        # For cached inferer, need to specify wrapped inferer
        wrapped_type = kwargs.pop('wrapped_type', 'deepseek')
        wrapped_inferer = create_inferer(wrapped_type, api_key, **kwargs)
        cache_file = kwargs.get('cache_file', 'inferer_cache.json')
        return CachedInferer(wrapped_inferer, cache_file)
    else:
        raise ValueError(f"Unknown inferer type: {inferer_type}")


# Default export for backward compatibility
Inferer = DeepSeekInferer


# Example usage
if __name__ == "__main__":
    # Example 1: DeepSeek
    deepseek = create_inferer(
        'deepseek',
        api_key="your-deepseek-api-key",
        model="deepseek-chat"
    )
    
    # Example 2: OpenRouter with specific model
    openrouter = create_inferer(
        'openrouter',
        api_key="your-openrouter-api-key",
        model="meta-llama/llama-3.1-70b-instruct"
    )
    
    # Example 3: Cached DeepSeek for development
    cached = create_inferer(
        'cached',
        api_key="your-api-key",
        wrapped_type='deepseek',
        cache_file='cache.json'
    )
    
    # Example 4: Placeholder for testing
    placeholder = create_inferer('placeholder', api_key="test")
    
    print("Available inferers:")
    print(f"1. {deepseek}")
    print(f"2. {openrouter}")
    print(f"3. {cached}")
    print(f"4. {placeholder}")