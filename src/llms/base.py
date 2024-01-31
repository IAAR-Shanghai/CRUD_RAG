import copy
from abc import ABC, abstractmethod

from loguru import logger

class BaseLLM(ABC):
    def __init__(
            self, 
            model_name: str = "gpt-3.5-turbo", 
            temperature: float = 1.0, 
            max_new_tokens: int = 1024, 
            top_p: float = 0.9,
            top_k: int = 5,
            **more_params
        ):
        self.params = {
            'model_name': model_name if model_name else self.__class__.__name__,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'top_k': top_k,
            **more_params
        }

    def update_params(self, inplace: bool = True, **params):
        if inplace:
            self.params.update(params)
            return self
        else:
            new_obj = copy.deepcopy(self)
            new_obj.params.update(params)
            return new_obj

    @abstractmethod
    def request(self, query:str) -> str:
        return ''

    def safe_request(self, query: str) -> str:
        """Safely make a request to the language model, handling exceptions."""
        try:
            response = self.request(query)
        except Exception as e:
            logger.warning(repr(e))
            response = ''
        return response

