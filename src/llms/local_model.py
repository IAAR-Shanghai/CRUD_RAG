import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib import import_module

try:
    conf = import_module("src.configs.real_config")
except ImportError:
    conf = import_module("src.configs.config")
from src.llms.base import BaseLLM

class Qwen_7B_Chat(BaseLLM):
    def __init__(self, model_name='qwen_7b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = conf.Qwen_7B_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }

    def request(self, query: str) -> str:
        query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


class Baichuan2_13B_Chat(BaseLLM):
    def __init__(self, model_name='baichuan2_13b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = conf.Baichuan2_13b_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
    
    def request(self, query: str) -> str:
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


class ChatGLM3_6B_Chat(BaseLLM):
    def __init__(self, model_name='chatglm3_6b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = conf.ChatGLM3_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, use_fast=False, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }
    
    def request(self, query: str) -> str:
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response


class Qwen_14B_Chat(BaseLLM):
    def __init__(self, model_name='qwen_14b', temperature=1.0, max_new_tokens=1024):
        super().__init__(model_name, temperature, max_new_tokens)
        local_path = conf.Qwen_14B_local_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(local_path, device_map="auto",
                                                     trust_remote_code=True).eval()
        self.gen_kwargs = {
            "temperature": self.params['temperature'],
            "do_sample": True,
            "max_new_tokens": self.params['max_new_tokens'],
            "top_p": self.params['top_p'],
            "top_k": self.params['top_k'],
        }

    def request(self, query: str) -> str:
        query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
        input_ids = self.tokenizer.encode(query, return_tensors="pt").cuda()
        output = self.model.generate(input_ids, **self.gen_kwargs)[0]
        response = self.tokenizer.decode(
            output[len(input_ids[0]) - len(output):], skip_special_tokens=True)
        return response
    
