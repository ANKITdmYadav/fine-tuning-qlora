import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
model_name="Qwen/Qwen2.5-3B-Instruct"
bits_config=BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True

)
# loader.py

def load_quantized_model():
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bits_config,device_map="auto")
    return quantized_model
def load_full_model():
    return AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16,device_map={"":"cpu"})
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token=tokenizer.eos_token
    return tokenizer