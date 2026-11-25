import torch
from peft import (prepare_model_for_kbit_training,LoraConfig,get_peft_model)
from models.loader import load_quantized_model,load_tokenizer
model=load_quantized_model()
tokenizer=load_tokenizer()
model=prepare_model_for_kbit_training(model)
lora_config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj"])
model=get_peft_model(model,lora_config)
model.print_trainable_parameters()
