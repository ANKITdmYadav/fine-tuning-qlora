import os
import torch
import pandas as pd
from Evaluation.questions import QUESTIONS
from peft import PeftModel
import time
OUTPUT_DIR= "C:\\Users\\komma\\Desktop\\QLORA\\output"
OUTPUT_FILE=os.path.join(OUTPUT_DIR,"lora_results.csv")
from models.loader import load_quantized_model,load_tokenizer
model=load_quantized_model()
tokenizer=load_tokenizer()
LORA_PATH="C:\\Users\\komma\\Desktop\\QLORA\\Medical_Qwen"
lora_model=PeftModel.from_pretrained(model, LORA_PATH)
def generate(question):
    messages = [{
        "role": "user",
        "content": question
    }]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    device = next(lora_model.parameters()).device
    inputs = tokenizer(text,return_tensors="pt").to(device)
    outputs = lora_model.generate(**inputs,max_new_tokens=200,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer=tokenizer.decode(generated_tokens,skip_special_tokens=True)
    answer=" ".join(answer.split())
    return answer
results = []
for question in QUESTIONS:
    print(question)
    try:
        start=time.time()
        answer = generate(question)
        end=time.time()
    except Exception as e:
        answer=str(e)
    results.append({"Question": question, "Answer": answer,"Inference_time(sec)":round(end-start,2)})
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print("lora Model Evaluation Completed")
