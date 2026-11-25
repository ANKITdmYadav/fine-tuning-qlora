import os
import torch
import pandas as pd
from Evaluation.questions import QUESTIONS
from peft import PeftModel
from merge_adapter import merged_model
import time
OUTPUT_DIR= "C:\\Users\\komma\\Desktop\\QLORA\\output"
OUTPUT_FILE=os.path.join(OUTPUT_DIR,"merge_results.csv")
from models.loader import load_tokenizer
tokenizer=load_tokenizer()
merge_model=merged_model()
def generate(question):
    messages = [{
        "role": "user",
        "content": question
    }]
    text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inputs = tokenizer(text,return_tensors="pt").to(merge_model.device)
    outputs = merge_model.generate(**inputs,max_new_tokens=200,do_sample=False,pad_token_id=tokenizer.eos_token_id)
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
print("Merged Model Model Evaluation Completed")