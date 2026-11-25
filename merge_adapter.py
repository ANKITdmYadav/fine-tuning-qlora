import os
from models.loader import load_full_model,load_tokenizer
from peft import PeftModel
model=load_full_model()
tokenizer=load_tokenizer()
OFFLOAD_DIR = r"C:\Users\komma\Desktop\QLORA\offload"
os.makedirs(OFFLOAD_DIR, exist_ok=True)
OUTPUT_DIR="C:\\Users\\komma\\Desktop\\QLORA\\Merged_Model"
os.makedirs(OUTPUT_DIR,exist_ok=True)
lora_path="C:\\Users\\komma\\Desktop\\QLORA\\Medical_Qwen"
new_model=PeftModel.from_pretrained(model,lora_path)
merged_model = new_model.merge_and_unload(safe_merge=True,offload_folder=OFFLOAD_DIR,low_cpu_mem_usage=True)
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Merged Succesfully")


