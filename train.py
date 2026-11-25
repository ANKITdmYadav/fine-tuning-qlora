import torch
from datasets import load_from_disk
from transformers import (DataCollatorForLanguageModeling,Trainer,TrainingArguments)
from models.peft_model import model
from models.loader import load_tokenizer
tokenizer=load_tokenizer()
DATASET_PATH="C:\\Users\\komma\\Desktop\\QLORA\\HF_DATASET"
OUTPUT_DIR = "C:\\Users\\komma\\Desktop\\QLORA\\Medical_Qwen"
train_dataset=load_from_disk(DATASET_PATH)
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=5,
    save_steps=100,
    save_strategy="epoch",
    logging_strategy="steps",
    save_total_limit=2,
    seed=42,
    fp16=True,
    optim="paged_adamw_8bit",
       report_to="none")
trainer = Trainer(model=model,args=training_args,train_dataset=train_dataset,data_collator=data_collator)
trainer.train()
trainer.save_state()
print("Saving LoRA Adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training completed Successfully")
