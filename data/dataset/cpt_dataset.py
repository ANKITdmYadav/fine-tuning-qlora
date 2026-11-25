import os
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
model_name="Qwen/Qwen2.5-3B-Instruct"
input_file="C:\\Users\\komma\\Desktop\\QLORA\\CPT\\train.txt"
output_file="C:\\Users\\komma\\Desktop\\QLORA\\HF_DATASET"
Path(output_file).mkdir(parents=True, exist_ok=True)
chunk_size=512
min_chunk_size=128
class CPTDatasetBuilder:
    def __init__(self,model_name,input_file,output_file,chunk_size,min_chunk_size):
        self.model_name=model_name
        self.input_file=input_file
        self.output_file=output_file
        self.chunk_size=chunk_size
        self.min_chunk_size=min_chunk_size
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token=self.tokenizer.eos_token
    def load_text(self):
        with open(self.input_file,"r",encoding="utf-8") as f:
            text=f.read()
        return text
    def tokenize(self,text):
        token_ids=self.tokenizer.encode(text,add_special_tokens=False)
        return token_ids
    def create_chunks(self, token_ids):
        chunks = []
        for i in range(0,len(token_ids),self.chunk_size):
            chunk = token_ids[i:i + self.chunk_size]
            if len(chunk) < self.min_chunk_size:
                continue
            chunks.append({"input_ids": chunk})
        return chunks
    def create_dataset(self,chunks):
        dataset=Dataset.from_list(chunks)
        return dataset
    def save_dataset(self,dataset):
        dataset.save_to_disk(str(self.output_file))
    def process(self):
        text = self.load_text()
        token_ids = self.tokenize(text)
        chunks = self.create_chunks(token_ids)
        dataset = self.create_dataset(chunks)
        self.save_dataset(dataset )
        # print(dataset)
        # print(dataset[0])
builder=CPTDatasetBuilder(model_name,input_file,output_file,chunk_size,min_chunk_size)
builder.process()