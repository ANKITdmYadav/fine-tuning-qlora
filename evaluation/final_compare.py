import os
import pandas as pd
BASE_FILE = r"C:\Users\komma\Desktop\QLORA\output\base_results.csv"
LORA_FILE = r"C:\Users\komma\Desktop\QLORA\output\lora_results.csv"
LORA_BASE_FILE = r"C:\Users\komma\Desktop\QLORA\output\merged_results.csv"
OUTPUT_DIR=r"C:\Users\komma\Desktop\QLORA\output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR,"final_compare_results.csv")
base_df=pd.read_csv(BASE_FILE)
lora_df=pd.read_csv(LORA_FILE)
lora_base_df=pd.read_csv(LORA_BASE_FILE)
base_df=base_df.rename(columns={"Answer":"Base Model Answer"})
lora_df=lora_df.rename(columns={"Answer":"LoRA Model Answer"})
lora_base_df=lora_base_df.rename(columns={"Answer":"Merged Model Answer"})
comparison_df=pd.merge(base_df,lora_df,on="Question",how="inner")
comparison_df=pd.merge(comparison_df,lora_base_df,on="Question",how="inner")
comparison_df.to_csv(OUTPUT_FILE,index=False)


