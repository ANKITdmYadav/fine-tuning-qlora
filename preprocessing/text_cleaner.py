import os
import re
from pathlib import Path
import unicodedata

# # Input_Files="C:\\Users\\komma\\Desktop\\QLORA\\Output"
# # for filename in os.listdir(Input_Files):
# #     if filename.endswith(".txt"):
# #         txt_path = os.path.join(Input_Files, filename)
# #         with open(txt_path, "r", encoding="utf-8") as f:
# #             text = f.read()
# #         text="".join(line.strip() for line in text.splitlines())
# Input_File="C:\\Users\\komma\\Desktop\\QLORA\\Extracted_text\\DIABETES.txt"

# with open(Input_File, "r", encoding="utf-8") as f:
#     text = f.read()
# text=re.sub(r"Page \s+\d+","",text)
# text=re.sub(r"copyright.*","",text)
# text=re.sub(r"All rights reserved.*","",text)
# text=re.sub(r"www.*","",text)
# text=re.sub(r"http://.*","",text)
# text=" ".join(text.split())
# print(text)
input_dir = "C:\\Users\\komma\\Desktop\\QLORA\\Extracted_text"
output_dir = "C:\\Users\\komma\\Desktop\\QLORA\\Cleaned_text"
Path(output_dir).mkdir(parents=True, exist_ok=True)
class TextCleaner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
    def remove_page_numbers(self, text):
        text=re.sub(r"Page\s*[:=]?\s*\d+", "", text,flags=re.IGNORECASE)
        text=re.sub(r"^\d+$", "", text, flags=re.MULTILINE)#MULTILINE is used to make the regex match the start and end of each line, rather than the start and end of the entire string.
        return text
    def remove_copyright(self, text):
        text = re.sub(r"copyright.*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"All rights reserved.*", "", text, flags=re.IGNORECASE)#IGNORECASE is used to make the regex case-insensitive, so it will match "All rights reserved" regardless of capitalization.
        text=re.sub(r"©.*","",text)
        return text
    def remove_urls(self, text):
        text=re.sub(r"https?://\S+|www\.\S+","",text)
        text=re.sub(r'https?://(www\.)?(\w+)(\.\w+)',"",text)
        return text
    def normalize_unicode(self, text):
        return unicodedata.normalize("NFKC", text)
    def remove_emails(self,text):
        text=re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z-]+\.(com|edu|net)", "", text)
        text=re.sub(r"\S+@\S+", "", text)
        return text
    def remove_phone_numbers(self,text):
        text=re.sub(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',"",text)
        text=re.sub(r'\(\+\d{2}\s\d{2}\)\s\d{3}\s\d{2}\s\d{2}',"",text)
        text=re.sub(r"\+?\d[\d\s\-\(\)]{7,}","",text)
        return text
    def remove_fig_captions(self,text):
        text=re.sub(r"Figure\s+\d+.*","",text,flags=re.IGNORECASE)
        text=re.sub(r"Fig\s+\d+.*","",text,flags=re.IGNORECASE)
        text=re.sub(r"Table\s+\d+.*","",text,flags=re.IGNORECASE)
        text=re.sub(r"Image\s+\d+.*","",text,flags=re.IGNORECASE)
        return text
    def remove_extra_whitespace(self, text):
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n+", "\n", text)
        return text
    def remove_equal(self,text):
        text=re.sub(r'=+',"",text)
        return text
    def remove_boxes(self,text):
        text=re.sub(r'[\u2500-\u257F]+',"",text)
        return text
Cleaner=TextCleaner(input_dir, output_dir)
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        txt_path = Path(input_dir) / filename
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
            text=Cleaner.normalize_unicode(text)
            text=Cleaner.remove_page_numbers(text)
            text=Cleaner.remove_copyright(text)
            text=Cleaner.remove_urls(text)
            text=Cleaner.remove_emails(text)
            text=Cleaner.remove_phone_numbers(text)
            text=Cleaner.remove_fig_captions(text)
            text=Cleaner.remove_boxes(text)
            text=Cleaner.remove_equal(text)
            text=Cleaner.remove_extra_whitespace(text)
        output_path = Path(output_dir) / filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
print(f"Cleaned text saved to: {output_path}")