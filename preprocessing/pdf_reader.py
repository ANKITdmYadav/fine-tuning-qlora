import os
import fitz
#pdf_path="C:/Users/komma/Desktop/QLORA/DataSet/DIABETES.pdf"
# with fitz.open(pdf_path) as pdf:
#     text = "".join(page.get_text() for page in pdf)

# print(text)
# pdf=fitz.open(pdf_path)
# text=""
# for page in pdf:
#     text = page.get_text()
#     print("="*50)
#     print(text)
pdf_folder="C:/Users/komma/Desktop/QLORA/DataSet"
Output_folder="C:/Users/komma/Desktop/QLORA/Extracted_text"
os.makedirs(Output_folder, exist_ok=True)
for filename in os.listdir(pdf_folder):
   if filename.endswith(".pdf"):
       pdf_path = os.path.join(pdf_folder, filename)
       pdf=fitz.open(pdf_path)
       print(f"Reading PDF: {filename}")
       text = ""
       txt_name=filename.replace(".pdf",".txt")
       txt_path=os.path.join(Output_folder,txt_name)
       with open(txt_path,"w",encoding="utf-8") as f:
        for page in pdf:
            #text=page.get_text("text")#For Production use "text" instead of "blocks" to get the text in a more readable format.
            f.write(f"\n{"-"*40}PAGE {page.number+1} {"-"*40}\n")
            blocks=page.get_text("blocks")
            blocks=sorted(blocks, key=lambda x: (x[1], x[0]))
            for block in blocks:
                block_text=block[4].strip()
                if block_text:
                    f.write(block_text)
                    

