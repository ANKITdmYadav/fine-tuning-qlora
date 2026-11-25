import os
from pathlib import Path
input_file=Path("C:\\Users\\komma\\Desktop\\QLORA\\Cleaned_text")
Output_file=Path("C:\\Users\\komma\\Desktop\\QLORA\\CPT")
Path(Output_file).mkdir(parents=True, exist_ok=True)
with open(Output_file / "train.txt","w",encoding="utf-8") as ff:
    for filename in os.listdir(input_file):
        if filename.endswith(".txt"):
            file_path=input_file / filename
            with open(file_path,"r",encoding="utf-8") as f:
                text=f.read()
            #text=text.lower()
            ff.write("<|START OF TEXT>|")
            ff.write(text)
            ff.write("\n")
            ff.write("<|END OF TEXT>|")
            ff.write("\n\n")