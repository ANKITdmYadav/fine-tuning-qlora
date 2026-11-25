import os
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = r"C:\Users\komma\Desktop\QLORA\output\final_compare_results.csv"

OUTPUT_DIR= r"C:\Users\komma\Desktop\QLORA\output"
OUTPUT_FILE=os.path.join(OUTPUT_DIR,"similarity_report.csv")

# ==========================================================
# Load Model
# ==========================================================

print("Loading Sentence Transformer...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Model Loaded Successfully")

# ==========================================================
# Read CSV
# ==========================================================

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# Function
# ==========================================================

def calculate_similarity(sentence1, sentence2):

    sentence1 = str(sentence1)
    sentence2 = str(sentence2)

    embedding1 = embedding_model.encode(sentence1)

    embedding2 = embedding_model.encode(sentence2)

    similarity = cosine_similarity(

        [embedding1],

        [embedding2]

    )[0][0]

    return round(float(similarity),4)

# ==========================================================
# Lists
# ==========================================================

base_lora_scores = []

lora_merged_scores = []

status = []

# ==========================================================
# Calculate Similarity
# ==========================================================

for _, row in df.iterrows():

    base = row["Base Model Answer"]

    lora = row["LoRA Model Answer"]

    merged = row["Merged Model Answer"]

    score1 = calculate_similarity(

        base,

        lora

    )

    score2 = calculate_similarity(

        lora,

        merged

    )

    base_lora_scores.append(score1)

    lora_merged_scores.append(score2)

    if score2 >= 0.95:

        status.append("🌟 Excellent Match")

    elif score2 >= 0.90:

        status.append("✅ Very Good Match")

    elif score2 >= 0.80:

        status.append("👍 Good Match")

    else:

        status.append("⚠ Review Output")

# ==========================================================
# Add Columns
# ==========================================================

df["Base_vs_LoRA"] = base_lora_scores

df["LoRA_vs_Merged"] = lora_merged_scores

df["Merge Status"] = status

# ==========================================================
# Save
# ==========================================================

df.to_csv(

    OUTPUT_FILE,

    index=False

)

print()

print("="*70)

print("Semantic Evaluation Completed")

print("="*70)

print()

print(df[[
    "Question",
    "Base_vs_LoRA",
    "LoRA_vs_Merged",
    "Merge Status"
]])

print()

print("Saved to")

print(OUTPUT_FILE)