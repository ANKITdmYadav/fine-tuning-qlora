# 🏥 Healthcare Clinical Assistant using QLoRA Continued Pretraining

## 📌 Project Overview

This project demonstrates an end-to-end Large Language Model (LLM) fine-tuning pipeline by adapting **Qwen2.5-3B-Instruct** to the healthcare domain using **Continued Pretraining (CPT)** with **QLoRA**.

The model was trained on a custom medical corpus created from healthcare PDF documents covering diseases such as:

- Diabetes
- Tuberculosis
- Cancer
- Fatty Liver Disease
- And other clinical topics

The project includes the complete workflow from data preprocessing to model evaluation and adapter merging.

---

# 🎯 Objectives

- Build a domain-specific healthcare language model.
- Learn Continued Pretraining (CPT).
- Implement QLoRA for memory-efficient fine-tuning.
- Compare Base, LoRA, and Merged models.
- Evaluate semantic similarity after adapter merging.

---

# 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- QLoRA
- BitsAndBytes
- Sentence Transformers
- Scikit-Learn
- Pandas
- PyMuPDF

---

# 📂 Dataset

Medical PDF documents containing healthcare information.

Pipeline:

Medical PDFs

↓

Text Extraction

↓

Text Cleaning

↓

Corpus Creation

↓

Tokenization

↓

Chunking

↓

HF Dataset

---

# 🏗 Project Workflow

```text
Medical PDFs
      │
      ▼
PDF Extraction
      │
      ▼
Text Cleaning
      │
      ▼
Medical Corpus
      │
      ▼
Tokenizer
      │
      ▼
Chunk Dataset
      │
      ▼
HF Dataset
      │
      ▼
Qwen2.5-3B
      │
      ▼
QLoRA Continued Pretraining
      │
      ▼
Medical LoRA Adapter
      │
      ▼
Merge Adapter
      │
      ▼
Evaluation
```

---

# 🏛 Architecture

```text
                   Medical PDFs
                        │
                        ▼
               PDF Text Extraction
                        │
                        ▼
                 Data Cleaning
                        │
                        ▼
              Medical Corpus
                        │
                        ▼
          Tokenization + Chunking
                        │
                        ▼
             HuggingFace Dataset
                        │
                        ▼
        Qwen2.5-3B (4-bit Quantized)
                        │
                        ▼
            QLoRA Continued Pretraining
                        │
                        ▼
                LoRA Adapter Weights
             ┌──────────┴──────────┐
             ▼                     ▼
      LoRA Inference         Merge Adapter
             │                     │
             └──────────┬──────────┘
                        ▼
            Semantic Similarity Evaluation
```

---

# ⚙ Training Configuration

| Parameter | Value |
|-----------|--------|
| Base Model | Qwen2.5-3B-Instruct |
| Fine-Tuning | Continued Pretraining (CPT) |
| Quantization | 4-bit NF4 |
| Adapter | LoRA |
| Optimizer | paged_adamw_8bit |
| Epochs | 3 |
| Learning Rate | 2e-4 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |

---

# 📊 Evaluation

The project evaluates three models:

- Base Qwen
- Qwen + LoRA
- Merged Qwen

Semantic similarity is computed using:

- Sentence Transformers
- Cosine Similarity

---

# 📈 Results

Example metrics:

| Metric | Score |
|---------|-------|
| Average Base → LoRA Similarity | ~0.87 |
| Average LoRA → Merged Similarity | ~0.91 |

The merged model preserves the behavior of the LoRA model while producing a standalone deployable model.

---

# 📁 Project Structure

```text
Healthcare-Clinical-Assistant-QLoRA/

├── preprocessing/
├── evaluation/
├── models/
├── output/
├── train.py
├── merge_adapter.py
├── cpt_dataset.py
├── requirements.txt
└── README.md
```

---

# 🚀 How to Run

```bash
git clone <repo>

pip install -r requirements.txt

python train.py

python merge_adapter.py

python evaluation/evaluate_base.py

python evaluation/evaluate_lora.py

python evaluation/evaluate_merged.py
```

---

# 🔮 Future Improvements

- Supervised Fine-Tuning (SFT)
- Retrieval-Augmented Generation (RAG)
- Medical Question Answering
- Clinical Report Generation
- Larger Medical Corpora

---

# 👨‍💻 Author

Sandeep Sai

B.Tech Computer Science

Generative AI | LLM Fine-Tuning | Machine Learning