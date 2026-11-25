# Healthcare SLM Fine-Tuning Benchmark — LoRA vs QLoRA

Research project expanding a healthcare-domain QLoRA continued-pretraining workflow into a controlled comparison of LoRA and QLoRA.

## Research contributions
1. **Comparative fine-tuning framework** — matched base model, dataset, split, prompts, and training settings.
2. **Resource-efficiency benchmarking** — memory, elapsed training time, throughput, trainable parameters, adapter size, inference latency.
3. **Healthcare evaluation** — rubric-based review and quality/safety analysis to examine quality-efficiency trade-offs.

## Notebook order
1. `01_dataset_analysis.ipynb`
2. `02_baseline_model_evaluation.ipynb`
3. `03_lora_finetuning.ipynb`
4. `04_qlora_finetuning.ipynb`
5. `05_resource_benchmarking.ipynb`
6. `06_healthcare_evaluation.ipynb`
7. `07_comparative_analysis.ipynb`
8. `08_optimization_experiments.ipynb`

`src_benchmark_helpers.ipynb` contains reusable helpers.

## Hardware note
Qwen2.5-3B is a relatively large checkpoint for a 4 GB GPU. Full experiments may require Colab, Kaggle, or another GPU environment with sufficient memory. Start with a small sequence length and small subset, then scale after validating memory use.

## Reproducibility
Use a shared seed, keep dataset splits fixed, save model/dataset IDs, commit hash, library versions, configs, and actual results. All output cells are intentionally unexecuted in this starter archive; run the notebooks to produce authentic output. No benchmark numbers are prefilled.

## Safety and scope
Research and education only. No clinical deployment, diagnosis, treatment recommendations, or patient data. Use datasets with appropriate licenses and privacy protections. A healthcare QA benchmark does not establish clinical validity.

## Notebook outputs and smoke tests

`00_pipeline_smoke_test.ipynb` contains saved outputs from deterministic, locally constructed synthetic fixtures. These are explicitly labelled as software smoke-test outputs and must not be represented as clinical results or model-training benchmarks. Training notebooks have no fabricated performance outputs; they produce measurements only after execution.
