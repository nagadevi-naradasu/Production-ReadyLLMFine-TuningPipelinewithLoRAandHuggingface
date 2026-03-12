# 🚀 Production-Ready LLM Fine-Tuning Pipeline

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Track-gold)](https://wandb.ai/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-blue)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-green)](https://fastapi.tiangolo.com/)

An end-to-end MLOps system for fine-tuning Large Language Models (LLMs) using **QLoRA** (4-bit quantization) and deploying them as high-performance REST APIs.

---

## 🛠 Architecture & Features

- **Fine-Tuning**: Leverages `Peft` and `TRL` (SFTTrainer) for parameter-efficient adaptation.
- **Quantization**: Uses `BitsAndBytes` for 4-bit quantization, enabling fine-tuning on consumer-grade GPUs.
- **Experiment Tracking**: Managed via `Weights & Biases` (WandB) for logging loss, metrics, and comparisons.
- **Containerization**: Fully orchestrated with `Docker Compose` for reproducible training and serving.
- **API Serving**: High-speed inference using `FastAPI` with automated health checks.

---

## 📂 Project Structure

```text
├── config/
│   └── lora_config.json      # LoRA adaptation parameters
├── data/
│   └── processed/            # Cleaned and formatted training/val data
├── models/
│   └── fine_tuned_adapter/   # Saved LoRA adapter weights
├── results/
│   ├── evaluation_metrics.json
│   └── comparison.md         # Qualitative model comparison
├── scripts/
│   ├── prepare_data.py       # Data download and formatting
│   ├── run_training.py       # Core QLoRA training loop
│   └── evaluate_model.py     # Metrics calculation and inference test
├── main.py                   # FastAPI service
├── Dockerfile                # Unified environment for training & API
└── docker-compose.yml        # Service orchestration
```

---

## 🚦 Getting Started

### 1. Environment Configuration

Copy the example environment file and provide your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
| :--- | :--- |
| `WANDB_API_KEY` | Your [Weights & Biases](https://wandb.ai/settings) API key. |
| `HF_TOKEN` | Your [Hugging Face](https://huggingface.co/settings/tokens) Read token. |
| `BASE_MODEL_ID` | The model identifier (e.g., `mistralai/Mistral-7B-v0.1`). |

### 2. Implementation Pipeline

#### Phase A: Data Preparation
Pre-process the `databricks/databricks-dolly-15k` dataset into model-ready instruction formats:
```bash
docker-compose run training python scripts/prepare_data.py
```

#### Phase B: QLoRA Fine-Tuning
Execute the parameter-efficient training loop:
```bash
docker-compose run training python scripts/run_training.py
```

#### Phase C: Evaluation & Metrics
Generate perplexity, ROUGE, and BLEU scores, along with qualitative comparisons:
```bash
docker-compose run training python scripts/evaluate_model.py
```

---

## 🌐 API Deployment

Deploy the inference server:

```bash
docker-compose up api
```

### Endpoints

- **Health Check**: `GET http://localhost:8000/health`
- **Text Generation**: `POST http://localhost:8000/generate`

**Sample Request:**
```json
{
  "prompt": "Explain the concept of Quantum Entanglement to a 5-year old.",
  "max_new_tokens": 128
}
```

---

## 📊 Verification Checklist

- [x] **Containerization**: Single Dockerfile for all services.
- [x] **Quantization**: 4-bit QLoRA implementation for memory efficiency.
- [x] **Experiment Tracking**: Integrated WandB logging.
- [x] **Evaluation**: Quantitative (JSON) and Qualitative (Markdown) outputs.
- [x] **Scalability**: FastAPI-based microservice architecture.
