# 🩺 Dr. Fine RAG: The Mysterious Drug

An immersive, full-stack GenAI medical investigation game in French powered by a local LLM, RAG architecture, and a structured modular backend.

## 🚧 Project Status & Roadmap (MVP)

Note: This project is currently in an active MVP (Minimum Viable Product) stage. The full software architecture, end-to-end pipelines, and decoupled web interface are fully implemented and functional. However, the project is still evolving: the final LLM fine-tuning session is pending, and the RAG pipeline is currently being integrated and refined.

## 🎮 About the Project

Dr. Fine RAG turns medical diagnostics into an interactive puzzle game.

The Core Concept: An LLM acts as a patient suffering from a specific medical condition (and a mysterious drug history).

The Goal: The user (playing as the doctor) must interview the patient, leverage medical knowledge retrieval, and submit an accurate final diagnosis within a strict turn limit.

The Technical Approach: Built with a clean separation of concerns, utilizing cosine similarity for precise text-based RAG matching and a modular pipeline spanning from raw data to web UI inference.

## 📁 Exact Project Structure

```text
dr-fine-rag/
│
├── data/
│   └── raw/
│       └── diseases.json         # Raw medical records dataset for RAG
│
├── notebooks/                    # End-to-end Rational Pipelines & Experimentation
│   ├── 01_rag_exploration.ipynb  # RAG exploration & similarity tests
│   ├── 02_generate_patient_dataset.ipynb # Synthetic data generation
│   ├── 03_finetune_patient_model.ipynb   # Unsloth fine-tuning pipeline
│   ├── 04_test_inference.ipynb           # Model inference validation
│   └── 05_test_victory_condition.ipynb   # Game loop & judge logic tests
│
├── src/                          # Core application source code
│   ├── game/
│   │   └── engine.py             # Core game loop & turn management
│   ├── models/
│   │   ├── inference.py          # LLM inference handler
│   │   ├── patient_generator.py  # Patient persona generation logic
│   │   └── train.py              # Training scripts
│   ├── rag/
│   │   └── retriever.py          # Cosine similarity text retrieval
│   ├── config.py                 # Centralized configuration
│   └── __init__.py
│
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignored files
├── README.md                     # Project documentation
├── app.py                        # Streamlit Frontend UI
├── main.py                       # FastAPI Backend entrypoint
├── requirements.txt              # Project dependencies
└── run.bat                       # Automated Windows launcher script
```

## 🛠️ Tech Stack & Key Choices

Language: Python 3.10+

Backend & API: FastAPI, Uvicorn, Pydantic

Frontend: Streamlit, Requests

LLM Engine: Unsloth (optimized local training & inference framework)

RAG Engine: Vector/text retrieval relying on Cosine Similarity for efficient medical record matching.

## 🚀 Getting Started

### 1. Clone the repository

git clone https://github.com/your-username/dr-fine-rag.git
cd dr-fine-rag


## 2. Configure Environment

Copy the example environment configuration:

```text
pip install -r requirements.txt
```

### 3. Run the Application

You can launch the entire ecosystem (FastAPI Backend + Streamlit Frontend) using the automated script:

On Windows: Double-click or run run.bat in your terminal.

Manual Launch:

Backend: ```text uvicorn main:app --port 8000  ```

Frontend: ```text streamlit run app.py ```

## 🌐 Endpoints & Access

Web Interface (Streamlit): http://localhost:8501

Interactive API Docs (FastAPI/Swagger): http://127.0.0.1:8000/docs

## 💡 Engineering Highlights

Decoupled Architecture: Clean isolation between the Streamlit UI, FastAPI endpoints, and core game/model logic.

Structured R&D Pipeline: Clear separation between exploration notebooks (notebooks/) and production code (src/).

Cosine Similarity Focus: Optimized semantic text matching tailored for medical knowledge lookup.
