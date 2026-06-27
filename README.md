# Indic RAG Evaluator 🇮🇳

A specialized evaluation framework designed to benchmark Retrieval-Augmented Generation (RAG) pipelines on **Hindi and Code-Mixed (Hinglish) datasets**. 

This repository was built as a **Proof of Work (PoW) for the Applied AI Engineer role at Sarvam AI**, demonstrating the ability to build and evaluate agentic workflows tailored for the Indian AI ecosystem.

## Why this exists?
Standard evaluation metrics (like RAGAS) are optimized for English. When evaluating systems on Indian languages, translation artifacts and code-mixed tokenization often break standard retrieval metrics (Context Precision, Answer Relevancy). 
This framework introduces a custom LLM-as-a-judge pipeline optimized for Indic languages.

## Features
- 📊 **Code-Mixed Context Evaluation**: Evaluates retrieval precision when queries are in Hinglish but documents are in English/Hindi.
- 🎯 **Answer Faithfulness**: Custom prompts for evaluating hallucinations in native Hindi outputs.
- ⚡ **FastAPI Integration**: Ready to be deployed as an evaluation microservice.

## Tech Stack
- **LangChain** (Custom Prompts & Chains)
- **FastAPI** (Evaluation API)
- **Pandas & NumPy** (Metric Aggregation)
- **OpenAI/Sarvam APIs** (LLM as a judge)

## Project Structure
```text
├── evaluator.py       # Core evaluation logic for Indic languages
├── api.py             # FastAPI endpoints for real-time evaluation
├── requirements.txt   # Dependencies
└── README.md
```

## Quick Start
```bash
pip install -r requirements.txt
python evaluator.py
```

*Built by [Samarth Sugandhi](https://github.com/Sxmxxrth) — AI/ML Engineer.*
