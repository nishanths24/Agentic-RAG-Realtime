# Agentic RAG Realtime

[![Status](https://img.shields.io/badge/Status-Building%20(Not%20Completed)-orange?style=for-the-badge&logo=git)](https://github.com/nishanths24/Agentic-RAG-Realtime)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Build & Test](https://github.com/nishanths24/Agentic-RAG-Realtime/actions/workflows/build.yml/badge.svg)](https://github.com/nishanths24/Agentic-RAG-Realtime/actions)

> 🚧 **Notice: Project Under Active Development**  
> This project is currently **building (work in progress / not completed)**. Core document ingestion and chunking components are in place, with agentic tool calling and real-time retrieval features actively being implemented.

---

## Project Overview

**Agentic RAG Realtime** is an intelligent Retrieval-Augmented Generation system designed to combine document retrieval with real-time web & API tools for dynamic, agentic reasoning.

### Key Highlights
- **Document Ingestion Engine**: Clean and preprocess text data from various sources with character-level and token-level chunking with configurable overlap.
- **Agentic Decision Loop**: Planned integration with LLM-powered autonomous agents capable of deciding when to retrieve local context vs. calling real-time external tools.
- **Real-Time Tool Calling**: Integrating live search and external APIs to supplement static vector store knowledge.

---

## Project Roadmap & Current Progress

- [x] **Project Scaffolding & Architecture**
- [x] **Document Ingestion & Text Normalization**
- [x] **Sliding Window Chunking with Overlap**
- [x] **Unit Testing & CI Pipeline (`unittest` + GitHub Actions)**
- [ ] 🔨 **Embedding Generation & Vector Storage** *(In Progress / Building)*
- [ ] 🔨 **Agentic Reasoning & Tool Orchestration** *(In Progress / Building)*
- [ ] ⏳ **Real-Time Tool Integrations (Search, APIs)** *(Upcoming)*
- [ ] ⏳ **Evaluation & Benchmark Suite** *(Upcoming)*

---

## Project Structure

```text
Agentic RAG Realtime/
├── .github/
│   └── workflows/
│       └── build.yml          # Automated CI build & test workflow
├── data/
│   └── documents/
│       └── travel_policy.txt  # Sample policy document for ingestion
├── src/
│   └── agentic_rag/
│       ├── __init__.py
│       └── ingestion.py       # Cleaning, file reading, and chunking logic
├── tests/
│   └── test_ingestion.py      # Unit tests for text cleaning and chunking
├── .gitignore                 # Environment and build exclusion rules
├── main.py                    # Project entry point
├── run_ingestion.py           # Ingestion pipeline demonstration script
└── README.md                  # Project documentation & status
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/nishanths24/Agentic-RAG-Realtime.git
cd Agentic-RAG-Realtime
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 3. Run the Ingestion Pipeline
```bash
python run_ingestion.py
```

### 4. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## Tech Stack
- **Language**: Python 3.11+
- **Testing**: `unittest`
- **CI/CD**: GitHub Actions
- **Frameworks** *(Upcoming)*: LangChain / LlamaIndex / Custom Agentic Workflow

---

## Author
- **Nishanth S** - [nishanths24](https://github.com/nishanths24)
