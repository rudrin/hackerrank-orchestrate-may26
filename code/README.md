# HackerRank Orchestrate Agent

## Overview

This project builds an agent to process support tickets from `support_tickets/support_tickets.csv` and generate structured responses using an LLM plus FAISS-powered knowledge retrieval.

The pipeline performs:

- FAISS index creation from markdown knowledge sources in `data/`
- Query retrieval and reranking for relevant content
- Risk and confidence estimation
- Request classification into a fixed taxonomy
- Response generation with justification for each decision
- Incremental CSV output to `support_tickets/output.csv`

## Setup

1. Create and activate a Python environment.
2. Install dependencies from the `code/` directory:
   ```bash
   cd code
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` in the repository root.
4. Set your Groq API key in `.env`:
   ```ini
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Running the App

From the repo root, run:

```bash
python code/app/main.py
```

This will:

- build or load the FAISS index from `code/index.faiss`
- process tickets in `support_tickets/support_tickets.csv`
- append results to `support_tickets/output.csv`
- track progress in `support_tickets/.progress.json`

## Output

The output CSV contains:

- `Issue`
- `Subject`
- `Company`
- `Response`
- `Product Area`
- `Status`
- `Request Type`
- `Justification`
