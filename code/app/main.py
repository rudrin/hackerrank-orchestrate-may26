# app/main.py
import os
import json
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Find .env file in project root
print("Loading environment variables...")
# env_path = Path(__file__).parent.parent.parent / ".env"
env_path = "D:\\hackerrank\\hackerrank-orchestrate-may26\\.env.example"
print(f"Looking for .env file at: {env_path}")
load_dotenv(env_path)

import pandas as pd
from pipeline import process_ticket
from ingestion import DataIngestion
from system import LLMSystem


# Progress tracking file
PROGRESS_FILE = "support_tickets/.progress.json"


def load_progress():
    """Load the last processed index from progress file."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return -1, False
                data = json.load(f)
                return data.get("last_processed_index", -1), data.get("header_written", False)
        except (json.JSONDecodeError, ValueError):
            # If file is corrupted/empty, reset to start
            return -1, False
    return -1, False


def save_progress(index, header_written=False):
    """Save the last processed index to progress file."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_processed_index": index, "header_written": header_written}, f)


def append_result(output_path, result, header=True):
    """Append a single result to the CSV file."""
    df = pd.DataFrame([result])
    
    # Define column order
    columns = ["Issue", "Subject", "Company", "Response", "Product Area", "Status", "Request Type"]
    df = df[columns]  # Ensure correct column order
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not header:
        # Append without header if file exists and header=False
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        # Create new file with header
        df.to_csv(output_path, mode="w", header=header, index=False)


def main():
    # Initialize FAISS vector database
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    index_path = os.path.join(data_dir, "index.faiss")
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    # Try to load existing index, or build new one
    ingestion = DataIngestion(data_dir=data_dir)
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        print("Loading existing FAISS index...")
        ingestion.load_index(index_path, metadata_path)
    else:
        print("Building FAISS index from data files...")
        stats = ingestion.build_index()
        print(f"Indexed {stats.get('total_chunks', 0)} chunks")
        ingestion.save_index(index_path, metadata_path)
    
    # Initialize LLM system
    print("Initializing LLM system...")
    system = LLMSystem(provider="groq", model="openai/gpt-oss-120b")
    
    # Attach ingestion to system for retrieval
    system.ingestion = ingestion
    
    # Load support tickets
    tickets_path = os.path.join(os.path.dirname(__file__), "..", "..", "support_tickets", "support_tickets.csv")
    df = pd.read_csv(tickets_path)
    print(f"Processing {len(df)} support tickets...")
    
    # Load progress to resume from last processed index
    start_index, header_written = load_progress()
    start_index = start_index + 1  # Resume from next unprocessed ticket
    print(f"Resuming from ticket {start_index + 1}/{len(df)}...")
    
    # Output path
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "support_tickets", "output.csv")
    
    # If output file exists and has content, header is already written
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        header_written = True
    
    # Process each ticket starting from resume point
    for idx in range(start_index, len(df)):
        row = df.iloc[idx]
        print("\t\t"+"#"*25+f" Processing ticket {idx + 1}/{len(df)} "+"#"*25+"\t\t")
        result = process_ticket(row, system)
        
        # Write result immediately to CSV
        append_result(output_path, result, header=not header_written)
        header_written = True  # Header written on first iteration
        print(f"Result written to {output_path}")
        
        # Save progress after each ticket
        save_progress(idx, header_written)
    
    print(f"All results saved to {output_path}")


if __name__ == "__main__":
    main()