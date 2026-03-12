import os
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

def format_instruction(sample):
    """
    Formats a sample into a single instruction-response string.
    """
    instruction = sample.get("instruction", "")
    context = sample.get("context", "")
    response = sample.get("response", "")
    
    if context:
        text = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
    else:
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    
    return {"text": text}

def main():
    print("Downloading dataset...")
    # Load the databricks-dolly-15k dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    print("Formatting data...")
    # Apply the formatting function to each sample
    formatted_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Convert to list of dicts
    data_list = list(formatted_dataset)
    
    print(f"Total samples: {len(data_list)}")
    
    # Split into train and validation sets (90/10)
    train_data, val_data = train_test_split(data_list, test_size=0.1, random_state=42)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create the output directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save to JSON files
    with open("data/processed/train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    
    with open("data/processed/validation.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    
    print("Data preparation complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()
