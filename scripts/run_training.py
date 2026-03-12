import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb
from dotenv import load_dotenv

load_dotenv()

def main():
    # 1. Configuration
    base_model_id = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-v0.1")
    output_dir = "./models/fine_tuned_adapter"
    processed_data_path = "./data/processed"
    
    with open("config/lora_config.json", "r") as f:
        lora_params = json.load(f)
    
    # 2. WandB initialization
    wandb.init(project="llm-finetuning-lora", name="mistral-7b-dolly")
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 4. Load Model with 4-bit Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 5. Prepare model for LoRA
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=lora_params["target_modules"],
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type=lora_params["task_type"],
    )
    
    model = get_peft_model(model, lora_config)
    
    # 6. Load Dataset
    train_dataset = load_dataset("json", data_files=os.path.join(processed_data_path, "train.json"), split="train")
    eval_dataset = load_dataset("json", data_files=os.path.join(processed_data_path, "validation.json"), split="train")
    
    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=25,
    )
    
    # 8. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )
    
    # 9. Start Training
    print("Starting training...")
    trainer.train()
    
    # 10. Save Adapter
    print(f"Saving fine-tuned adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Finish WandB
    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    main()
