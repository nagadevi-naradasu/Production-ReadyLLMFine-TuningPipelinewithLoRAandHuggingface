import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LLM Fine-Tuning API")

# Global variables for model and tokenizer
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    generated_text: str

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    base_model_id = os.getenv("BASE_MODEL_ID", "mistralai/Mistral-7B-v0.1")
    adapter_path = "./models/fine_tuned_adapter"
    
    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Check if adapter exists
    if not os.path.exists(adapter_path):
        print(f"Warning: Adapter path {adapter_path} not found. Loading base model only.")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print(f"Loading base model and adapter: {adapter_path}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        # Note: merge_and_unload() doesn't work with 4-bit quantization directly in some versions
        # but for production inference, it's often better to merge if possible.
        # For this implementation, we'll keep it as a PeftModel for compatibility with 4-bit.
        # model = model.merge_and_unload() 
    
    print("Model loaded successfully.")

@app.get("/health")
async def health_check():
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the generated text if necessary
        # result = generated_text[len(request.prompt):].strip()
        
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
