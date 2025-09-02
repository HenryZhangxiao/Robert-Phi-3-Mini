# inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
ADAPTER_PATH = "phi3-lora-adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load base model in 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

def chat(prompt: str, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_input = "How's it going?"
    print("User:", user_input)
    response = chat(f"User: {user_input}\nFriend:")
    print("Friend:", response)
