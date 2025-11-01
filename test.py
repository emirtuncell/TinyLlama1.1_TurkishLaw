from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# -------------------------------
# Model ayarları
# -------------------------------
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_model_name = "emirtuncel/finetuned_law"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Base model + tokenizer
# -------------------------------
print("Base model yükleniyor...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# pad_token ekle (yoksa generate çalışmaz)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# -------------------------------
# LoRA adapter yükleme
# -------------------------------
print("Fine-tuned adapter yükleniyor...")
model = PeftModel.from_pretrained(base_model, adapter_model_name)
model.to(device)
model.eval()

# -------------------------------
# Prompt (chat biçimiyle)
# -------------------------------
prompt = "Bir sözleşmenin geçerli olabilmesi için hangi şartlar gereklidir?"

chat = [
    {"role": "user", "content": prompt}
]
inputs = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)

# -------------------------------
# Yanıt üretme
# -------------------------------
print("\nModel yanıt üretiyor...\n")
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# -------------------------------
# Yanıtı yazdır
# -------------------------------
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Sadece model yanıtını ayırmak istersen:
if prompt in response:
    print("\n--- Yanıt ---\n")
    print(response.split(prompt)[-1].strip())
