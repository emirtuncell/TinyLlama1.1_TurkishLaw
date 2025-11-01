# ===============================
# GEMMA 2B Fine-Tune (RAG tabanlı hukuk modeli)
# ===============================

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
import pandas as pd
import torch
torch.cuda.empty_cache()
from transformers import BitsAndBytesConfig

# -------------------------------
# 1️⃣ Veri setini yükle
# -------------------------------
df = pd.read_csv("turkish_law_dataset_filtered.csv")
# ⚠️ Tüm veriyle eğitmek için bu satır kaldırıldı
# df = df.head(100)

print(f"Toplam veri sayısı: {len(df)}")
print(df.head())

# Veri formatlama
def format_example(example):
    return {
        "text": f"Kullanıcı: {example['soru']}\n\nYapay Zeka (Hukuk Uzmanı): {example['context']}"
    }

dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_example)

# -------------------------------
# 2️⃣ Model ve Tokenizer
# -------------------------------
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # bazı modellerde gerekli

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,        
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# -------------------------------
# 3️⃣ LoRA (PEFT) konfigürasyonu
# -------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -------------------------------
# 4️⃣ Eğitim ayarları
# -------------------------------
training_args = TrainingArguments(
    output_dir="./finetuned-law2",
    num_train_epochs=3,  # 🔁 3 epoch: tüm veriyi 3 kez döner
    per_device_train_batch_size=2,  # VRAM durumuna göre 1 veya 2
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=1,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none"
)

# -------------------------------
# 5️⃣ Trainer başlat
# -------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=lambda examples: examples["text"],
)

# -------------------------------
# 6️⃣ Eğitimi başlat
# -------------------------------
trainer.train()

# -------------------------------
# 7️⃣ Fine-tune edilmiş modeli kaydet
# -------------------------------
trainer.save_model("./finetuned-law2")
tokenizer.save_pretrained("./finetuned-law2")

print("✅ Fine-tuning başarıyla tamamlandı ve './finetuned-law2' dizinine kaydedildi.")
