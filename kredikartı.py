import pandas as pd
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

# Yeni CSV dosyasını yükle
data = pd.read_csv('creditcard5.csv')

# Hedef değişkeni ayarla
data['Class'] = data['Class'].astype(int)  # Hedef değişkenin ayarlanması

# Hedef değişkenin benzersiz değerlerini kontrol et
print("Benzersiz değerler:", data['Class'].unique())

# Veri setini eğitim ve değerlendirme olarak ayır
train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

# Hugging Face Dataset formatına dönüştür
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# Model ve tokenizer'ı yükle
model_name = "distilbert-base-uncased"  # Kullanılacak model
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Veri setini tokenize et
def tokenize_function(examples):
    # V1 ve V2'yi birleştirerek bir dize oluşturun
    texts = [f"{v1} {v2}" for v1, v2 in zip(examples['V1'], examples['V2'])]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # Girişlerin doğru formatta olmasını sağlamak için
    )

# Tokenizasyon işlemini güncelleyin
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Girişlerin yapısını kontrol et
first_example = tokenized_train_dataset[0]
print("Input IDs:", first_example['input_ids'])
print("Attention Mask:", first_example['attention_mask'])

# Eğitim argümanlarını ayarla
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,  # Batch boyutunu artırmayı deneyin
    num_train_epochs=3,
)

# Kayıp ve doğruluk hesaplama fonksiyonu
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Trainer'ı oluştur ve modeli eğit
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # Değerlendirme veri setini ekleyin
    compute_metrics=compute_metrics,  # Kayıp ve doğruluk hesaplama fonksiyonu
)

# Modeli eğit
trainer.train()