from transformers import (
    AutoModelForCausalLM, TrainingArguments, Trainer
)
import torch

from dataset import DialogDataset

dataset_name = "daily_dialog"
model_name = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
dialog_dataset = DialogDataset(dataset_name, model_name)

training_args = TrainingArguments(
    save_strategy="no",
    warmup_steps=len(dialog_dataset) // 64,
    logging_steps=200,
    weight_decay=0.0,
    num_train_epochs=3,
    learning_rate=5e-5,
    fp16=True if torch.cuda.is_available() else False,
    lr_scheduler_type="cosine",
    logging_dir='./logs',
    output_dir='./output',
    per_device_train_batch_size=4
)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=dialog_dataset)
print('starting to train the model')
trainer.train()

# Save Model
model.save_pretrained("discord_gpt_model", safe_serialization=False)

