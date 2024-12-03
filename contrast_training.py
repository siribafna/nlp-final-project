import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import json 

# Step 1: Load Dataset and Tokenize
def prepare_triplet_dataset(example, tokenizer, max_length):
    return {
        "anchor": tokenizer(example["anchor"], truncation=True, padding="max_length", max_length=max_length),
        "positive": tokenizer(example["positive"], truncation=True, padding="max_length", max_length=max_length),
        "negative": tokenizer(example["negative"], truncation=True, padding="max_length", max_length=max_length),
    }

def collate_fn(batch):
    def stack_and_convert(key):
        return torch.stack([torch.tensor(item[key]) for item in batch])

    return {
        "anchor_input_ids": stack_and_convert("anchor")[:, "input_ids"],
        "positive_input_ids": stack_and_convert("positive")[:, "input_ids"],
        "negative_input_ids": stack_and_convert("negative")[:, "input_ids"],
    }

# Load data
with open("triplets.json", "r") as f:
   data = json.load(f)
dataset = Dataset.from_list(data)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

max_length = 64
dataset = dataset.map(lambda x: prepare_triplet_dataset(x, tokenizer, max_length), batched=True)

# Step 2: Define the Model
class TripletModel(nn.Module):
    def __init__(self, model_name, embedding_dim):
        super(TripletModel, self).__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.base_model.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        return self.projection(pooled_output)

model = TripletModel("bert-base-uncased", embedding_dim=128)

# Step 3: Define Loss and Training Loop
class TripletLossTrainer:
    def __init__(self, model, lr):
        self.model = model
        self.criterion = nn.TripletMarginLoss(margin=1.0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train(self, dataloader, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                anchor = self.model(batch["anchor_input_ids"], batch["anchor_attention_mask"])
                positive = self.model(batch["positive_input_ids"], batch["positive_attention_mask"])
                negative = self.model(batch["negative_input_ids"], batch["negative_attention_mask"])
                
                loss = self.criterion(anchor, positive, negative)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# Step 4: Evaluate the Model
def evaluate_embeddings(model, dataset):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for batch in dataset:
            anchor = model(batch["anchor_input_ids"], batch["anchor_attention_mask"])
            embeddings.append(anchor.cpu().numpy())
            labels.extend(batch["label"])
    return np.array(embeddings), np.array(labels)

def visualize_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis", s=10)
    plt.colorbar()
    plt.title("t-SNE Visualization of Triplet Embeddings")
    plt.show()

# Example Execution
if __name__ == "__main__":
    # DataLoader for training
    print("Starting")
    # Initialize Trainer and Train
    trainer = TripletLossTrainer(model, lr=5e-5)
    trainer.train(dataloader, num_epochs=3)

    # Evaluate
    embeddings, labels = evaluate_embeddings(model, dataset["test"])

    # Visualize
    visualize_embeddings(embeddings, labels)
