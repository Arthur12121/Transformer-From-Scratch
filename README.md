# Transformer-From-Scratch
Transformer architecture implemented from scratch using PyTorch for educational and research purpose

# Transformer From Scratch using PyTorch

A complete implementation of the original Transformer architecture built from scratch using PyTorch without relying on `nn.Transformer`.

This project was developed for educational purposes to understand every component of the Transformer architecture in detail.

---

# Features

- Transformer implemented from scratch
- Multi-Head Self-Attention
- Positional Encoding (Sinusoidal)
- Feed Forward Network (FFN)
- Residual Connections
- Layer Normalization
- Transformer Encoder
- Transformer Decoder
- Masked Self-Attention
- Cross Attention
- Token Embedding
- Output Projection Layer
- Custom Dataset
- Custom DataLoader
- Complete Training Pipeline
- Model Saving

---

# Project Structure

```
MyProject/
│
├── model.py          # Transformer implementation
├── tokenization.py   # Tokenization and vocabulary
├── interface.py      # Inference interface
├── train.py          # Training script
├── book.txt          # Training data
├── book1.txt
└── README.md
```

---

# Transformer Architecture

```
Input Tokens
      │
Embedding
      │
Positional Encoding
      │
Transformer Encoder × N
      │
────────────────────────────
      │
Target Tokens
      │
Embedding
      │
Positional Encoding
      │
Masked Multi-Head Attention
      │
Cross Attention
      │
Feed Forward Network
      │
Output Projection
      │
Softmax
```

---

# Technologies

- Python 3
- PyTorch
- Object-Oriented Programming (OOP)

---

# Implemented Components

- Position Encoder
- Multi-Head Attention
- Feed Forward Network
- Encoder Layer
- Decoder Layer
- Residual Connections
- Layer Normalization
- Masking
- Embedding
- Output Projection
- Training Loop

---

# Training

The model is trained using:

- CrossEntropyLoss
- Adam Optimizer
- Mini-Batch Gradient Descent

Example:

```python
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

---

# Future Improvements

- GPU Training
- Mixed Precision Training
- Beam Search Decoding
- Learning Rate Scheduler
- Early Stopping
- BLEU Score Evaluation
- TensorBoard Support
- Better Tokenization
- Transformer Encoder for Classification
- Network Intrusion Detection using Transformer

---

# Educational Goal

The main objective of this repository is to understand the Transformer architecture by implementing each component manually instead of using the built-in PyTorch Transformer modules.

---

# Future Research

This implementation will be extended to:

- Transformer Encoder for Classification
- AI for Network Security
- Intrusion Detection Systems (IDS)
- Time Series Forecasting

---

# NOTE

Important note: This version is untrained. You can train it using a text source—such as a book—which you can change if desired. First
you must launch the model and then the interface to generate the reference lists


.<img width="710" height="985" alt="transformers" src="https://github.com/user-attachments/assets/544f09f8-b781-48d1-8db2-114b7419a7a9" />
