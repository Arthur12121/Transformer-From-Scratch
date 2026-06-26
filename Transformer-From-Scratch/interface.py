import torch
import pickle

from model import transformer

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)


model = transformer(vcobe=vocab_size)

model.load_state_dict(torch.load("Model01.pth"))

model.eval()


PAD = word2idx["<PAD>"]
UNK = word2idx["<UNK>"]
SOS = word2idx["<SOS>"]
EOS = word2idx["<EOS>"]


while True:
 sentence = input("put here text: ").lower().split()

 tokens = [
    word2idx.get(word, UNK)
    for word in sentence
 ]


 WINDOW = 5

 while len(tokens) < WINDOW:
    tokens.append(PAD)

 tokens = tokens[:WINDOW]

 encoder = torch.tensor(
    [tokens],
    dtype=torch.long
 )



 decoder = torch.tensor(
    [[SOS]],
    dtype=torch.long
 )

 with torch.no_grad():

    for _ in range(WINDOW):

        output = model(encoder, decoder)

        next_token = output[:, -1].argmax(dim=-1)

        decoder = torch.cat(
            (
                decoder,
                next_token.unsqueeze(1)
            ),
            dim=1
        )

        if next_token.item() == EOS:
            break



 result = []

 for idx in decoder[0]:

    word = idx2word[idx.item()]

    if word in ["<SOS>", "<PAD>"]:
        continue

    if word == "<EOS>":
        break

    result.append(word)


 print(" ".join(result))