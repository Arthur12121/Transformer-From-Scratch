import torch
import pickle

with open("book1.txt", "r", encoding="utf-8") as f:
    text = f.read()


text = text.lower()
text = text.replace("\n", " ")

tokens = text.split()


special_tokens = [
    "<PAD>",
    "<UNK>",
    "<SOS>",
    "<EOS>"
]

vocab = special_tokens + sorted(set(tokens))

word2idx = {
    word: idx
    for idx, word in enumerate(vocab)
}

idx2word = {
    idx: word
    for word, idx in word2idx.items()
}

vocab_size = len(vocab)



encoded = torch.tensor(
    [
        word2idx.get(word, word2idx["<UNK>"])
        for word in tokens
    ],
    dtype=torch.long
)


window = 5

encoder_input = []
decoder_input = []
target = []

SOS = word2idx["<SOS>"]
EOS = word2idx["<EOS>"]

for i in range(len(encoded) - window):

    sentence = encoded[i:i + window]

    # Encoder
    encoder_input.append(sentence)

    # Decoder يبدأ بـ <SOS>
    decoder = torch.cat((
        torch.tensor([SOS]),
        sentence[:-1]
    ))

    decoder_input.append(decoder)

    
    target_sentence = torch.cat((
        sentence[1:],
        torch.tensor([EOS])
    ))

    target.append(target_sentence)

encoder_input = torch.stack(encoder_input)
decoder_input = torch.stack(decoder_input)
target = torch.stack(target)

with open("word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)

with open("idx2word.pkl", "wb") as f:
    pickle.dump(idx2word, f)

print("Vocabulary Size :", vocab_size)
print("Encoder :", encoder_input.shape)
print("Decoder :", decoder_input.shape)
print("Target  :", target.shape)
