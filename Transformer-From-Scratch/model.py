import torch
import math
from torch.utils.data import DataLoader , Dataset
from tokenzation import encoder_input , decoder_input , target , vocab_size

####main_componts####

class TextDataset(Dataset):

    def __init__(self, encoder_input, decoder_input, target):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.target = target

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return (
            self.encoder_input[idx],
            self.decoder_input[idx],
            self.target[idx]
        )
    

class outputproject(torch.nn.Module):

    def __init__(self, d_model , vcobe):
        super().__init__()

        self.linear = torch.nn.Linear(d_model , vcobe)

    def forward(self , x):

        x = self.linear(x)

        return x

class PositionEncoder(torch.nn.Module):

    def __init__(self, d_model , max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len , d_model)

        Position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )

            *
            (-math.log(10000.0) / d_model)
        )

        pe[: , 0::2] = torch.sin(Position * div_term)
        pe[: , 1::2] = torch.cos(Position * div_term)

        self.register_buffer("pe" , pe.unsqueeze(0))
    
    def forward(self , x):

        seq_len = x.size(1)
        return x + self.pe[: , :seq_len]

class FFN(torch.nn.Module):

    def __init__(self, d_model , hidden_layers):
        super().__init__()

        self.LinearLayers1 = torch.nn.Linear(d_model , hidden_layers)
        self.Relu = torch.nn.ReLU()
        self.LinearLayers2 = torch.nn.Linear(hidden_layers , d_model)

    def forward(self , x):

        x = self.LinearLayers1(x)
        x = self.Relu(x)
        x = self.LinearLayers2(x)

        return x

class multiheadattention(torch.nn.Module):
    
    def __init__(self, d_model ,seq_len, head_num ):
        super().__init__()

        assert d_model % head_num == 0

        self.d_model = d_model
        self.seq_len = seq_len

        self.num_head = head_num

        self.wq = torch.nn.Linear(d_model , d_model)
        self.wk = torch.nn.Linear(d_model , d_model)
        self.wv = torch.nn.Linear(d_model , d_model)
        self.wo = torch.nn.Linear(d_model , d_model)

        self.dk = self.d_model // self.num_head

    def forward(self , x , mask = None):

        batch_input = x.size(0)
        seq_len_input = x.size(1)

        Query = self.wq(x)
        key   = self.wk(x)
        value = self.wv(x)

        Query = Query.view(
            batch_input,
            seq_len_input,
            self.num_head,
            self.dk
        ).transpose(1 , 2)

        key = key.view(
            batch_input,
            seq_len_input,
            self.num_head,
            self.dk
        ).transpose(1 , 2)

        value = value.view(
            batch_input,
            seq_len_input,
            self.num_head,
            self.dk
        ).transpose(1 , 2)

        score = (Query @ key.transpose(-2 , -1) / math.sqrt(self.dk))

        if mask is not None:
            score = score.masked_fill(mask , -1e9)

        wieght = torch.softmax(score , dim=-1)

        output = wieght @ value

        output = output.transpose(1 , 2)

        output = output.contiguous().view(
            batch_input,
            seq_len_input,
            self.d_model
        )

        output = self.wo(output)
        return output
    
class TrasnEncoderLayer(torch.nn.Module):

    def __init__(self, d_model , seq_len , head_num , hidden_layer):
        super().__init__()

        self.multihead = multiheadattention(d_model = d_model,
                                            seq_len = seq_len,
                                            head_num = head_num)
        
        self.ffn = FFN(d_model=d_model , hidden_layers=hidden_layer)
        
        self.addnorm1 = torch.nn.LayerNorm(d_model)
        self.addnorm2 = torch.nn.LayerNorm(d_model)

    def forward(self , x):

        atten_out = self.multihead(x)
        
        addnorm1 = self.addnorm1(atten_out + x)

        fnn = self.ffn(addnorm1)

        output = self.addnorm2(fnn + addnorm1)

        return output

class TrasnDencoderLayer(torch.nn.Module):
    
    def __init__(self,  d_model , seq_len , head_num , hidden_layer):
        super().__init__()

        self.multiheadmask = multiheadattention(d_model = d_model,
                                            seq_len = seq_len,
                                            head_num = head_num,)
        
        self.multihead = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=head_num,
            batch_first=True
        )
        
        self.ffn = FFN(d_model=d_model , hidden_layers=hidden_layer)
        
        self.addnorm1 = torch.nn.LayerNorm(d_model)
        self.addnorm2 = torch.nn.LayerNorm(d_model)
        self.addnorm3 = torch.nn.LayerNorm(d_model)

    def forward(self , x , trasnmodel_encoder):

        seq_len = x.size(1)

        mask = torch.triu(
        torch.ones(seq_len, seq_len),
        diagonal=1
         ).bool()

        mask = mask.unsqueeze(0).unsqueeze(0)

        multihead_mask_out = self.multiheadmask(x , mask)

        addnorm1_out = self.addnorm1(multihead_mask_out + x)

        multihead , _= self.multihead(addnorm1_out , trasnmodel_encoder , trasnmodel_encoder)

        addnorm2_out = self.addnorm2(multihead + addnorm1_out)

        fnn_out = self.ffn(addnorm2_out)

        addnorm3_out = self.addnorm3(fnn_out + addnorm2_out)

        return addnorm3_out
    
class transformer(torch.nn.Module):

    def __init__(self, vcobe , d_model=64 ,seq_len=128 ,head_num=4 , hidden_layer=128):
        super().__init__()

        hidden_layer = d_model * 4

        self.output_projection = torch.nn.Linear(d_model , vcobe)

        self.embedding = torch.nn.Embedding(
            num_embeddings=vcobe,
            embedding_dim=d_model
        )

        self.position = PositionEncoder(
            d_model=d_model,
            max_len=5000
        )

        self.encoder_layer = torch.nn.ModuleList(
            [TrasnEncoderLayer(d_model=d_model , seq_len=seq_len ,head_num=head_num ,hidden_layer=hidden_layer)
             for _ in range(2)
             ]
        )

        self.dencoder_layer = torch.nn.ModuleList(
            [TrasnDencoderLayer(d_model=d_model , seq_len=seq_len ,head_num=head_num ,hidden_layer=hidden_layer)
             for _ in range(2)
             ]
        )
    
    def forward(self , src , target):

        x = src

        x = self.embedding(x)

        x = self.position(x)

        for layer in self.encoder_layer:

            x = layer(x)

        encoder_layer_out = x
        
        decoder_emb = self.embedding(target)

        decoder_pos = self.position(decoder_emb)

        decoder_out = decoder_pos

        for layer in self.dencoder_layer:
         
         decoder_out = layer(decoder_out, encoder_layer_out)

        output = self.output_projection(decoder_out)

        return output
    

def main(target):
 model = transformer(vcobe=vocab_size)


 certiaiotn = torch.nn.CrossEntropyLoss()

 optimzer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
 )

 dataset = TextDataset(encoder_input , decoder_input, target)

 loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
 )

 for epoch in range(60):

    total_loss = 0

    for encoder, decoder, target in loader:

        prediction = model(encoder, decoder)

        prediction = prediction.reshape(-1, vocab_size)
        target = target.reshape(-1)

        loss = certiaiotn(prediction, target)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")


 torch.save(model.state_dict() , "Model01.pth")

if __name__ == "__main__":
    main(target)
 