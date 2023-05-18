import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, param):
        super(Seq2Seq, self).__init__()
        input_dim = param["input_dim"]
        output_dim = param["output_dim"]
        self.hidden_dim = param["hidden_dim"]
        self.num_layers = param["num_layers"]

        self.embedding = nn.Embedding(13,13)
        self.encoder = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.decoder = nn.LSTM(self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        _, (hidden, cell) = self.encoder(x, (h0, c0))
        output, _ = self.decoder(hidden, (hidden, cell))

        output = self.fc(output)

        return output