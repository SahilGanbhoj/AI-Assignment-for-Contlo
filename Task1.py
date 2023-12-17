import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

class GPT2LayerNorm(nn.Module):
    def __init__(self, hidden_size, epsilon=1e-5):
        super(GPT2LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class GPT2Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(GPT2Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = nn.Dropout(p=dropout_rate)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(self.head_dim)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_shape = context.size()[:-2] + (self.num_heads * self.head_dim,)
        context = context.view(*new_shape)

        output = self.out(context)
        return output

class GPT2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout_rate=0.1):
        super(GPT2MLP, self).__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size)
        self.c_activation = nn.GELU()
        self.c_dropout = nn.Dropout(dropout_rate)
        self.c_output = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_activation(x)
        x = self.c_dropout(x)
        x = self.c_output(x)
        return x

class GPT2Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate=0.1):
        super(GPT2Layer, self).__init__()
        self.attention = GPT2Attention(hidden_size, num_heads, dropout_rate)
        self.intermediate = GPT2MLP(hidden_size, intermediate_size, dropout_rate)

        self.attention_norm = GPT2LayerNorm(hidden_size)
        self.output_norm = GPT2LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        x = hidden_states + self.dropout(attention_output)
        x = self.attention_norm(x)

        intermediate_output = self.intermediate(x)
        x = x + self.dropout(intermediate_output)
        x = self.output_norm(x)

        return x

class GPT2Model(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072, dropout_rate=0.1):
        super(GPT2Model, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([GPT2Layer(hidden_size, num_heads, intermediate_size, dropout_rate) for _ in range(num_layers)])

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        hidden_states = embeddings

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states

# Testing the GPT-2 model
vocab_size = 50000  # Adjust based on your actual vocabulary size
model = GPT2Model(vocab_size)

# Loading the original GPT-2 125M model checkpoints and running a sample prediction
checkpoint_path = "D:\\Contlo\\download_model.py" #Tougest part was to find this file any seting up the right environment for the same
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = GPT2LMHeadModel.from_pretrained(checkpoint_path)

# Sample prediction
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # Replace with your input sequence
output = model(input_ids)
print(output)
