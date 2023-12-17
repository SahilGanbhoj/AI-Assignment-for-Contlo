import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
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
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
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

class GPT2RotaryPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super(GPT2RotaryPositionalEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.alpha = nn.Parameter(torch.zeros(1, hidden_size // 2))
        self.beta = nn.Parameter(torch.zeros(1, hidden_size // 2))

    def forward(self, x):
        seq_len = x.size(1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * -(math.log(10000.0) / self.hidden_size))
        sinusoid = torch.sin(position * div_term)

        rotary_emb = torch.cat([sinusoid, torch.cos(position * div_term)], dim=-1)
        rotary_emb = rotary_emb.unsqueeze(0).expand(x.size(0), -1, -1)

        return x + self.alpha * rotary_emb[:, :, :self.hidden_size // 2] + self.beta * rotary_emb[:, :, self.hidden_size // 2:]

class GPT2GroupQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, group_size, dropout_rate=0.1):
        super(GPT2GroupQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_size = group_size
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
        batch_size, seq_len, _ = hidden_states.size()
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        # Reshape tensors to group queries
        query = query.view(batch_size, self.num_heads, self.group_size, -1)
        key = key.view(batch_size, self.num_heads, self.group_size, -1)
        value = value.view(batch_size, self.num_heads, self.group_size, -1)

        # Perform attention within groups
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(self.head_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.head_dim * self.num_heads)

        # Linear transformation for output
        output = self.out(context)
        return output


class GPT2SlidingWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size, dropout_rate=0.1):
        super(GPT2SlidingWindowAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
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
        batch_size, seq_len, _ = hidden_states.size()
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))

        # Pad the sequence for sliding window
        hidden_states = F.pad(hidden_states, (0, 0, self.window_size // 2, self.window_size // 2), value=0)

        # Perform sliding window attention
        output = []
        for i in range(seq_len):
            window_query = query[:, :, i:i + self.window_size, :].contiguous().view(batch_size, self.num_heads, -1, self.head_dim)
            window_key = key[:, :, i:i + self.window_size, :].contiguous().view(batch_size, self.num_heads, -1, self.head_dim)
            window_value = value[:, :, i:i + self.window_size, :].contiguous().view(batch_size, self.num_heads, -1, self.head_dim)

            attention_scores = torch.matmul(window_query, window_key.transpose(-1, -2))
            attention_scores = attention_scores / torch.sqrt(self.head_dim)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            context = torch.matmul(attention_probs, window_value)
            context = context.view(batch_size, self.num_heads, -1, self.head_dim)
            context = context.permute(0, 2, 1, 3).contiguous()
            context = context.view(batch_size, -1, self.head_dim * self.num_heads)

            output.append(context)

        output = torch.stack(output, dim=1)
        output = self.out(output)
        return output


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072, dropout_rate=0.1):
        super(GPT2Model, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.rotary_positional_embedding = GPT2RotaryPositionalEmbedding(hidden_size)  # Use Rotary Positional Embedding
        self.layers = nn.ModuleList([GPT2Layer(hidden_size, num_heads, intermediate_size, dropout_rate) for _ in range(num_layers)])
        self.sliding_window_attention = GPT2SlidingWindowAttention(hidden_size, num_heads, dropout_rate)  # Add Sliding Window Attention

    def forward(self, input_ids):
        embeddings = self.embeddings(input_ids)
        embeddings = self.rotary_positional_embedding(embeddings)  # Apply Rotary Positional Embedding

        for layer in self.layers:
            embeddings = layer(embeddings)

        # Apply Sliding Window Attention
        embeddings = self.sliding_window_attention(embeddings)

        return embeddings
