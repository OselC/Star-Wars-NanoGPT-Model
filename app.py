import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import streamlit as st

# Hyperparameters
batch_size = 64 # Number of independent sequences will be processed in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# -------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x

# Super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

# --- Load Metadata and Model Weights ---
def load_saved_model(weights_path, meta_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model on device: {device}...")

    # Load mappings
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    stoi = meta['stoi']
    itos = {int(k): v for k, v in meta['itos'].items()}
    vocab_size = len(stoi)

    # Instantiate structure and map state weights
    model = BigramLanguageModel(vocab_size)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, stoi, itos, device

# Streamlit UI
st.title("Ask NanoGPT")

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

weights_path = 'trained_gpt_model/gpt_char_model.pth'
meta_path = 'trained_gpt_model/model_meta.json'

try:
    model, stoi, itos, device = load_saved_model(weights_path, meta_path)
except FileNotFoundError:
    st.markdown("Error: Could not find 'gpt_char_model.pth' or 'model_meta.json'. Please export them first!")

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])


# Prompt input template
user_input = st.chat_input('Type your prompt here')

# If the user hits enter
if user_input:
    # Display the prompt
    st.chat_message('user').markdown(user_input)

    st.session_state.messages.append({'role':'user', 'content':user_input})

    # Convert prompt to tensor
    encoded_prompt = encode(user_input)
    if not encoded_prompt:
        continuation = "(Your prompt contained no characters known to my vocabulary!)"

    else:
        context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
        
        # Generating everything at once
        generated_tensor = model.generate(context, max_new_tokens=500)
        
        # We only want to print what comes *after* the user's prompt
        full_text = decode(generated_tensor[0].tolist())
        continuation = full_text[len(user_input):]

    # Show the LLM Response
    st.chat_message('assistant').markdown(continuation)
    # Store the LLM response in state
    st.session_state.messages.append({'role':'assistant', 'content':continuation})