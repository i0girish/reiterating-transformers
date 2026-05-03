import re
import torch
from torch.utils.data import Dataset, DataLoader
import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

## dataset class
class WordDataset(Dataset):
  def __init__(self, data, seq_length):
    self.data = data
    self.seq_length = seq_length

  def __len__(self):
    return len(self.data) - self.seq_length

  def __getitem__(self,idx):
    x = self.data[idx:idx+self.seq_length]
    y = self.data[idx+1:idx+self.seq_length+1]
    return torch.tensor(x), torch.tensor(y)

seq_length = 20
batch_size = 64

dataset = WordDataset(encoded_text, seq_length)
loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)

with open('/content/Alice_in_wonderland.txt','r',encoding='utf-8') as f:
  text = f.read().lower()
tokens = re.findall(r"\b\w+\b|[^\w\s]", text)

vocab = sorted(set(tokens))
vocab_size = len(vocab)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
encoded_text = [word2idx[w] for w in tokens]

## positional encoder class
class PositionalEncoding(nn.Module):
  def __init__(self,d_model, max_len = 200):
    super().__init__()
    pe = torch.zeros(max_len,d_model)
    pos = torch.arange(0,max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0,d_model,2) *(-math.log(10000.0)/ d_model))

    pe[:,0::2] = torch.sin(pos * div)
    pe[:,1::2] = torch.cos(pos * div)

    self.pe = pe.unsqueeze(0)

  def forward(self,x):
    return x+self.pe[:,:x.size(1)].to(x.device)

## self attention class 
class SelfAttention(nn.Module):
  def __init__(self,d_model,heads):
    super().__init__()
    self.d_model = d_model
    self.heads = heads
    self.head_dim = d_model // heads

    self.qkv = nn.Linear(d_model, 3*d_model)
    self.fc = nn.Linear(d_model, d_model)

  def forward(self,x,return_attn=False):
    B,T,C = x.shape
    qkv = self.qkv(x)
    qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # causal mask
    mask = torch.tril(torch.ones(T, T)).to(x.device)
    scores = scores.masked_fill(mask == 0, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    out = attn @ v

    out = out.transpose(1, 2).reshape(B, T, C)
    if return_attn:
      return self.fc(out), attn
    return self.fc(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden,p_drop):
        super().__init__()
        self.attn = SelfAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.GELU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x,return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x, attn
        else:
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, heads=4, ff_hidden=256, p_drop = 0.1,n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, ff_hidden, p_drop) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x,return_attn=False):
        x = self.embed(x)
        x = self.pos(x)
        all_attn = []

        for block in self.blocks:
            if return_attn:
                x, attn = block(x, return_attn=True)
                all_attn.append(attn)
            else:
                x = block(x)

        x = self.ln(x)
        logits = self.fc(x)

        if return_attn:
            # list of attention maps from each layer
            return logits, all_attn
        return logits

## training 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TinyTransformerLM(len(vocab)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 9

for epoch in range(epochs):
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        out = model(xb)
        loss = criterion(out.view(-1, len(vocab)), yb.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

def generate_text(start_words, length=30, temp=0.8):
  model.eval()
  words = start_words[:]

  for _ in range(length):
    x = torch.tensor([[word2idx[w] for w in words[-seq_length:]]]).to(device)
    with torch.no_grad():
        logits = model(x)[0, -1]
    probs = torch.softmax(logits / temp, dim=0)
    idx = torch.multinomial(probs, 1).item()
    words.append(idx2word[idx])

  text = ' '.join(words)
  text = text.replace(' ,', ',').replace(' .', '.')
  return text



def generate_with_attention(model, start_tokens, stoi, itos,
                            max_new_tokens=30, device='cpu', temperature=1.0):
    model.eval()
    tokens = start_tokens[:]

    last_all_attn = None

    for _ in range(max_new_tokens):
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, all_attn = model(x, return_attn=True)

        logits = logits[0, -1] / temperature
        probs = F.softmax(logits, dim=0)
        next_token = torch.multinomial(probs, 1).item()

        tokens.append(next_token)

        # keep attention from latest step
        last_all_attn = all_attn

    words = [itos[t] for t in tokens]
    return words, last_all_attn

def plot_attention(words, all_attn, layer=0, head=0):
    """
    all_attn: list of attention maps from each layer
              each item: (B, heads, T, T)
    """

    attn_map = all_attn[layer][0, head].cpu().numpy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(attn_map,
                xticklabels=words,
                yticklabels=words)
    plt.title(f'Attention Heatmap — Layer {layer}, Head {head}')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.show()


start_words = ["alice", "was"]
start_tokens = [word2idx[w] for w in start_words]

words, all_attn = generate_with_attention(
    model, start_tokens, word2idx, idx2word,
    max_new_tokens=25,
    device=device,
    temperature=0.9
)

print("Generated:\n", " ".join(words))

# Try different layers and heads
plot_attention(words, all_attn, layer=4, head=0)
plot_attention(words, all_attn, layer=5, head=1)
plot_attention(words, all_attn, layer=1, head=2)
