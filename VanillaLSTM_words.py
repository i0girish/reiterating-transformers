import re
import torch
import torch.nn as nn

with open('/content/Alice_in_wonderland.txt','r',encoding='utf-8') as f:
  text = f.read().lower()

tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
vocab = sorted(set(tokens))
vocab_size = len(vocab)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
encoded_text = [word2idx[w] for w in tokens]

seq_length = 20
sequences = []
targets = []
for i in range(0,len(encoded_text) - seq_length):
  seq = encoded_text[i:i+seq_length]
  target = encoded_text[i+seq_length]
  sequences.append(seq)
  targets.append(target)
 
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)
split = int(0.9*len(X))

X_train,X_val = X[:split],X[split:]
y_train, y_val = y[:split],y[split]

class VanillaLSTM(nn.Module):
  def __init__(self,vocab_size,embed_size,hidden_size):
    super().__init__()

    self.hidden_size = hidden_size
    self.embedding = nn.Embedding(vocab_size,embed_size)

    self.Wf = nn.Linear(embed_size + hidden_size, hidden_size)
    self.Wi = nn.Linear(embed_size + hidden_size, hidden_size)
    self.Wc = nn.Linear(embed_size + hidden_size, hidden_size)
    self.Wo = nn.Linear(embed_size + hidden_size, hidden_size)

    self.fc = nn.Linear(hidden_size, vocab_size)

  def forward(self,x, h_prev, c_prev):
    batch_size,seq_len = x.size()
    h = h_prev
    c = c_prev

    for t in range(seq_len):
      x_t = self.embedding(x[:,t])

      z = torch.cat((x_t,h),dim=1)

      f_t = torch.sigmoid(self.Wf(z))
      i_t = torch.sigmoid(self.Wi(z))
      c_tilde = torch.tanh(self.Wc(z))

      c = f_t * c + i_t * c_tilde

      o_t = torch.sigmoid(self.Wo(z))
      h = o_t * torch.tanh(c)

    out = self.fc(h)
    return out,h,c

embed_size = 64
hidden_size = 128

model = VanillaLSTM(vocab_size, embed_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def get_batches( X, y, batch_size):
  for i in range(0,len(X),batch_size):
    yield X[i:i+batch_size], y[i:i+batch_size]

batch_size = 32
epochs = 15

for epoch in range(epochs):
  total_loss = 0;
  num_batches = 0;

  for xb, yb in get_batches(X_train, y_train, batch_size):

    if xb.size(0) != batch_size:
      continue

    h = torch.zeros(xb.size(0), hidden_size)
    c = torch.zeros(xb.size(0), hidden_size)

    optimizer.zero_grad()

    outputs, h , c = model(xb, h, c)
    loss = criterion(outputs, yb)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    num_batches += 1

  print(f"Epoch {epoch+1}, Avg Loss: {total_loss/num_batches:.4f}")

def generate_text_lstm(model, start_text, length= 300):
  temperature = 0.7
  model.eval()

  h = torch.zeros(1, hidden_size)
  c = torch.zeros(1, hidden_size)

  # Convert start_text to lowercase to match the vocabulary
  start_text_lower = start_text.lower()
  input_words = re.findall(r"\b\w+\b|[^\w\s]", start_text_lower)
  chars_input = [word2idx[ch] for ch in input_words]

  for ch in chars_input[:-1]:
    x = torch.tensor([[ch]])
    _,h,c = model(x,h,c)

  current_char = chars_input[-1]
  result = start_text

  for _ in range(length):
    x = torch.tensor([[current_char]])
    out, h,c = model(x,h,c)

    probs = torch.softmax(out/temperature,dim=1)
    current_char = torch.multinomial(probs,1).item()

    result += ' ';
    result += idx2word[current_char]

  return result
print(generate_text_lstm(model,"Alice",500))
