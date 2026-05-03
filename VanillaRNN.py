import torch
import torch.nn as nn

with open('/content/Alice_in_wonderland.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#print(len(text))
#print(text[:500])
text = text.replace('\r','')
text = text.replace('\n',' ')
text = text.replace('  ',' ')
chars = sorted(list(set(text)))
vocab_size = len(chars)
#print(chars)
#print(vocab_size)

char_to_idx = {ch: i for i,ch in enumerate(chars)}
idx_to_char = {i: ch for ch,i in char_to_idx.items()}

#print(char_to_idx['A'])
#print(idx_to_char[10])
encoded_text = [char_to_idx[ch] for ch in text]
seq_length = 80

sequences = []
targets = []
for i in range(0, len(encoded_text) - seq_length):
  seq = encoded_text[i:i+seq_length]
  target = encoded_text[i+seq_length]
  sequences.append(seq)
  targets.append(target)

X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)

#print(X.shape, y.shape)
split = int(0.9*(len(X)))

X_train,X_val = X[:split],X[split:]
y_train,y_val = y[:split],y[split:]

class VanillaRNN(nn.Module):
  def __init__(self,vocab_size,embed_size,hidden_size):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size,embed_size);
    self.Wx = nn.Linear(embed_size,hidden_size)
    self.Wh = nn.Linear(hidden_size,hidden_size)

    self.fc = nn.Linear(hidden_size,vocab_size)

  def forward(self,x,h_prev):
    batch_size, seq_length = x.size()
    h = h_prev

    for t in range(seq_length):
      x_t = self.embedding(x[:,t])
      h = torch.tanh(self.Wx(x_t) + self.Wh(h))

    out  = self.fc(h)
    return out, h

vocab_size = len(chars)
embed_size = 32
hidden_size = 128
model = VanillaRNN(vocab_size,embed_size, hidden_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

## training loop
batch_size = 64
epochs = 12

def get_batches(X,y,batch_size):
  for i in range(0,len(X),batch_size):
    yield X[i:i+batch_size], y[i:i+batch_size]

for epoch in range(epochs):
  total_loss = 0
  number_of_batches = 0

  for xb, yb in get_batches(X_train,y_train,batch_size):
    h = torch.zeros(xb.size(0),hidden_size)
    if xb.size(0) != batch_size:
      continue

    optimizer.zero_grad()

    outputs, h = model(xb,h.detach())

    loss = criterion(outputs,yb)
    loss.backward()
    optimizer.step()

    number_of_batches += 1
    total_loss += loss.item()

  print(f"Epoch {epoch+1},Average Loss: {(total_loss/number_of_batches):.4f}")

## text generation 
def generate_text(model, start_text, length=200):
  model.eval()

  h = torch.zeros(1,hidden_size)

  chars_input = [char_to_idx[ch] for ch in start_text]

  for ch in chars_input[:-1]:
    x = torch.tensor([[ch]])
    _,h = model(x,h)

  current_char = chars_input[-1]
  result = start_text

  for _ in range(length):
    x = torch.tensor([[current_char]])
    out,h = model(x,h)

    probs = torch.softmax(out,dim = 1)
    current_char = torch.multinomial(probs,1).item()

    result += idx_to_char[current_char]

  return result

print(generate_text(model,"Wonderland ",200))
