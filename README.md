# reiterating-transformers
This is an attempt at a firsthand reinvestigation of the history of 'RNNs to transformers' and comparing their individual advantages.

## corpus
Throughout the initial phases of this endeavor, we use the corpus, 'Alice in Wonderland', which is available publicly, and I got my hands on it at https://www.gutenberg.org/ebooks/11 

Initially, we focus on character-level generation and hence have a tiny vocabulary of around 60 characters, as present in the corpus. 

## Next character predication using RNNs.
The RNNS follow a simpler architecture where we use the previous hidden state and current input to generate the next hidden state, which is ultimately used to guess the next character output.
The raw input data is converted to character tokens and segmented progressively based on a sequence length. This allows each letter to be trained based on past seq_length characters.
Once batched together, the data is ready for model training.
### The Model: Our Vanilla RNN has the following architecture: 
Variables->
<img width="621" height="184" alt="rnn_var" src="https://github.com/user-attachments/assets/82726276-87ad-4934-a795-378ff894d14a" />
Forward Pass ->
<img width="777" height="71" alt="rnn_forward" src="https://github.com/user-attachments/assets/e259a19c-67b9-479b-bee9-e34b540de3ed" />
Repeating training character sequences eventually trains the embedding and Wx & Wh to handle meaning and form sensible words.
### Observation
The model ended up generating a few understandable words based on the corpus, but it is highly unstable and soon loses its ability to generate meaningful words with length.
Tweaking a few parameters shows a very slight improvement in the output meaning.

##  character predictions using LSTM
LSTMs introduce more operations in a single pass of the model, introducing context pass along with hidden states. They are represented via C_t and h_t and are operated upon using four main intended operations: forget gate focuses on what must be forgotten from the current context (Wf), next we have a combination which focuses on what to add to the context and how much to add it (Wi & Wc), while the last one focuses on making hte current layers ready for next pass(Wo).
### The Model: Our Vanilla LSTM has the following architecture
Variables->
<img width="672" height="322" alt="image" src="https://github.com/user-attachments/assets/bd7898cb-5b69-410c-a093-08ba58d8bac0" />
Forward pass->
<img width="784" height="291" alt="image" src="https://github.com/user-attachments/assets/905e80b2-0332-4639-9161-bfaf21387fce" />
### Observations
This time the model generates the words more robustly and has better performance in meaningful words but that is the extent as the sentence pertains no meaning and is highly ambiguous

After these observations, it was clear what the limits of character-level embeddings and generations were, and hence we moved on to word-level generations.
From here on out, we use the vocabulary of unique words gathered from the corpus, and hence we see a significant rise in vocab_size. This ensures we always get meaningful words, and the model can focus more on semantic meaning and inter-word relations while forming a sentence.

## Word-wise generation via LSTM
We use the same architecture for the LSTM model, making forward passes with embeddings of words instead of characters. So the training process goes smoothly in a familiar manner. 
The data is also processed so that each word is trained upon the previous x words.
### Observations
We see the sentence starts off with little meaning and eventually loses all context and doesn't show basic English formations. This primarily happens due to the cost & parameter trade-off.
A portion of the outcome looks like this:
```Alice coaxing yourself , the little timidly turned all day , and reduced the king , and the jury “ advance — ” “ i ’ m a poor man , your majesty , ” thought alice . “ fifteenth , ” said the dormouse . “ i think you ’ re a trial ? ” said the hatter . “ i can ’ t help it more , ” said the hatter . “ write that first witness ! ” said the hatter . “ i give your minute , ” said the gryphon . “ and some of his garden_ . ’ ” “ i beg pardon , ” beautifully the king , and he got up the court and crossed his teacup and - butter , and , being in such confusion hanging as serpents ! ” the hatter threw on in a whisper voice . “ yes , ” said the hatter . “ _stolen ! _ ” the dormouse hastily replied ; “ after all the way of the trial ! ” the king said to the jury , and began staring at alice ; and the queen jumped were all crowded at the dormouse .```

Since LSTMs focus on word-wise training, we end up requiring much heavier computations and exponential Time Complexity. This is where the transformers show their magic as they train upon a chunk of sentences all at once using the attention mechanism. This aligns directly with the parallel processing nature of the GPU and increases training speed and efficiency.
A general transformer’s architecture, as described in ‘Attention is all you need’, has both encoder and decoder focusing on understanding meaning for classification and generation, respectively. 
While models like BERT, which specializes in sentence classification and similar tasks, are an encoder-heavy model, GPT, which focuses on producing full sentences, is a decoder-heavy model. Similar to this, we will implement a decoder-only transformer with only one layer first and then see what we can tweak to increase efficiency.

##  Word-wise generation using Decoder Transformer
We begin implementation by pre-processing the data in a slightly different manner, using a PyTorch DataLoader, which does a better job in getting the corpus training data ready. 
The Transformer initially only has one attention layer and one forward layer. 
### The Model: The transformer has the following architecture
Positional Encoding -> 
<img width="656" height="367" alt="image" src="https://github.com/user-attachments/assets/1ff048b5-642c-4be6-aeb5-5fadc8c636e0" />
Self attention->
<img width="263" height="68" alt="image" src="https://github.com/user-attachments/assets/bd87f480-b6ba-4918-be43-265d2060b764" />
The actual attention implementation requires several calculations: 
<img width="598" height="329" alt="image" src="https://github.com/user-attachments/assets/facc86c8-c193-40d0-848a-2addf7e1b5c1" />
Forward Pass->
<img width="783" height="67" alt="image" src="https://github.com/user-attachments/assets/e1627047-ca7d-46df-aee9-69d4912ee8f7" />
### Observations
we see the sentence gets a performance boost even though we still dont get a grammatically correct sentence. This is primarily due to a single transformer missing enough parameters to store all sentence information.
Hence, we make the following tweaks:
- Normalization layer after both attention and the forward layer 
- Pre LN instead of Post LN
- 6 layers instead of 1 
- Dropout in the transformer block to prevent over-fitting.

After training the final model with all the adjustments , we get a result which looks this:
```alice was sad and lonely on a little ledge of rock, and, as they came nearer, alice could hear him sighing as if his heart would break. she pitied him deeply. “ what is his sorrow ? ” she asked the gryphon, and the gryphon answered, very nearly in the same words as before, ```

As suggested in the original transformer paper, the more attention layers we have, the more inter-token relations form on a wider distance. We can observe that in the attention heatmap for this 6 layered transformer:
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/32145099-2b0a-411b-8ee8-8187518e979f" />
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/ffb715bf-3cf9-468b-9067-7a966cded9ab" />
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/7fbf0c79-9e53-4852-a86b-58738e71656c" />
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/9843be50-976d-4f70-9e69-5f5c49d32906" />
<img width="450" height="400" alt="image" src="https://github.com/user-attachments/assets/116d45b3-dd65-4593-9ff5-0d7dd2139c4b" />





