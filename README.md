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
