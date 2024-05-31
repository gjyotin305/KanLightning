# KANFORMER

this subrepository contains the code for the KAN based transformer with tentative results
 
 ## Goals

 - [X] It is a decoder-only Transformer where we use Mixture of experts the experts are Feed-forward networks composed of KAN layers. 
 - [X] I use Rotary Position Embedding (RoPE) to encode the token positions, 
 - [X] the attention mechanism is a vanilla Multihead-attention layer where the query-key-value projector is a KAN layer.
 - [ ] Test it on some Transformer related task

