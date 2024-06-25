maxlen = 256 # maximum of length
batch_size = 4
max_pred = 5  # max tokens of prediction
n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 12 # number of heads in Multi-Head Attention
d_model = 768 # E
d_ff = 768 * 4  
d_k = d_v = 64
d_q = d_k
n_segments = 2
n_classify = 2
initial_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, "[UNK]": 4} 