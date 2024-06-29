from KanFormerRevamp.batch_iterator import BatchIterator
from KanFormerRevamp.early_stopping import EarlyStopping
from KanFormerRevamp.kanformer_model import Transformer, LabelSmoothingLoss
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from tqdm import tqdm_notebook
import numpy as np
import wandb
from sklearn.metrics import confusion_matrix

max_len = 199

wandb.init(project="kanert_no_lightning")

# Import the dataset. Use clean_review and label columns
train_dataset = pd.read_csv('/home/palashdas/jyotin/.KanBERT/imdb/train_dataset_feat_clean.csv')

# Change columns order
train_dataset = train_dataset[['clean_review', 'label']]

val_dataset = pd.read_csv('/home/palashdas/jyotin/.KanBERT/imdb/val_dataset_feat_clean.csv')

# Change columns order
val_dataset = val_dataset[['clean_review', 'label']]

print(val_dataset.head())

batch_size = 64

train_iterator = BatchIterator(train_dataset, batch_size=batch_size, vocab_created=False, vocab=None, target_col=None,
                               word2index=None, sos_token='<SOS>', eos_token='<EOS>', unk_token='<UNK>',
                               pad_token='<PAD>', min_word_count=3, max_vocab_size=None, max_seq_len=0.9,
                               use_pretrained_vectors=False, glove_path='/home/palashdas/jyotin/.KanBERT/glove/', glove_name='glove.6B.100d.txt',
                               weights_file_name='/home/palashdas/jyotin/.KanBERT/glove/weights.npy')

val_iterator = BatchIterator(val_dataset, batch_size=batch_size, vocab_created=False, vocab=None, target_col=None,
                             word2index=train_iterator.word2index, sos_token='<SOS>', eos_token='<EOS>',
                             unk_token='<UNK>', pad_token='<PAD>', min_word_count=3, max_vocab_size=None,
                             max_seq_len=0.9, use_pretrained_vectors=False, glove_path='/home/palashdas/jyotin/.KanBERT/glove',
                             glove_name='glove.6B.100d.txt', weights_file_name='/home/palashdas/jyotin/.KanBERT/glove/weights_val.npy')


# Initialize parameters
vocab_size = len(train_iterator.word2index)
dmodel = 64
output_size = 2
padding_idx = train_iterator.word2index['<PAD>']
n_layers = 4
ffnn_hidden_size = dmodel * 2
heads = 8
pooling = 'max'
dropout = 0.5
label_smoothing = 0.1
learning_rate = 0.001
epochs = 100

# Check whether system supports CUDA
CUDA = torch.cuda.is_available()

model = Transformer(vocab_size, dmodel, output_size, max_len, padding_idx, n_layers,\
                    ffnn_hidden_size, heads, pooling, dropout)

# Move the model to GPU if possible
if CUDA:
    model.cuda()
    
# Add loss function    
if label_smoothing:
    loss_fn = LabelSmoothingLoss(output_size, label_smoothing)
else:
    loss_fn = nn.NLLLoss()
    
model.add_loss_fn(loss_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.add_optimizer(optimizer)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
model.add_scheduler(scheduler)

device = torch.device('cuda' if CUDA else 'cpu')

model.add_device(device)

# Create the parameters dictionary and instantiate the tensorboardX 

# Instantiate the EarlyStopping
early_stop = EarlyStopping(wait_epochs=3)

train_losses_list, train_avg_loss_list, train_accuracy_list = [], [], []
eval_avg_loss_list, eval_accuracy_list, conf_matrix_list = [], [], []

for epoch in range(epochs):

    print('\nStart epoch [{}/{}]'.format(epoch+1, epochs))

    train_losses, train_avg_loss, train_accuracy = model.train_model(train_iterator)

    train_losses_list.append(train_losses)
    train_avg_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_accuracy)

    _, eval_avg_loss, eval_accuracy = model.evaluate_model(val_iterator)

    eval_avg_loss_list.append(eval_avg_loss)
    eval_accuracy_list.append(eval_accuracy)
    # conf_matrix_list.append(conf_matrix)

    print('\nEpoch [{}/{}]: Train accuracy: {:.3f}. Train loss: {:.4f}. Evaluation accuracy: {:.3f}. Evaluation loss: {:.4f}'\
            .format(epoch+1, epochs, train_accuracy, train_avg_loss, eval_accuracy, eval_avg_loss))

    wandb.log({"train_loss": train_avg_loss, "train_acc": train_accuracy, "eval_acc": eval_accuracy, "eval_loss": eval_avg_loss})