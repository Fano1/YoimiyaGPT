import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as f 

#making my own fucking gpt cuz i ran out of gpt 4 model

#lets start by using data
with open("datask.txt", "r") as f:
    text = f.read()

#all possible character
chars = set(text)
lst = sorted(list(chars))
print(lst)
vocabSize = len(lst)

#Encoder and decoder
#we can use tiktoken too
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[x] for x in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.Tensor(encode(lst[:1000]))

#lets make the train and test data split set kinda unsupervised learning
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

block_size = 8 #chunk of text to train the data on
train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.long(), y.long()

xb, yb = get_batch('train')

class BigramlanguageModel(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.tokenEmbedding_table = nn.Embedding(vocabSize, vocabSize)
    
    def forward(self, idx, targets):
        logits = self.tokenEmbedding_table(idx) #BTC

        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)


        loss = f.cross_entropy(logits, targets)
        
        return logits, loss


m = BigramlanguageModel(vocabSize=vocabSize)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

