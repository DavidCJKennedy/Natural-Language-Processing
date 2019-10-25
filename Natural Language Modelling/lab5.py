# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
trigrams = []
vocab = []
word_to_ix = {}

trainingSet = ["""The mathematician ran .""", 
               """The mathematician ran to the store .""", 
               """The physicist ran to the store .""",
               """The philosopher thought about it .""", 
               """The mathematician solved the open problem ."""]

for sentence in trainingSet:
    split = sentence.split()
    split = ["START"] + split
        
    trigrams.append([([split[i], split[i + 1]], split[i + 2])
            for i in range(len(split) - 2)])

    vocab.extend(set(split))
    word_to_ix.update({word: i for i, word in enumerate(vocab)})


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(200):
    total_loss = torch.Tensor([0])
    count = 0
    for trigram in trigrams:
        count = count + 1
        targets = []
        predictions = []
        for context, target in trigram:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in variables)
            context_idxs = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idxs))

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_var)
            log_probs_np = log_probs.detach()
            log_probs_np = np.array(log_probs_np)
            index = np.argmax(log_probs_np[-1])
                        
            targets.append(target)
            predictions.append(vocab[index])
                        
            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a variable)
            loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            losses.append(total_loss)
        
        if count == 2 and epoch == 99:
            print(targets)
            print(predictions)
            
            targs = np.array(targets)
            preds = np.array(predictions)
        
            score = np.sum(targs == preds)
            print("Total number of targets correct: " + str(score) + ", out of: " + str(len(trigram)))
        

mathematician = model.embeddings(torch.LongTensor([vocab.index("mathematician")]))
physicist = model.embeddings(torch.LongTensor([vocab.index("physicist")]))
philosopher= model.embeddings(torch.LongTensor([vocab.index("philosopher")]))

cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
physicistCos = cos(mathematician, physicist)
philosopherCos = cos(mathematician, philosopher)

print(physicistCos)
print(philosopherCos)