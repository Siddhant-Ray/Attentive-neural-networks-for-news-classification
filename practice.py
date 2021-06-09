from __future__ import unicode_literals, print_function, division
from io import open 
import glob
import os

"""Data preprocessing"""

def findFiles(path): return glob.glob(path)

#print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def unitoAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

#print(unitoAscii('Ślusàrski'))

category_lines = {}
all_categories = []

def readFilelines(filename):
    lines = open(filename , encoding = 'utf-8').read().strip().split('\n')    
    return [unitoAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readFilelines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

#print(category_lines['Italian'][:5])

"""Helper functions to create tensors"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1 , n_letters)
    for li , letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1 
    return tensor 

#print(letterToTensor('J'))    
#print(lineToTensor('Jack').size())

"""RNN definition"""

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_to_output = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        output = self.softmax(output)

        return output, hidden 

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


rnn = RNN(n_letters, n_hidden, n_categories).to(device)
print(rnn) 


'''input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input, hidden)

input = lineToTensor("Albert")
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
#print(output)'''

def categoryFromOutput(output):
    top, top_index = output.topk(1)
    category_index = top_index[0].item()
    return all_categories[category_index], category_index

#print(categoryFromOutput(output))

import random 

def randomChoice(l):
    return l[random.randint(0, len(l) - 1) ]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print("category = ", category, "line = ", line)



criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden().to(device)

    rnn.zero_grad()

    for line in range(0, line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[line], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for parameter in rnn.parameters():
        parameter.data.add_(parameter.grad.data, alpha = -learning_rate)

    return output, loss.item()

import time
import math 

num_iter = 100000
print_after_every = 5000
plot_after_every = 1000

current_loss = 0
all_losses = []

def timeFromStart(start):
    now = time.time()
    s = now - start
    m = math.floor(s/60)
    s -= m*60
    return "%dm %ds" % (m, s)

start = time.time()

for iteration in range(1, num_iter + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    category_tensor = category_tensor.to(device)
    line_tensor = line_tensor.to(device)
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    if iteration % print_after_every == 0:
        guess, guess_index = categoryFromOutput(output)
        if guess == category:
            correct = 'yes'
        else: 
            correct = 'no (%s)'%category
        print("%d %d%% (%s) %.4f %s / %s %s" % (iteration, iteration / num_iter * 100, timeFromStart(start), loss, line, guess, correct))

    if iteration % plot_after_every == 0:
        all_losses.append(current_loss/plot_after_every)
        current_loss = 0


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for item in range(0, line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[item], hidden)
    return output

for item in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor).to(device)
    guess, guess_index = categoryFromOutput(output)
    category_index = all_categories.index(category)
    confusion[category_index, guess_index] +=1
    
for num in range(n_categories):
    confusion[num] = confusion[num]/confusion[num].sum()
    

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for n_val in range(n_predictions):
            value = topv[0][n_val].item()
            category_index = topi[0][n_val].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi') 








