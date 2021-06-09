import numpy as np
import random
import os
from pathlib import Path

import torch
import torch.nn as nn 
from torch import optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizerFast
from transformers import DistilBertModel, DistilBertConfig
from transformers import AdamW

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import seaborn as sns
from pylab import savefig
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker


from model import ModDistilBertForSequenceClassification


from helper import DatasetLoader
from helper import newsDataset
from helper import Metrics

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# Load the dataset using the helper class
data_set = DatasetLoader()

# Calculate the number of categories and get the dictionary mapping categories to news entries
number_of_categories, category_news  = data_set.get_categories()

# Return the news entries (>=5 in length) and their categories
list_of_sentences, labels = data_set.get_sentences_and_labels()

print(number_of_categories)

# Return the number of items per category
count_of_news_entries_per_class = data_set.class_item_count()
#print(count_of_news_entries_per_class)

print(len(list_of_sentences))
print(len(labels))

label_list = list(category_news.keys())
index_class_map_dict1={}

for idx, value in enumerate(label_list):
    index_class_map_dict1[value]=idx

#print(index_class_map_dict1)
label_values = list(index_class_map_dict1.values())

# Convert the text labels for the categories to unique integers
number_labels=[]

for label in labels:
    number_labels.append(index_class_map_dict1[label])

train_texts = list_of_sentences
train_labels = number_labels

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

# Tokenizer for the DistilBERT model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, return_tensors='pt', truncation=True, padding=True)
val_encodings = tokenizer(val_texts, return_tensors='pt', truncation=True, padding=True)
print(train_encodings.keys())

# Load the NewsDatset class which transforms input itnoa form recognized by PyTorch's DataLoader class
train_dataset = newsDataset(train_encodings, train_labels)
val_dataset = newsDataset(val_encodings, val_labels)

# Check if GPU exists for training the model, else default to CPU
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

# Load the model and cast it onto the device (GPU or CPU)
model = ModDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels = number_of_categories)
model.to(device)

# Set the weight decay for regularization
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0001}
]

# Set the optimizer for the training process
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# Data loader class from PyTorch splits the data into batches, with random shuffling
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# We use weighted CrossEntropy Loss as our loss function, weights are inverse of the class size
num_of_items_per_class = count_of_news_entries_per_class

weights = []
for i in num_of_items_per_class:
    weights.append(1/i)
    
# Weights must be manually cast to the device
class_weights = torch.FloatTensor(weights).to(device)

train_losses = []
train_acc = []
val_losses = []
val_acc = []
running_loss = 0
running_acc = 0

train_f1_score_macro = []
train_f1_score_raw = []

val_f1_score_macro = []
val_f1_score_raw = []

### Training function with validation ###
def train_model():

    global train_losses, train_acc
    global val_losses, val_acc
    global running_loss, running_acc
    global train_f1_score_macro, train_f1_score_raw 
    global val_f1_score_macro, val_f1_score_raw
    
    
    for epoch in tqdm(range(9)):

        model.train()

        pred_temp = 0
        true_temp = 0
        y_true = []
        y_pred = []

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, feat_for_tsne = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, labels, class_weights)
            accuracy = (outputs.logits.argmax(-1) == labels).float().sum()
            
            running_acc += accuracy.item()
            running_loss += loss.item()
            
            #predictions for f1 score
            pred_temp = outputs.logits.argmax(-1).cpu().detach().numpy()
            true_temp = labels.cpu().detach().numpy()


            for item in pred_temp:
                y_pred.append(item)

            for item in true_temp:
                y_true.append(item)

            loss.backward()
            optimizer.step()

        print("train_loss for epoch = {epoch}".format(epoch = epoch + 1 ), "is", running_loss/len(train_loader))
        train_losses.append(running_loss/len(train_loader))
        running_loss = 0
        
        print("train_acc for epoch = {epoch}".format(epoch = epoch + 1), "is", running_acc/len(train_texts))
        train_acc.append(running_acc/len(train_texts))
        running_acc = 0
        
        f1score = f1_score(y_true, y_pred, average="macro")
        f1score_none = f1_score(y_true, y_pred, average=None)
        
        print("train_F1 macro score for epoch = {epoch}".format(epoch = epoch + 1), "is", f1score)
        print("F1 raw score for epoch = {epoch}".format(epoch = epoch + 1), "is", f1score_none)
        
        train_f1_score_macro.append(f1score)
        train_f1_score_raw.append(f1score_none)
        
        pred_temp = 0
        true_temp = 0
        y_true = []
        y_pred = []

        ### Choose after how many epochs should validation be done
        if epoch%1 == 0: 

            model.eval()
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs, feat_for_tsne = model(input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs.logits, labels, class_weights)
                running_loss += loss.item()
                accuracy = (outputs.logits.argmax(-1) == labels).float().sum()
                running_acc += accuracy.item()
                
                #predictions for f1 score
                pred_temp = outputs.logits.argmax(-1).cpu().detach().numpy()
                true_temp = labels.cpu().detach().numpy()
                
                
                for item in pred_temp:
                    y_pred.append(item)
                
                for item in true_temp:
                    y_true.append(item)
                    
                            
                
        print("val_loss for epoch = {epoch}".format(epoch = epoch + 1), "is", running_loss/len(val_loader))
        val_losses.append(running_loss/len(val_loader))
        running_loss = 0
        print("val_acc for epoch = {epoch}".format(epoch = epoch + 1), "is", running_acc/len(val_texts))
        val_acc.append(running_acc/len(val_texts))
        running_acc = 0
        
        f1score = f1_score(y_true, y_pred, average="macro")
        f1score_none = f1_score(y_true, y_pred, average=None)
        
        print("F1 macro score for epoch = {epoch}".format(epoch = epoch + 1), "is", f1score)
        print("F1 raw score for epoch = {epoch}".format(epoch = epoch + 1), "is", f1score_none)
        
        val_f1_score_macro.append(f1score)
        val_f1_score_raw.append(f1score_none)


### Method to plot losses, accuracy and F1-scores for training and validation ###
def plot_losses():

    epochs =[]

    for i in range(9):
        epochs.append(i)

        plt.figure(figsize=(10,10))
        plt.plot(epochs, train_losses, linewidth=5)
        plt.plot(epochs, val_losses, linewidth=5)
        plt.legend(['train loss', 'val loss'], loc = 'upper right')
        plt.xticks(size = 20)
        plt.yticks(size = 20)

        path = os.getcwd()
        folder = "new_data_figs2"
        FILE = "loss-bert.png"

        path_save = os.path.join(path, folder)
        path_save = os.path.join(path_save, FILE)

        #plt.savefig(path_save)
        plt.show()

        plt.figure(figsize=(10,10))
        plt.plot(epochs, val_acc, linewidth =5)
        plt.legend(['val_acc'], loc = 'upper right')
        plt.xticks(size = 20)
        plt.yticks(size = 20)

        path = os.getcwd()
        folder = "new_data_figs2"
        FILE = "acc-bert.png"

        path_save = os.path.join(path, folder)
        path_save = os.path.join(path_save, FILE)

        #plt.savefig(path_save)
        plt.show()

        plt.figure(figsize=(10,10))
        plt.plot(epochs, train_f1_score_macro, linewidth =5)
        plt.plot(epochs, val_f1_score_macro, linewidth =5)
        plt.legend(['train_F1','val_F1'], loc = 'upper right')
        plt.xticks(size = 20)
        plt.yticks(size = 20)

        path = os.getcwd()
        folder = "new_data_figs2"
        FILE = "F1-bert.png"

        path_save = os.path.join(path, folder)
        path_save = os.path.join(path_save, FILE)

        #plt.savefig(path_save)
        plt.show()

        ### Run if you want to save the values to a file 
        '''f = open("values2.txt",'w')
        for i in range(len(epochs)):
            
            f.write("train_loss for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(train_losses[i]) + "\n")
            f.write("train_acc for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(train_acc[i]) + "\n")
            f.write("val_loss for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(val_losses[i]) + "\n")
            f.write("val_acc for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(val_acc[i]) + "\n")
            f.write("Training F1 macro score for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(train_f1_score_macro[i]) + "\n")
            f.write("Training F1 macro raw for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(train_f1_score_raw[i]) + "\n")
            f.write("Validation F1 macro score for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(val_f1_score_macro[i]) + "\n")
            f.write("Validation F1 macro raw for epoch = {epoch}".format(epoch = epochs[i]+1) + " is " + str(val_f1_score_raw[i]) + "\n")
            f.write("\n""\n")
        f.close()'''

### Method to save the trained model for future use ###
def save_model():

    folder = "new_bert_model_test"
    FILE = "model.pth"

    path_save = os.path.join(path, folder)
    path_save = os.path.join(path_save, FILE)

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path_save)

### Saved new model on reduced dataset
path = os.getcwd()

folder = "new_bert_model_test"
FILE = "model.pth"

path_save = os.path.join(path, folder)
path_save = os.path.join(path_save, FILE)

model_file = Path(path_save)

if model_file.is_file():
    # If saved model is present, load the saved model else train a model and save it
    checkpoint = torch.load(path_save)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

else:
    train_model()
    plot_losses()
    save_model()


### Validation phase for last epoch ###

pred_temp = 0
true_temp = 0
y_true = []
y_pred = []

all_actual_targets = []
all_top_three_preds = []

model.eval()

for idx, batch in enumerate(val_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs, feat_for_tsne = model(input_ids, attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, labels, class_weights)
    running_loss += loss.item()
    
    # Top three predictions for MRR calculation
    values, indices = torch.topk(outputs.logits , 3)
    
    top_three_preds = indices.cpu().detach().numpy()
    actual_targets = labels.cpu().detach().numpy()
    
    for item1 in top_three_preds:
        all_top_three_preds.append(item1)
        
    for item2 in actual_targets:
        all_actual_targets.append(item2)
    
    accuracy = (outputs.logits.argmax(-1) == labels).float().sum()
    running_acc += accuracy.item()

    # predictions for f1 score
    pred_temp = outputs.logits.argmax(-1).cpu().detach().numpy()
    true_temp = labels.cpu().detach().numpy()
        
    for item in pred_temp:
        y_pred.append(item)

    for item in true_temp:
        y_true.append(item)

# Calculate ranks for Mean Reciprocal Rank (MRR) calculation
ranks = [] 

for value in range(len(all_actual_targets)):
    if all_actual_targets[value] == all_top_three_preds[value][0]:
        ranks.append(1)
    elif all_actual_targets[value] == all_top_three_preds[value][1]:
        ranks.append(2)
    elif all_actual_targets[value] == all_top_three_preds[value][2]:
        ranks.append(3)
    else:
        ranks.append(0)
        

### Load the accuracy and MRR metrics
metrics = Metrics()

### Print the results for accuracy, F1-score and MRR
print("Accuracy for top prediction is :",metrics.accuracy(all_top_three_preds, all_actual_targets))
print("Accuracy for top 3 predictions is :", metrics.accuracyTop3(all_top_three_preds, all_actual_targets))
print("Mean Reciprocal Rank for top 3 predictions is :" , metrics.meanReciprocalRank(ranks))

f1score = f1_score(y_true, y_pred, average="macro")
f1score_none = f1_score(y_true, y_pred, average=None)

print("F1 macro score for epoch = {epoch}".format(epoch = epoch), "is", f1score)
print("F1 raw score for epoch = {epoch}".format(epoch = epoch), "is", f1score_none)