import os
from io import open 
import random
import matplotlib
import matplotlib.pyplot as plt 

import unicodedata
import string

import pandas as pd
import torch
from torch.utils.data import DataLoader

class DatasetLoader(object):

    category_news = {}
    all_categories = []
    num_categories = 0
    all_letters = string.ascii_letters + ".,;'& "
    n_letters = len(all_letters)
    list_of_sentences = []
    labels =[]

    def __init__(self):
        pass

    def reset_states(self):
        self.category_news = {}
        self.all_categories = []
        self.num_categories = 0
        self.list_of_sentences = []
        self.labels =[]


    """ Convert the news entries to ASCII values"""    
    def unitoAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    ### Set folder as news_data_test/ or news_data_test2/ depending on which model to use.
    """ Read the news data files generated without or after combination """
    def readFile_byline(self,filename):
        line_of_news = open("news_data_test/"+filename , encoding = 'utf-8').read().split('\n')    
        return [self.unitoAscii(line) for line in line_of_news]

    """ Get the number of categories and the dictionary mapping eaxh entry ot its category"""    
    def get_categories(self):
        list_of_files = os.listdir("news_data_test/")

        #Dictionary which maps every category of news to its description
        for _file in list_of_files:
            category = _file.split(".")[0]
            self.all_categories.append(category)
            news_descp = self.readFile_byline(_file)
            self.category_news[category] = news_descp

        self.num_categories = len(self.all_categories)
        return (self.num_categories, self.category_news)

    """ Get the final news entries and labels after dropping news entries less than 5 in length"""
    def get_sentences_and_labels(self):
        self.reset_states()
        self.get_categories()

        category_plus_news_list = []

        for key in self.category_news.keys():
            for news_item in self.category_news[key]:
                category_plus_news_list.append((key, news_item))

        for pair in category_plus_news_list:
            label = pair[0]
            sentences = pair[1]
    
            if len(sentences.split(" ")) >= 5:
    
                self.list_of_sentences.append(sentences)
                self.labels.append(label)

        return (self.list_of_sentences, self.labels)


    """ Get the number of items in every news category as a list for future use"""
    def class_item_count(self):
        self.reset_states()
        self.get_sentences_and_labels()

        label_list = list(self.category_news.keys())
        index_class_map_dict1={}

        for idx, value in enumerate(label_list):
            index_class_map_dict1[value]=idx

        label_values = list(index_class_map_dict1.values())

        number_labels=[]

        for label in self.labels:
            number_labels.append(index_class_map_dict1[label])

        df = pd.DataFrame({'news': self.list_of_sentences, 'label': number_labels})

        list_of_items_per_class = []

        for i in range(self.num_categories):
            masked = df[df['label'] == i]
            num = len(masked)
            list_of_items_per_class.append(num)
    
        #print(len(list_of_items_per_class))
        return list_of_items_per_class
       
    """ Plot the number of news entries per category and the average number of words per category"""
    def data_plotter(self):    
        self.reset_states()
        self.get_categories()

        count_of_news_category = {}
        average_words_per_number_of_samples = []

        #Counting news items per category of news 
        for key in self.category_news.keys():
            item = self.category_news.get(key)
            number_of_news_items = len(item)
            count_of_news_category[key] = number_of_news_items
            count_of_words = 0
            for sentence in item:
                count_of_words += len(sentence.split(" "))
            average_words_per_number_of_samples.append(count_of_words/number_of_news_items)

        
        #print(average_words_per_number_of_samples)

        if not os.path.isdir('new_data_figs'):
            os.mkdir('new_data_figs')
            

        path = os.getcwd()
        folder = "new_data_figs"
        FILE = "Number_of_articles_per_category.png"

        path_save = os.path.join(path, folder)
        path_save = os.path.join(path_save, FILE)

        fig = plt.figure()
        category = count_of_news_category.keys()
        value = count_of_news_category.values()
        plt.bar(category, value)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(path_save)
        plt.show()

        path = os.getcwd()
        folder = "new_data_figs"
        FILE = "Words_per_category.png"

        path_save = os.path.join(path, folder)
        path_save = os.path.join(path_save, FILE)



        fig1 = plt.figure()
        category = count_of_news_category.keys()
        value = average_words_per_number_of_samples
        plt.bar(category, value)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(path_save)
        plt.show()

class Metrics(object):

    def __init__(self):
        pass
    
    # Method to calculate top 1 accuracy
    def accuracy(self, predicted, target):
        count = 0
        for i in range(len(predicted)):
            
            if predicted[i][0] == target[i]:
                count +=1
                
        return count/len(predicted)

    #Method to calcualte top3 accuracy
    def accuracyTop3(self, predicted, target):
        count = 0
        for i in range(len(predicted)):
            
            if predicted[i][0] == target[i] or predicted[i][1] == target[i] or predicted[i][2] == target[i]:
                count +=1
                
        return count/len(predicted)

    # Method to caclulate MRR
    def meanReciprocalRank(self,arrayOfRanks):
        _sum = 0
        for i in arrayOfRanks:
            
            if i != 0:
                _sum += 1/i
            else:
                _sum += i
                
        return _sum/len(arrayOfRanks)

""" Class to load dataset for training the model"""
class newsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: (val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class classDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: (val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

