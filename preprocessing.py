import json
import os

"""Preprocessing the JSON file to a useful form, in this case, every category of the useful part of the news
is written as a new line for every new news sentence in a file of that category"""

items = []

for item in open('News_Category_Dataset_v2.json', 'r'):
    items.append(item)

item_in_news_list =[]    
category_class = {}
all_categories =[]

#items = items[0:100]

new_dir = "news_data"
parent_dir = "/scratch/sidray/Attentive-recurrent-neural-networks-for-categorizing-and-generating-news"

path = os.path.join(parent_dir, new_dir)
os.mkdir(path)

new_path = parent_dir+"/"+new_dir+"/"
#print(new_path)

for item in items:
    item_in_news_list = item.split(", \"")
    category = item_in_news_list[0].split(": ")[1]
    category = category.replace("\"", "")
    #print(category)
    news = item_in_news_list[1].split(": ")[1] + ". " + item_in_news_list[4].split(": ")[1]
    news = news.replace("\"","")
    #print(news)
    file_name = new_path+"%s.txt"%category
    f = open(file_name, "a")
    f.write(news+"\n")
    f.close

    if category not in all_categories:
        all_categories.append(category)



