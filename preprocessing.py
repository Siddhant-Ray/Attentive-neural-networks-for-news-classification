import json
import os

"""Preprocessing the JSON file to a useful form, in this case, every category of the useful part of the news
is written as a new line for every new news sentence in a file of that category"""

items = []

for item in open('RawDataset/News_Category_Dataset_v2.json', 'r'):
    items.append(item)

item_in_news_list =[]    
category_class = {}
all_categories =[]

#items = items[0:100]

new_dir = "news_data_test2"
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
    #news = item_in_news_list[1].split(": ")[1]
    news = news.replace("\"","")
    #print(news)
    file_name = new_path+"%s.txt"%category

    ### These categories are combined and dropped based on the algorithm. Comment out section for generating base dataset with no 
    ### combination of classes based on out algorithm.
    if category == "PARENTING":
        file_name = new_path+"PARENTS.txt"
    if category == "TASTE":
        file_name = new_path+"FOOD & DRINK.txt"
    if category == "HEALTHY LIVING":
        file_name = new_path+"WELLNESS.txt"
    if category == "ENVIRONMENT":
        file_name = new_path+"GREEN.txt"
    if category == "MONEY":
        file_name = new_path+"BUSINESS.txt"
    if category == "COLLEGE":
        file_name = new_path+"EDUCATION.txt"
    if category == "COMEDY":
        file_name = new_path+"ENTERTAINMENT.txt"
    if category == "IMPACT" or category == "FIFTY":
        continue
    

    ### These categories are combined initially
    if category == "THE WORLDPOST" or category == "WORLDPOST":
        file_name = new_path+"WORLD NEWS.txt"
    if category == "ARTS" or category == "CULTURE & ARTS":
        file_name = new_path+"ARTS & CULTURE.txt"
    f = open(file_name, "a")
    f.write(news+"\n")
    f.close

    if category not in all_categories:
        all_categories.append(category)


