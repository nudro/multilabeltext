#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 07:07:19 2018

@author: 559048
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle

#answers
answers = pd.read_csv('pythonquestions/Answers.csv', encoding = "ISO-8859-1")
answers.rename(columns={'Body': 'answers'}, inplace=True)
answers.fillna(value='none', inplace=True)

#questions
questions = pd.read_csv('pythonquestions/Questions.csv', encoding = "ISO-8859-1")
questions.rename(columns={'Body': 'question'}, inplace=True)
questions.fillna(value='none', inplace=True)

#tags
tags = pd.read_csv('pythonquestions/Tags.csv')
tags.fillna(value='none', inplace=True) #remove all the NaN's otherwise you get bad MLB errors
tags.loc[(tags['Tag'] =='none')]

print('The first five records for each dataset...')
print(answers.head())
print(questions.head())
print(tags.head())

print('The number of records for each dataset...')
print(len(questions))
print(len(answers)) #naturally there are more answers than questions, multiple answers per question
print(len(tags))


def make_qa(questions, answers):
    print('Making first dataframe, merging questions and answers...')
    #So we will rename the Id in the questions dataframe to ParentId in order to merge on the same identifier with the answers dataframe.
    questions.rename(columns={'Id': 'ParentId'}, inplace=True)
    q_and_a = questions.merge(answers, on='ParentId', how='left')
    
    #You see that one question (ParentId) can have many answers following up to it. At this point, let's clean up our data a little.
    q_and_a2 = q_and_a[['ParentId', 'Title', 'question', 'answers', 'Score_x', 'Score_y']].copy()
    main = q_and_a2.groupby('ParentId').agg(lambda x: x.tolist())
    
    #Since the Title and question are the same each time, let's remove all but the first index in each record.
    main = main.reset_index()
    Title = main['Title'].apply(lambda x: x[:1])
    Question = main['question'].apply(lambda x: x[:1])
    final = pd.concat([main['ParentId'], Title, Question, main['answers'], main['Score_x'], main['Score_y']], axis=1)
    print(final.head())
    return final

#default number = 50
def reduce_tags(tags, number):
    print('Reducing number of labels to only', number, 'labels...')
    top_minus_python = number+1
    (tags['Tag'].value_counts())[1:top_minus_python].plot(kind='barh', figsize=(8, 16)) #the first 50 minus the tag `python`
    top_tags = (tags['Tag'].value_counts())[1:top_minus_python]
    top_tags = (top_tags.reset_index())['index'].tolist() #reset the index, use the index only becuase we just want tag names
    print(top_tags)
    #remove everything else else
    #let's make a copy of 'tags'

    reduced_tags = tags.copy()

    reduced_tags['min_tag'] = np.where(reduced_tags['Tag'].isin(top_tags), reduced_tags['Tag'], "other")

    print(reduced_tags.head(10))
    
    reduced_tags.drop('Tag', axis=1, inplace=True)
    reduced_tags = reduced_tags.loc[(reduced_tags['min_tag'] != 'other')]
    
    t_r = reduced_tags.groupby('Id')['min_tag'].apply(list)
    t_r = t_r.reset_index()
    t_r.rename(columns={'Id': 'ParentId', 'min_tag':'Tag'}, inplace=True)
    return t_r


def merge_tags(t_r, final):
    print('Merging tags with the first dataframe, creating another dataframe...')
    #there are multiple tags for each Id, so you need to group by
    final2 = t_r.merge(final, on='ParentId', how='right')
    final2.dropna(subset = ['Tag'], inplace=True)

    #fill in missing values
    final2.fillna(value='0', inplace=True)
    print(final2.shape)
    
    #let's merge the Title, question, and answers together in that order

    final2['Combined_Text'] = final2['Title'] + final2['question'] + final2['answers']
    #create a new df
    final3 = final2[['ParentId', 'Combined_Text', 'Score_x', 'Score_y', 'Tag']].copy()
    return final3

def binarizer(final3):
    print('Making multilabels...')
    mlb = MultiLabelBinarizer()
    labels_binarized = mlb.fit_transform(final3['Tag'])
    print(labels_binarized.shape)
    print(mlb.classes_)
    print(len(mlb.classes_))
    classes = mlb.classes_
    outfile = open('/Users/559048/Documents/UVA/DataPalooza/mlb_classes','wb')
    pickle.dump(classes, outfile)
    outfile.close()
    print('Saved array of multilabels to file...')
    return labels_binarized

def clean_text(final3):
    #remove HTML

    final3['Combined_Text'] = final3['Combined_Text'].astype(str)

    print('Removing HTML tags and cleaning up text, this will take up to 10 minutes...')
    final3['clean'] = [BeautifulSoup(text).get_text() for text in final3['Combined_Text']]
    final3['clean'] = final3['clean'].str.replace(',', '')
    final3['clean'] = final3['clean'].str.replace('[', '')
    final3['clean'] = final3['clean'].str.replace(']', '')
    final3['clean'] = final3['clean'].str.replace('\\', '')
    final4 = final3[['ParentId', 'clean', 'Tag']].copy()
    
    #send to list for word2vec
    text = final3['clean'].tolist()
    print('The final dataframe is ready')
    return text, final4

def tokenizer(text):
    print('Tokenizing cleaned text for word2vec texts')
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = nltk.corpus.stopwords.words('english')
    
    text_ready = []

    # loop through document list
    for i in text:
    
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
    
        text_ready.append(stopped_tokens)
        
    return text_ready


"""
Now you'll have a few ingredients for the model: 
    
    'data' which is the data for training your Word2Vec model
    
    'y' which are your 'y' values from the binarized labels
    
    final3_df which has the feature 'clean', which is your 'X' value


"""
#ran the below on 10 labels


final_df = make_qa(questions, answers)

t_r_df = reduce_tags(tags, 10)

final3_df = merge_tags(t_r_df, final_df)

y = binarizer(final3_df) #array
joblib.dump(y, '/Users/559048/Documents/UVA/DataPalooza/10_labels/labels_binarized.joblib') 

text_cln, final4 = clean_text(final3_df)
final4.to_csv('/Users/559048/Documents/UVA/DataPalooza/10_labels/final4.csv')

data = tokenizer(text_cln) #array
outfile = open('/Users/559048/Documents/UVA/DataPalooza/10_labels/data_for_w2v','wb')
pickle.dump(data, outfile)
outfile.close()

print('Done')
    
    
    
    
    
    
    
    