# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#import libraries
import pymongo
from pymongo import MongoClient
import pandas as pd
#import dns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import time
import pickle

#download important nltk requirements
nltk.download("wordnet")
nltk.download("punkt")


#important functions and classes
def load_data(pymongo_link):
    client=MongoClient(pymongo_link)
    collection=client.upspace.data
    cursor=collection.find()
    list_cursor=list(cursor)
    df=pd.DataFrame(list_cursor)
    return df

def clean_authors(list_of_authors):
    new_list=[]
    for author in list_of_authors:
      author=author.replace(".","").replace(",","").replace("(","").replace(")","").replace(" ","")
      author=author.lower()
      new_list.append(author)
    return new_list

def authors_to_string(authors):
     text=" ".join(authors)
     return text

def clean_abstract(abstract):
   abstract=abstract.replace("\n","").replace("\r","").replace(r'\x','').replace("'s","").replace("\'n","")
   abstract=abstract.lower()
   return abstract

lemmatizer=WordNetLemmatizer()

def clean_text(text):
    new_text=[]
    text_list=text.split(" ")
    for word in text_list:
        word_lemma=lemmatizer.lemmatize(word)
        new_text.append(word_lemma)
    new_text=" ".join(new_text)
    return new_text

def test_recommend(title, cosine_sim,indices):
    recommended_texts = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_2_indices = list(score_series.iloc[1:3].index)

    for i in top_2_indices:
        recommended_texts.append(list(indices)[i])

    return recommended_texts

vect=TfidfVectorizer(stop_words="english",min_df=2)

#Tfidf vectorizer for tags
vect_tags=TfidfVectorizer(stop_words="english",min_df=2)


def load_model():
    start=time.time()
    df=load_data("mongodb+srv://Tabby:1234@cluster0.c2f7n.mongodb.net/test")
    df_uncleaned=df.copy()
    print(f"There are {df.shape[0]} documents in the database")
    df["authors"]=df["authors"].apply(clean_authors)
    df["abstract"]=df["abstract"].apply(clean_abstract)
    df["Text_without_authors"]=df["title"]+" "+df["abstract"]
    df["Text"]=df["Text_without_authors"]+df["authors"].apply(authors_to_string)
    df["Text_without_authors"]=df["Text_without_authors"].apply(clean_text)
    df["Text"]=df["Text"].apply(clean_text)
    vect.fit(df["Text"])
    print(f"There are {len(vect.vocabulary_)} important words in the documents used for recommendation")
    Word_bag=vect.transform(df["Text"])
    print("Word bag gotten!!!")
    system=cosine_similarity(Word_bag,Word_bag)
    print("cosine similarity matrix gotten!!!")
    np.save("utils/cosine_similarity",system)
    print(f"The shape of the cosine similarity matrix {system.shape}")
    indices_id_info = df[['_id','Text_without_authors']]
    indices_id_info.to_csv("utils/indices_texts.csv")
    
    #fit the Tag Vectorizer
    vect_tags.fit(df["Text_without_authors"])
    pickle.dump(vect_tags, open("utils/vect_tags.pkl", 'wb'))
    print(f"There are {len(vect_tags.vocabulary_)} important words in the vocabulary used for getting the tags")
