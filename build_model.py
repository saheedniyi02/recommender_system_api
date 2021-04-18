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
    special_char=[".","(",")"," "]
    for author in list_of_authors:
        for char in special_char:
            author=author.replace(char,"")
        author=author.lower()
        new_list.append(author)
    return new_list\

def clean_abstract(abstract):
   special_char=["\n","\r"]
   for char in special_char:
       abstract=abstract.replace(char,"")
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

def recommend(title, cosine_sim):
    recommended_texts = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_2_indices = list(score_series.iloc[1:3].index)

    for i in top_2_indices:
        recommended_texts.append(list(indices)[i])

    return recommended_texts

vect=TfidfVectorizer(stop_words="english")

if __name__ == '__main__':
    df=load_data("mongodb+srv://Tabby:1234@cluster0.c2f7n.mongodb.net/test")
    df_uncleaned=df.copy()
    print(f"There are {df.shape[0]} documents in the database")
    df["authors"]=df["authors"].apply(clean_authors)
    df["abstract"]=df["abstract"].apply(clean_abstract)
    df["Text"]=" "+df["title"]+" "+df["abstract"]#+df["authors"].apply(authors_to_string)
    df["Text"]=df["Text"].apply(clean_text)
    vect.fit(df["Text"])
    print(f"There are {len(vect.vocabulary_)} important words in the documents used for recommendation")
    test_index=int(input("Input a document index for which we will output the recommended similar documents to it\n"))
    Word_bag=vect.transform(df["Text"])
    print("Word bag gotten!!!")
    system=cosine_similarity(Word_bag,Word_bag)
    print("cosine similarity matrix gotten!!!")
    np.save("utils/cosine_similarity",system)
    print(f"The shape of the cosine similarity matrix {system.shape}")
    indices_id = pd.Series(df['_id'])
    indices_id.to_csv("utils/indices.csv")

    #This is just to test the model using the prices
    indices= pd.Series(df['title'])
    Title=indices.iloc[test_index]
    print(Title)
    recommended_texts=recommend(Title,system)
    print(recommended_texts)
    df_info=df_uncleaned[df_uncleaned["title"].isin(recommended_texts)][["_id","authors","title","abstract"]]
    print(df_info)
    print("\n\n")
    print(df_info.authors)


