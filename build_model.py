# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#import libraries
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import WordNetLemmatizer
import nltk, time, numpy as np, pandas as pd, os
from dotenv import load_dotenv

load_dotenv()

#download important nltk requirements
nltk.download("wordnet")
nltk.download("punkt")


#important functions and classes
def load_data(pymongo_link):
    client=MongoClient(pymongo_link)
    list_cursor=list(client.upspace.data.find())
    return pd.DataFrame(list_cursor)

def clean_authors(list_of_authors):
    new_list=[]
    for author in list_of_authors:
      author=author.replace(".","").replace(",","").replace("(","").replace(")","").replace(" ","")
      author=author.lower()
      new_list.append(author)
    return new_list

def clean_abstract(abstract):
   abstract=abstract.replace("\n","").replace("\r","")
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
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_2_indices = list(score_series.iloc[1:3].index)

    return [list(indices)[i] for i in top_2_indices]

vect=TfidfVectorizer(stop_words="english")


def load_model():
    start=time.time()
    df=load_data(os.getenv('MONGODB_CONNECTION_STRING'))
    df_uncleaned=df.copy()
    print(f"There are {df.shape[0]} documents in the database")
    df["authors"]=df["authors"].apply(clean_authors)
    df["abstract"]=df["abstract"].apply(clean_abstract)
    df["Text"]=""+df["title"]+""+df["abstract"]+df["authors"].apply(lambda authors: " ".join(authors))
    df["Text"]=df["Text"].apply(clean_text)
    vect.fit(df["Text"])
    print(f"There are {len(vect.vocabulary_)} important words in the documents used for recommendation")
    print(df)
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
    test_index=35#which ranges from 0 to the number of documents in the database-1. Just for testing the model
    print(indices)
    Title=indices.iloc[test_index]
    print(Title)
    recommended_texts=test_recommend(Title,system,indices)
    print(recommended_texts)
    df_info=df_uncleaned[df_uncleaned["title"].isin(recommended_texts)][["_id","authors","title","abstract"]]
    print(df_info)
    print("\n\n")
    print(df_info.authors)
    print("\n")
    end=time.time()
    print(f"Runtime for the program is {end-start}")



