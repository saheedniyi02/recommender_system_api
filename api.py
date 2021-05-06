# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask, jsonify, redirect
from flask.json import load

import pandas as pd, numpy as np, os
from build_model import load_model
from dotenv import load_dotenv
from pymongo import MongoClient
from flask_cors import CORS
import pickle
client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))

load_dotenv()
system=np.load("utils/cosine_similarity.npy")
data=pd.read_csv("utils/indices_texts.csv")
indices=data["_id"]
text_without_authors=data[["_id","Text_without_authors"]].set_index("_id")
vect_tags = pickle.load(open('utils/vect_tags.pkl','rb'))


def recommend(_id, cosine_sim = system):
    idx = indices[indices == _id].index[0]
    score_series = pd.Series(
        cosine_sim[idx]
    ).sort_values(ascending = False)
    top_2_indices = list(score_series.iloc[1:3].index)

    return [list(indices)[i] for i in top_2_indices]



def get_recommendations(_id):
    recommended_texts_id=recommend(_id)
    return [
        recommended_texts_id[0],
        recommended_texts_id[1]
    ]

def get_col_from_id(_id):
    try:
        from bson.objectid import ObjectId
        data =  dict(client.upspace.data.find_one({'_id': ObjectId(_id)}))
        data['_id'] = str(data['_id'])
        return data
    except Exception as e: 
        print(str(e))
        return
    
#Functions to get word tags
def get_feature_names():
    return vect_tags.get_feature_names()

def get_text(_id):
    return text_without_authors.loc[_id]["Text_without_authors"]

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=5):
    """get the feature names and tf-idf score of top n items"""

    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    # word index and corresponding tf-idf score
    for idx, score in sorted_items:

        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]

    return results

app=Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return redirect('https://documenter.getpostman.com/view/11853513/TzJsfJSw')

@app.route("/recommendations/<_id>",methods=["GET"])
def api(_id):
    try:
        publication_data = get_col_from_id(_id)
        if not publication_data:
            return {
                "message": "No publication found for that ID"
            }, 404
        publication_data['recommendations'] = [ 
           get_col_from_id(recommended_id) 
               for recommended_id in get_recommendations(_id)
        ]
        return jsonify({
            "data":publication_data,
            "message":f"recommendations successfully gotten for id"
        }),200
    except Exception as e :
        return jsonify({
            "message":f"recommendations couldn't be gotten for id:{_id}"
        }),500

@app.route("/update_recommendations", methods=['GET'])
def update_system():
    load_model()
    return "",204

@app.route("/get_tags/<_id>",methods=["GET"])
def api_tags(_id):
    try:
        tf_idf_vector=vect_tags.transform([get_text(_id)])
        sorted_items=sort_coo(tf_idf_vector.tocoo())
        feature_names=get_feature_names()
        keywords=extract_topn_from_vector(feature_names,sorted_items,5)
        return jsonify({"keywords":list(keywords.keys()),
            "message":f"recommendations gotten successfully for id:{_id}"}),200
    except:
        return jsonify({"message":"keywords could not be gotten for id:{_id}"}),404


if __name__ == '__main__':
    app.run()
