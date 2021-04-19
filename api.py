# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask, jsonify, redirect
from flask.json import load
import pandas as pd, numpy as np, os
from build_model import load_model
from dotenv import load_dotenv
from pymongo import MongoClient

client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))

load_dotenv()

system=np.load("utils/cosine_similarity.npy")
indices=pd.read_csv("utils/indices.csv")["_id"]


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

app=Flask(__name__)

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

if __name__ == '__main__':
    app.run()
