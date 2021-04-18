# -*- coding: utf-8 -*-
from __future__ import unicode_literals
<<<<<<< HEAD
from flask import Flask, jsonify, url_for, redirect
import pandas as pd, numpy as np
from build_model import load_model
=======
from flask import Flask, jsonify, redirect
from flask.json import load
import pandas as pd, numpy as np, os
from build_model import load_model
from dotenv import load_dotenv
from pymongo import MongoClient

client = MongoClient(os.getenv('MONGODB_CONNECTION_STRING'))

load_dotenv()
>>>>>>> origin

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
<<<<<<< HEAD
        {"0": recommended_texts_id[0]},
        {"1": recommended_texts_id[1]}
    ]

=======
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
>>>>>>> origin

app=Flask(__name__)

@app.route('/')
def home():
    return redirect('https://documenter.getpostman.com/view/11853513/TzJsfJSw')

@app.route("/recommendations/<_id>",methods=["GET"])
def api(_id):
    try:
<<<<<<< HEAD
        recommendations=get_recommendations(_id)
        return jsonify({
            "data":recommendations,
            "message":f"recommendations successfully gotten for id:{_id}"
        }),200
    except :
        return jsonify({
            "message":f"recommendations couldn't be gotten for id:{_id}"
        }),404

@app.route("/update_recommendations")
=======
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

@app.route("/update_recommendations", methods=['POST'])
>>>>>>> origin
def update_system():
    load_model()
    return "",204

if __name__ == '__main__':
    app.run()