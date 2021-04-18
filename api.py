# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask, jsonify, url_for, redirect
import pandas as pd, numpy as np
from build_model import load_model

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
        {"0": recommended_texts_id[0]},
        {"1": recommended_texts_id[1]}
    ]


app=Flask(__name__)

@app.route('/')
def home():
    return redirect('https://documenter.getpostman.com/view/11853513/TzJsfJSw')

@app.route("/recommendations/<_id>",methods=["GET"])
def api(_id):
    try:
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
def update_system():
    load_model()
    return "",204

if __name__ == '__main__':
    app.run()