# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask,jsonify,json
import pandas as pd
import numpy as np

system=np.load("utils/cosine_similarity.npy")
indices=pd.read_csv("utils/indices.csv")["_id"]


def recommend(_id, cosine_sim = system):
    recommended_texts = []
    idx = indices[indices == _id].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_2_indices = list(score_series.iloc[1:3].index)


    for i in top_2_indices:
        recommended_texts.append(list(indices)[i])

    return recommended_texts



def get_recommendations(_id):
    recommended_texts_id=recommend(_id)
    recommendations_list_of_dict=[]
    dict_first={"0":recommended_texts_id[0]}
    dict_second={"1":recommended_texts_id[1]}
    recommendations_list_of_dict.append(dict_first)
    recommendations_list_of_dict.append(dict_second)
    return recommendations_list_of_dict





app=Flask(__name__)

@app.route("/recommendations/<_id>",methods=["GET"])
def api(_id):
    print(_id)
    recommendations=get_recommendations(_id)
    return jsonify({"data":recommendations,"message":f"recommendations successfully gotten for {_id}"})


if __name__ == '__main__':
    app.run(debug=False)

