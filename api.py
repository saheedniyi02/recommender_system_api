# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from flask import Flask,jsonify,json
import pandas as pd
import numpy as np

system=np.load("utils/cosine_similarity.npy")
indices=pd.read_csv("utils/indices.csv")["_id"]
print(system)
print(indices)

def recommend(_id, cosine_sim = system):
    #print(_id)
    recommended_texts = []
    idx = indices[indices == _id].index[0]
    #print(idx)
    #rint(indices["_id"])
    #print(list(indices))
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_2_indices = list(score_series.iloc[1:3].index)
    #print(top_2_indices)
    #print(list(indices))

    for i in top_2_indices:
        recommended_texts.append(list(indices)[i])

    return recommended_texts

print(indices.iloc[55])
print(recommend(indices.iloc[55]))

def get_recommendations(_id):
    recommended_texts=recommend(_id)
    recommendations_dict={}
    for i in range(len(recommended_texts)):
        recommendations_dict[i]=recommended_texts[i]
    return recommendations_dict


print(get_recommendations(indices.iloc[70]))


app=Flask(__name__)

@app.route("/recommendations/<_id>",methods=["GET"])
def api(_id):
    recommendations=[]
    print(_id)
    recommendations_dict=get_recommendations(_id)
    for i,_id in recommendations_dict.items():
        recommendations.append({i:_id})
    return jsonify(json.dumps(recommendations))


if __name__ == '__main__':
    app.run(debug=True)

