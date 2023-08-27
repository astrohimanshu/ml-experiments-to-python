#!/usr/bin/env python
# coding: utf-8

# # Score Data with a Ridge Regression Model Trained on the Diabetes Dataset

# This notebook loads the model trained in the Diabetes Ridge Regression Training notebook, prepares the data, and scores the data.

import json
import numpy
from azureml.core.model import Model
import joblib


def init():
    global model
     # load the model from file into a global object
    model_path = Model.get_model_path(model_name="sklearn_regression_model.pkl")
    model = joblib.load(model_path)

def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)

    return {"result": result.tolist()}

#Load Model
init()

#Prepare Data And Predict
raw_data = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
request_header = {}
prediction = run(raw_data, request_header)
print("Test result: ", prediction)

