#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os #app name

app = Flask(__name__,template_folder='template') 

#load the saved model
def load_model():
    return pickle.load(open("fish_model.pkl", "rb")) 

#home page
@app.route('/')
def home():
    return render_template("index.html") 


@app.route('/predict',methods=['POST'])
def predict(): #For rendering results on HTML GUI        
    
    labels = ['Bream','Roach','Whitefish','Parkki','Perch']
    features = [float(x)  for x in request.form.values()] 
    values = [np.array(features)] 
    
    model = load_model() 
    
    prediction = model.predict(values)  
    
    return render_template("index.html", output="The Fish is {}".format(prediction)) 

if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(port=port,debug=True,use_reloader=False)

