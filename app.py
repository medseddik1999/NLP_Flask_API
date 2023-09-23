#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:13:24 2023

@author: midou
"""

from flask import Flask ,request , jsonify 
from modell import model , tokenizer ,convert_pred , clean_text 
import numpy as np 
from keras.preprocessing.sequence import pad_sequences 


app=Flask(__name__) 


@app.route('/predict' , methods=['POST'])  
def predict(): 
    json_=request.json
    texot=json_['text'] 
    texot = [clean_text(item) for item in texot]
    texot =tokenizer.texts_to_sequences(texot)
    texot=pad_sequences(texot,maxlen=81,padding='post') 
    predictions = model.predict(texot)
    pred=np.argmax(predictions , axis=1)  
    pred=[convert_pred(item) for item in pred] 
    
    return jsonify({"pred":pred}) 



if __name__=='__main__': 
    app.run(port=3000 , debug= True )

