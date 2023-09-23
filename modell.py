import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer 
from keras.models import load_model 
from keras.preprocessing.text import Tokenizer
import json  
from datetime import datetime, timedelta
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences 


model=load_model('model_fanacial_setiment.h5') 




def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    # Stem each word
    words = [stemmer.stem(word) for word in words]

    # Join the words back into a sentence
    cleaned_text = ' '.join(words)

    return cleaned_text



def convert_pred(pred): 
  if pred==0: 
    t=1 
  elif pred==1: 
    t=-1 
  else: 
    t=0 
  return(t)



model=load_model('model_fanacial_setiment.h5') 


def tok():
    with open("keras_tokenizer.json", "r") as json_file:
          tokenizer_json = json_file.read()
    tokenizer = tokenizer_from_json(tokenizer_json) 
    return(tokenizer)


tokenizer=tok()  





  







#source=pd.read_csv('source.csv')   


#liso=[]
#for index,row in source.iterrows(): 
#   try: 
#    tak=news201('stock' , source=source['id'][index])  
#    liso.append(row) 
#   except: 
#     pass 
    




