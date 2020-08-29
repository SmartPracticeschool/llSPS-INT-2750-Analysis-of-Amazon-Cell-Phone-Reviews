from keras.models import load_model
import tensorflow as tf
import os
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template,url_for
import pickle
import re
import nltk#natural language tool kit
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla=load_model('nlp.h5',compile=False) 
cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('base.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        topic = request.form['ms']
        review=re.sub("[^a-zA-Z]"," ",topic)
        review=review.lower()
        review=review.split()    
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        topic=" ".join(review)
        topic=cv.transform([topic])
        with graph.as_default():
            y_pred = cla.predict(topic)
        if(y_pred>0.5):
            topic = "POSITIVE"
        elif(y_pred<0.5):
            topic = "NEGATIVE"
        return render_template('base.html',label = topic)
       
if __name__ == '__main__':
    app.run(debug = False, threaded = True)

