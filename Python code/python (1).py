import numpy as np
import pandas as pd

ds = pd.read_csv(r"X:\20191226-reviews.csv", delimiter = ",")
ds.drop(columns=['asin','name','date','verified','helpfulVotes'] , inplace = True)
ds['sentiment'] = ds['rating'].apply(lambda rating: +1 if rating > 3 else 0)
ds.isnull().any()
ds["title"].fillna(ds["title"].mode()[0] , inplace = True)
ds["body"].fillna(ds["body"].mode()[0] , inplace = True)
ds.isnull().any()
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data=[]
ds=ds.head(1000)
for i in range(0,1000):
    review = ds["title"][i]
    review = re.sub('[^a-zA-Z]',' ',review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
from sklearn.feature_extraction.text import CountVectorizer
import pickle
cv = CountVectorizer(max_features = 3000)
x = cv.fit_transform(data).toarray()
with open('CountVectorizer','wb')as file:
    pickle.dump(cv,file)
y = ds.iloc[:,3:4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 3000, init = "random_uniform", activation = "relu"))
model.add(Dense(units = 4500, init = "random_uniform", activation = "relu"))
model.add(Dense(units = 1,init = "random_uniform", activation = "sigmoid"))
model.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train,y_train,epochs = 3)

model.save("nlp.h5")

y_pred=model.predict(x_test)
y_pred=(y_pred >0.5)

y_p = model.predict(cv.transform(["good"]))
y_p = y_p>0.5

text =  "Hello, I have this phone and used it until I decided to buy a flip phone. I have had NO problems with the battery or new cases--it has a new fish case on "
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)

y_p1 = model.predict(cv.transform([text]))
y_p1 = y_p>0.5