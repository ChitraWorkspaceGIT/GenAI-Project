# FAKE NEWS DETECTION PROJECT STEP BY STEP
__1. PREREQUISITES__

Setup IDE Environment

Open visual studio code

Go to Extensions and install:

Python & Jupyter

__Create Python Environment__

Open the terminal and run these one by one:
```
#python --version                  # Check Python version
#python -m venv venv                # Create a virtual environment
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#venv\Scripts\Activate         # Activate the environment
```
__2. Download Dataset__
In terminal
```
#pip install kaggle
#kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
```
(if Kaggle CLI fails)

Go to this URL: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Download the .zip file manually.

Extract it in your project folder(.kaggle), e.g. fake-and-real-news-dataset/data/

You should now see Fake.csv and True.csv in VS Code.

__3. Install Python Libraries__
In the terminal, install the required libraries:

(And the path C:\Users\chint.kaggle)
```
#pip install pandas scikit-learn streamlit joblib matplotlib seaborn
#pip list
```
Create and Run the Jupyter Notebook

Inside fake-and-real-news-dataset/data, create a new file: app.ipynb

__4 Importing Libraries & Preprocessing data__
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
```
__Load data__

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")
fake.head()
true.head()
Label the data

fake["label"] = 0
true["label"] = 1
fake.shape, true.shape

fake_manual_testing = fake.tail(10)
for i in range(23480,23470,-1):
    fake.drop([i], axis = 0, inplace = True)
    
    
true_manual_testing = true.tail(10)
for i in range(21416,21406,-1):
    true.drop([i], axis = 0, inplace = True)
fake.shape, true.shape

fake_manual_testing["label"] = 0
true_manual_testing["label"] = 1
fake_manual_testing.head(10)
true_manual_testing.head(10)
manual_testing = pd.concat([fake_manual_testing,true_manual_testing], axis = 0)
manual_testing.to_csv("manual_testing.csv")
Merge both datasets

data = pd.concat([fake, true], axis =0 )
data.head(10)
data.columns

Drop unnecessary columns

data = data.drop(["title", "subject", "date"], axis = 1)
data.isnull().sum()
Random Shuffle

data = data.sample(frac = 1)
data.head()
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)
data.columns
data.head()

Clean the text

def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]' , " ", text)
    text = re.sub("\\W" , " " , text)
    text = re.sub("https? : ://\S+|www\.\S+", "" , text)
    text = re.sub("<.*?>+" , " ", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub("\n" , " " , text)
    text = re.sub ("\w*\d\w*" , " " , text)
    return text
data["text"] = data["text"].apply(clean_text)
5. Vectorization and Model Training
Features and labels & Splitting the data

x= data["text"]
y= data["label"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
TF-IDF Vectorization(Convert text to vectors)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)
5.1 Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(xv_train, y_train)
Evaluate Model and Save It

prediction = lr.predict(xv_test)
lr.score(xv_test, y_test)
print(classification_report(y_test, prediction))

5.2 Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
prediction_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
print(classification_report(y_test, prediction_dt))

5.3 Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
prediction_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)
print(classification_report(y_test, prediction_gbc))

5.4 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
prediction_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)
print(classification_report(y_test, prediction_rfc))

6. Testing the Model
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(clean_text)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)

    prediction_lr = lr.predict(new_xv_test)
    prediction_dt = DT.predict(new_xv_test)
    prediction_gbc = GBC.predict(new_xv_test)
    prediction_rfc = RFC.predict(new_xv_test)

    return print("\n\nlr Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_label(prediction_lr[0]),
        output_label(prediction_dt[0]),
        output_label(prediction_gbc[0]),
        output_label(prediction_rfc[0])
    ))
news = "paste any lines of fake.csv or true.csv"
manual_testing(news)
news = "paste any lines of fake.csv or true.csv"
manual_testing(news)
6.1 Save model and vectorizer
import joblib

joblib.dump(vectorizer, "vectorizer.jb")
joblib.dump(lr, "lr_model.jb")
7. Create the Streamlit App
Inside fake-and-real-news-dataset/data, create a new file: app.py

import streamlit as st
import joblib 

vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

st.title("Fake news detection with LLM Classification")
st.write("Enter the news to check if it is real or no")

news_input = st.text_area("News Article:","")
if st.button("Predict"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        if prediction[0] == 1:
            st.success("This news is real")
        else:
            st.error("This news is fake")
    else:
        st.warning("Please enter a news article to check.")
7.1 Run the App
In the terminal:

#cd fake-and-real-news-dataset/data
#streamlit run app.py
Then go to the browser link it gives you (usually http://localhost:8501).

7.2 Test the App
Copy a news article from Fake.csv → paste into the app → It should show "FAKE"

Copy a news article from True.csv → paste → It should show "REAL"

8. Final Deployment: Streamlit + GitHub
Activate Virtual Environment

#python -m venv venv              
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#venv\Scripts\Activate         
Generate the requirements.txt file

pip install requirements.txt
pip freeze > requirements.txt
8.1 Create GitHub Repo
Go to https://github.com

Create new repo: GEN-AI-PROJECT

In terminal

git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fakenews-detection.git
git push -u origin main
streamlit run app.py

8.2 Deploy on Streamlit Cloud
Click on Deploy-->Paste the app.py url-->Mention name of the website-->Deploy it

Once deployed, you’ll get a public link like:

https://fakenews-detection.streamlit.app/

Test the app using samples from Fake.csv and True.csv.
