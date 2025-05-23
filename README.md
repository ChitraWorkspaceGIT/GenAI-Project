# FAKE NEWS DETECTION PROJECT STEP BY STEP
The project aims to develop a machine-learning model capable of identifying and classifying any news article as fake or not.

Dataset has two files: Fake.csv (23502 fake news article)  & True.csv (21417 true news article)

Dataset columns: Title, Text, Subject, Date

__Steps to follow__

*DataLoading and Preprocessing>>>Data Splitting>>>Training and Testing Data>>>Feature Extraction>>>Training the Classifier>>>Model Evaluation>>>Classifications

__Dependencies__

__Pandas:__ Pandas is a python library used for datasets to analyze, clean, explore, manipulate the data & used to prepare data for ML.

__Seaborn:__ Creates statistical data visualizations on top of matplotlib for exploring and understanding the data well with pandas.

__Matplotlib:__ It is a python library used for creating static, animated, interactive visualization like charts, graphs, enabling data analysis, exploration, presentation.

__Joblib:__ Used for saving and loading trained ML models. It is a python library used for serialization of the large data.

__Streamlit:__ Open souce python library. Used for building and sharing web apps for Datascience and ML Projects. 

__Scikit-learn:__ Open source python library(sk-learn) provides numerous algorithms for building various types of ML models.We can classify, train and preprocess the data.

1. TfidfVectorizer – Converts text into numerical vectors(feature extraction)

2. train_test_split – Splits your data for training and testing

3. Algorithms to classify: LogisticRegression, DecisionTreeClassifier, Gradient Boosting Classifier, RandomForestClassifier.

__*Logistic Regression:__ is a supervised ML model & linear model used for binary classifaction tasks(yes/no, true/false, fake/real)

__*Decisiontree Classifier:__ Splitting the data into branches, tree like structure to interpret and visualize.

__*Gradient boosting classifier:__ ensembling learning model combines the predictions of multiple weak model(decision trees) to make a strong prediction.

__*RandomForestClassifier:__ ensembling learning model uses multiple decision trees to classify the data more accurately.

__LLM EXPLAINER:__ Uses a large language model (LLM) like Mistral or GPT to generate human-readable explanations for machine learning predictions. like why a news article is labeled as "fake" or "real".

### __1. PREREQUISITES__

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
### __2. Download Dataset__

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

### __3. Install Python Libraries__

In the terminal, install the required libraries:

(And the path C:\Users\chint.kaggle)
```
#pip install pandas scikit-learn streamlit joblib matplotlib seaborn
#pip list
```
Create and Run the Jupyter Notebook

Inside fake-and-real-news-dataset/data, create a new file: app.ipynb

### __4 Importing Libraries & Preprocessing data__
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
```
data_fake=pd.read_csv('Fake.csv')
data_true=pd.read_csv('True.csv')

```
```data_fake.head()```
```data_true.head()```

__Label the data__
```
data_fake["label"] = 0
data_true["label"] = 1
```
```data_fake.shape, data_true.shape```
```
data_fake_manual_testing = fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis = 0, inplace = True)
    
data_true_manual_testing = true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i], axis = 0, inplace = True)
```
```data_fake.shape, data_true.shape```
```
data_fake_manual_testing["label"] = 0
data_true_manual_testing["label"] = 1
```
```data_fake_manual_testing.head(10)```
```data_true_manual_testing.head(10)```
```
manual_testing = pd.concat([fake_manual_testing,true_manual_testing], axis = 0)
manual_testing.to_csv("manual_testing.csv")
```
__Merge both datasets__
```
data_merge=pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)
```
```data_merge.columns```

__Drop unnecessary columns__
```
data=data_merge.drop(['title','subject','date'], axis = 1)
data.isnull().sum()
```
__Random Shuffle__
```
data = data.sample(frac = 1)
data.head()
```
```
data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)
```
```
data.columns
```
```data.head()```

__Preprocessing text(cleaning the text)__
```
def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>+', '', text) 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'\w*\d\w*', '', text)  
    text = re.sub(r'[^\w\s\?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```
```
data['text'] = data['text'].apply(clean_text)
```
### 5. Vectorization and Model Training

__Features and labels & Splitting the data__

Defining dependent and independent variable
```
x = data["text"]
y = data["label"]
#Splitting the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
```
__TF-IDF Vectorization(Convert text to vectors)__
```
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)
```
__5.1 Logistic Regression__
```
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(xv_train, y_train)
```
__Evaluate Model and Save It__
```
prediction = lr.predict(xv_test)
lr.score(xv_test, y_test)
```
```print(classification_report(y_test, prediction))```

__5.2 Decision Tree Classification__
```
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
```
```
prediction_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
```
```print(classification_report(y_test, prediction_dt))```

__5.3 Gradient Boosting Classifier__
```
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
```
```
prediction_gbc = GBC.predict(xv_test)
GBC.score(xv_test, y_test)
```
```print(classification_report(y_test, prediction_gbc))```

__5.4 Random Forest Classifier__
```
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
```
```
prediction_rfc = RFC.predict(xv_test)
RFC.score(xv_test, y_test)
```
```print(classification_report(y_test, prediction_rfc))```

### 6. Testing the Model
```
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

    pred_LR = lr.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_label(pred_LR[0]),
        output_label(pred_DT[0]),
        output_label(pred_GBC[0]),
        output_label(pred_RFC[0])
    ))
```
```
news = "paste any lines of fake.csv or true.csv"
manual_testing(news)
```
```
news = "paste any lines of fake.csv or true.csv"
manual_testing(news)
```
__6.1 Save Model and Vectorizer__
```
import joblib

joblib.dump(vectorizer, "vectorizer.jb")
joblib.dump(lr, "lr_model.jb")
```
### 7. LLM Explainer

In the terminal
```pip install requests```

Go to togetherai in chrome --->generate the api key

In the jupyter notebook

```TOGETHER_API_KEY = "paste the api key generated"```
article text: the news article you want to explain

prediction_label: model's prediction (fake/real)
```
import requests
import json
def get_explanation(article_text, prediction_label):
    prompt = f"""
Explain why this article is labeled as {prediction_label}:

Article:
\"\"\"{article_text}\"\"\"

Look for signs like clickbait, emotional words, or bias.
"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}", 
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1", 
        "prompt": prompt, 
        "max_tokens": 300,  
        "temperature": 0.7,  
        "top_p": 0.9, 
    }
    response = requests.post("https://api.together.xyz/inference", headers=headers, json=data)
    if response.status_code == 200:
        return response.json()['output']['choices'][0]['text']  
    else:
        print("Oops! Something went wrong.")
        print("Error code:", response.status_code)
        print("Details:", response.text)
        return None
```
```
article_text = "Virginia officials postpone lottery drawing to decide tied statehouse election"
prediction_label = "FAKE"

explanation = get_explanation(article_text, prediction_label)

print("Explanation:\n")
print(explanation)
```
### 8. Integrating SHaP

__Load your saved vectorizer and model__
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")
```
```
sample_text = ["This is Disturbing, \"Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters, and the very dishonest fake news media. The former reality show star had just one job to do and he couldn't do it. As our Country rapidly grows stronger and smarter, I want to wish all of my friends, supporters, enemies, haters, and even the very dishonest Fake News Media, a Happy and Healthy New Year, President Angry Pants tweeted."]

X_sample = vectorizer.transform(sample_text)
```
__Create the SHAP explainer using the trained model and transformed training data &  Plot the SHAP summary plot__
```
import shap
explainer = shap.Explainer(lr, xv_train)  # 'lr' is your Logistic Regression model

shap_values = explainer(xv_test[:10])  # Use the transformed test data

shap.summary_plot(shap_values, feature_names=vectorizer.get_feature_names_out())
```

### 9. Create the Streamlit App

Inside fake-and-real-news-dataset/data, create a new file: app.py
```
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
```
__9.1 Run the App__

In the terminal:
```
#cd fake-and-real-news-dataset/data
#streamlit run app.py
```
Then go to the browser link it gives you (usually http://localhost:8501).

__9.2 Test the App__

Copy a news article from Fake.csv → paste into the app → It should show "FAKE"

Copy a news article from True.csv → paste → It should show "REAL"

### 10. Final Deployment: Streamlit + GitHub

Activate Virtual Environment
```
#python -m venv venv              
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
#venv\Scripts\Activate         
```
__Generate the requirements.txt file__
```
pip install requirements.txt
pip freeze > requirements.txt
```
__10.1 Create GitHub Repo__

Go to https://github.com

Create new repo: GEN-AI-PROJECT

In terminal
```
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/GenAI-Project.git
git push -u origin main
```
```
streamlit run app.py
```
__10.2 Deploy on Streamlit Cloud__

Click on Deploy-->Paste the app.py url-->Mention name of the website-->Deploy it

Once deployed, you’ll get a public link like:

https://fakenews-detection.streamlit.app/

Test the app using samples from Fake.csv and True.csv.
