# Fake News Detection using Machine Learning (Streamlit App)
This project is a **Fake News Detection System** built using Machine Learning and Natural Language Processing.  
Users can input any news, and the system predicts whether it is **REAL** or **FAKE**.

# This project includes:
A full Machine Learning pipeline
NLP preprocessing using NLTK
TF-IDF Vectorizer
PassiveAggressiveClassifier model
Streamlit web application interface


# Features
Real-time Fake News Prediction  
NLP Preprocessing (tokenization, stopwords removal, lemmatization)  
TF-IDF vectorization for text processing  
Passive Aggressive Classifier Training  
Confusion Matrix & Accuracy Score  
Streamlit-based Web UI  
Supports CSV dataset  


# Project Structure
project/
│
├── app.py           (Main Streamlit Application)
├── first.py         (Model Training Script)
├── model.pkl        (Trained ML Model)
├── vector.pkl       (TF-IDF Vectorizer)
├── tfidf.pkl        (Optional vectorizer file)
├── news.csv         (Project Documentation)

--- 

```

# **How the System Works**

#1 **Model Training (first.py)**  
Loads dataset (`news.csv`)  
Cleans text (removes symbols, tokenizes, lemmatizes)  
Converts text → TF-IDF vectors  
Trains **PassiveAggressiveClassifier** 
Saves model + vectorizer as:
`model.pkl`
`vector.pkl`


```

#2 **Streamlit App (app.py)**  
Loads saved ML model  
Provides a simple UI for entering news  
Shows prediction instantly  
Displays:
  - Accuracy  
  - Confusion Matrix  
  - About Page

```


# Install Required Libraries

pip install -r requirements.txt


# 3. Run Streamlit App

streamlit run app.py


# app will open in the browser:

http://localhost:8501


---

# Dataset Requirements

The dataset `news.csv` must contain:

| Column | Description |
|--------|-------------|
| text   | News content |
| label  | REAL or FAKE |

Example:

| text                       | label |
|---------------------------|-------|
| "PM announces new..."     | REAL  |
| "Aliens attacked Delhi"   | FAKE  |

---

# Technologies Used

# **Languages**
 Python

# **Libraries**
# Machine Learning & NLP:
. scikit-learn  
. numpy  
. pandas  
. nltk  

# Visualization:
. matplotlib  
. seaborn  

#### Web App:
- Streamlit  

---

## Model Performance

The trained model provides:

. Accuracy Score  
. Classification Report  
. Confusion Matrix  

These are visible inside the Streamlit **Model Performance** section.



---


#  Author

**Deepa Fartiyal**  
Fake News Detection ML Project – 2025



# Thank You!

If you like this project, don’t forget to ⭐ star the repository!

 

