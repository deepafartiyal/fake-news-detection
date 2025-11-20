import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

# -------------------------
# CONFIG & ASSETS
# -------------------------
LOGO_PATH = r"C:\Users\deepa\Pictures\gettyimages-2157128431-612x612.jpg"
MODEL_PATH = "model.pkl"
VECT_PATH = "vector.pkl"
DATA_PATH = "news.csv"

# -------------------------
# NLTK SAFE DOWNLOAD
# -------------------------
for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except:
        nltk.download(pkg, quiet=True)

# -------------------------
# Load model + vectorizer
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_vector():
    model = pickle.load(open(MODEL_PATH, "rb"))
    tfidf = pickle.load(open(VECT_PATH, "rb"))
    return model, tfidf

model, tfidf = load_model_and_vector()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(DATA_PATH)
    except:
        return pd.DataFrame(columns=["text", "label"])

df = load_dataset()

# -------------------------
# UI STYLING
# -------------------------
st.markdown("""
<style>
.stApp { 
    background: linear-gradient(to bottom right, #eef2f7, #f7fbff);
}
.header {
    background: linear-gradient(90deg,#0ea678,#0c8d59);
    padding: 25px;
    border-radius: 15px;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.big-num {
    font-size: 30px;
    font-weight: 700;
    color: #0ea678;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR NAVIGATION
# -------------------------
st.sidebar.image(LOGO_PATH, width=130)
st.sidebar.title("📰 Fake News Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Check News", "Dataset", "Model Performance", "About"]
)

# -------------------------
# Functions - Plots
# -------------------------
def plot_class_distribution(df):
    fig, ax = plt.subplots()
    df["label"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    return fig

def plot_confusion(cm, labels):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    return fig

# -------------------------
# PAGE: Home
# -------------------------
if page == "Home":
    st.markdown("<div class='header'><h2>Fake News Detection Dashboard</h2></div>", unsafe_allow_html=True)

    st.markdown("### Dashboard Statistics")

    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='card'><div class='big-num'>{len(df)}</div>Total Articles</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='big-num'>{df['label'].nunique()}</div>Labels</div>", unsafe_allow_html=True)

    if not df.empty:
        sample = df.sample(1)["text"].values[0][:150] + "..."
        col3.markdown(f"<div class='card'><b>Random Sample</b><br>{sample}</div>", unsafe_allow_html=True)
    else:
        col3.markdown("<div class='card'>No data loaded</div>", unsafe_allow_html=True)

    st.markdown("### Class Distribution")
    if not df.empty:
        st.pyplot(plot_class_distribution(df))
    else:
        st.info("Dataset not found!")

# -------------------------
# PAGE: Check News
# -------------------------
elif page == "Check News":
    st.markdown("### Check News Article")

    text = st.text_area("Paste news text here:", height=250)

    if st.button("Check"):
        if text.strip() == "":
            st.warning("Enter some text!")

        else:
            cleaned = preprocess_text(text)
            vec = tfidf.transform([cleaned])
            pred = model.predict(vec)[0]

            st.markdown("---")

            if pred.lower() == "fake":
                st.error("❌ FAKE NEWS DETECTED")
            else:
                st.success("✔ REAL NEWS")

# -------------------------
# PAGE: Dataset
# -------------------------
elif page == "Dataset":
    st.markdown("### Dataset Preview")

    if df.empty:
        st.warning("Dataset not found!")
    else:
        st.dataframe(df.head(200))
        st.markdown("### Class Distribution")
        st.pyplot(plot_class_distribution(df))

# -------------------------
# PAGE: Model Performance
# -------------------------
elif page == "Model Performance":
    st.markdown("### Model Performance")

    if df.empty:
        st.warning("Dataset missing!")
    else:
        X = df["text"].astype(str).apply(preprocess_text)
        y = df["label"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_test_vec = tfidf.transform(X_test)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)

        st.metric("Accuracy", f"{acc:.2%}")

        st.markdown("### Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)))

        st.markdown("### Confusion Matrix")
        labels = sorted(df["label"].unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        st.pyplot(plot_confusion(cm, labels))

# -------------------------
# PAGE: About
# -------------------------
elif page == "About":
    st.markdown("### About This Project")
    st.write("""
    - This is a machine learning Fake News Detection Dashboard.
    - Built using Streamlit, TF-IDF Vectorizer, and PAC classifier.
    """)

    st.image(LOGO_PATH, width=250)
