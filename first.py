import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------
# NLTK downloads (if not already downloaded)
# ---------------------------------------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("📌 Loading dataset...")
data = pd.read_csv("news.csv")

# Make sure data has required columns
if 'text' not in data.columns or 'label' not in data.columns:
    raise ValueError("Dataset must have 'text' and 'label' columns")

print("\nDataset Head:")
print(data.head())

# ---------------------------------------------------------
# 2. CLEAN DATA
# ---------------------------------------------------------
print("\n🧹 Cleaning data...")
data = data.dropna().reset_index(drop=True)

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(preprocess_text)

# ---------------------------------------------------------
# 3. SPLIT DATA
# ---------------------------------------------------------
X = data['clean_text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ---------------------------------------------------------
# 4. TF-IDF VECTOR
# ---------------------------------------------------------
print("\n⚡ TF-IDF vectorization...")
tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ---------------------------------------------------------
# 5. TRAIN MODEL
# ---------------------------------------------------------
print("\n⚙ Training PassiveAggressiveClassifier...")
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------
# 6. CROSS-VALIDATION (optional)
# ---------------------------------------------------------
print("\n🔄 Performing 5-fold cross-validation...")
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("Cross-validation Accuracy: {:.2f}% ± {:.2f}%".format(cv_scores.mean() * 100, cv_scores.std() * 100))

# ---------------------------------------------------------
# 7. TEST MODEL
# ---------------------------------------------------------
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("\n🎯 Model Accuracy on Test Set: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------
# 8. SAVE MODEL AND TF-IDF VECTOR
# ---------------------------------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vector.pkl", "wb"))

print("\n✅ Model saved as model.pkl")
print("✅ TF-IDF vector saved as vector.pkl")

# ---------------------------------------------------------
# 9. MANUAL TEST FUNCTION
# ---------------------------------------------------------
def detect_fake_news(news):
    clean = preprocess_text(news)
    vectorized = tfidf.transform([clean])
    return model.predict(vectorized)[0]

# Example
test_news = "Government launches free laptop scheme for students"
print("\n🔍 Example Test:")
print("Input:", test_news)
print("Prediction:", detect_fake_news(test_news))