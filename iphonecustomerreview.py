#ARYA GOSAVI
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df = pd.read_excel('/content/drive/MyDrive/iphone.xlsx')
display(df.head())
display(df.info())
import os
print(os.listdir('/content/drive/MyDrive/'))
from google.colab import drive
drive.mount('/content/drive')
import os
print(os.listdir('/content/drive/MyDrive/'))
import pandas as pd
file_path = '/content/drive/MyDrive/iphone.xlsx'
df = pd.read_excel(file_path)
display(df.head())
import os
print(os.listdir('/content/drive/MyDrive/'))
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
file_path = '/content/drive/MyDrive/iphone.xlsx'
df = pd.read_excel(file_path)
display(df.head())
import re
import nltk
from nltk.corpus import stopwords
# Download stopwords if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
# Identify the review column
review_column = 'reviewDescription'
# Handle missing values
df[review_column].fillna('', inplace=True)
# Convert to lowercase
df[review_column] = df[review_column].str.lower()
# Remove special characters, punctuation, and numbers
df[review_column] = df[review_column].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
# Remove stop words
stop_words = set(stopwords.words('english'))
df[review_column] = df[review_column].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
display(df[[review_column]].head())
from sklearn.model_selection import train_test_split
# Define features (X) and target (y)
X = df['reviewDescription']
# Create the target variable 'sentiment' based on 'ratingScore'
# Assuming ratingScore > 3 is positive, ratingScore < 3 is negative, and ratingScore == 3 is neutral
# For simplicity in this binary classification example, let's consider > 3 positive (1) and <= 3 negative (0)
df['sentiment'] = df['ratingScore'].apply(lambda score: 1 if score > 3 else 0)
y = df['sentiment']
# Split the data into training and testing sets
# Using a test size of 0.2 and a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
# Instantiate TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit on training data and transform both training and test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Print the shapes
print("Shape of X_train_tfidf:", X_train_tfidf.shape)
print("Shape of X_test_tfidf:", X_test_tfidf.shape)
from sklearn.linear_model import LogisticRegression
# Instantiate a LogisticRegression model
model = LogisticRegression()
# Fit the model to the training data
model.fit(X_train_tfidf, y_train)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Make predictions on the testing set
y_pred = model.predict(X_test_tfidf)
# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Print the metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
