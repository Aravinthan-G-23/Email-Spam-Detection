import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'text']
# Preprocess the data
data['text'] = data['text'].str.lower().str.replace('[^\w\s]', '')
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)# Extract features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# Predict and evaluate the model
y_pred = model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
import seaborn as sns

sns.countplot(x='label', data=data, palette='Set2')
plt.title('Distribution of Spam vs. Ham')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
spam_indices = [i for i, label in enumerate(y_pred) if label == 'spam']
print("\nDetected Spam Emails:")
for i in spam_indices[:10]:  # Show first 10 spam emails
    print(f"- {X_test.iloc[i]}")
