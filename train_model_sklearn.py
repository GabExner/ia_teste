from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
from training_data import input_texts, target_texts

# Vetorização dos textos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(input_texts)
y = target_texts

# Treinamento do modelo
model = MultinomialNB()
model.fit(X, y)

# Salvar o modelo e o vetorizar
with open("chatbot_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)