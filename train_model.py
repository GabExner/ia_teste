import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Exemplo de dados de treinamento
X_train = ["mensagem de exemplo 1", "mensagem de exemplo 2", "mensagem de exemplo 3"]
y_train = ["label1", "label2", "label3"]

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Salvar o vectorizer e o modelo
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)