import requests
import json
import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# URL do endpoint para adicionar respostas
url = 'http://127.0.0.1:5000/add_response'

# Cabeçalhos da requisição
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='  # Base64 encoding de 'admin:password'
}

# Dados de treinamento (exemplo)
training_data = [
    {"message": "como você está?", "response": "Estou bem, obrigado!"},
    {"message": "qual é o seu nome?", "response": "Eu sou um chatbot."},
    {"message": "o que você faz?", "response": "Eu respondo perguntas."},
    {"message": "qual é a capital da França?", "response": "A capital da França é Paris."},
    {"message": "qual é a capital do Brasil?", "response": "A capital do Brasil é Brasília."},
    {"message": "qual é a capital do Japão?", "response": "A capital do Japão é Tóquio."},
    # Adicione mais dados de treinamento aqui
]

def load_training_data():
    X_train = [data['message'] for data in training_data]
    y_train = [data['response'] for data in training_data]
    return X_train, y_train

def preprocess_data(X_train):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    return X_train_tfidf, vectorizer

def train_model(X_train_tfidf, y_train):
    model = SGDClassifier()
    model.fit(X_train_tfidf, y_train)
    return model

def save_model(model, vectorizer, model_path, vectorizer_path):
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

while True:
    # Enviar dados de treinamento para o endpoint
    for data in training_data:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print(response.json())
        else:
            print(f"Erro ao enviar dados: {response.status_code} - {response.text}")

        # Aguardar um pouco entre as requisições para evitar sobrecarregar o servidor
        time.sleep(1)

    # Executar treinamento
    X_train, y_train = load_training_data()
    X_train_tfidf, vectorizer = preprocess_data(X_train)
    model = train_model(X_train_tfidf, y_train)
    save_model(model, vectorizer, model_path, vectorizer_path)

    print("Modelo treinado e salvo com sucesso!")

    # Aguardar um pouco antes de reiniciar o loop
    time.sleep(60)