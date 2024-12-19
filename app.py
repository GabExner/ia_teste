import json
import logging
import os
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_httpauth import HTTPBasicAuth
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask_swagger import swagger
from flask_swagger_ui import get_swaggerui_blueprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import numpy as np
from training_data import input_texts, target_texts
import pickle

# Baixar os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///responses.db'
db = SQLAlchemy(app)

auth = HTTPBasicAuth()

users = {
    "admin": "password"
}

@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

class Response(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(200), unique=True, nullable=False)
    response = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()
    # Adicionar mensagens de exemplo ao banco de dados
    if Response.query.count() == 0:
        examples = [
            {"message": "olá", "response": "Olá, como posso ajudar?"},
            {"message": "oi", "response": "Oi, tudo bem?"},
            {"message": "tchau", "response": "Tchau, até logo!"}
        ]
        for example in examples:
            response_entry = Response(message=example["message"], response=example["response"])
            db.session.add(response_entry)
        db.session.commit()

# Configurar logs
logging.basicConfig(level=logging.INFO)

# Treinar um modelo de classificação de texto
def train_model():
    responses = Response.query.all()
    messages = [r.message for r in responses]
    responses = [r.response for r in responses]
    if not messages or not responses:
        raise ValueError("O conjunto de dados de treinamento está vazio.")
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(messages, responses)
    return model

with app.app_context():
    model = train_model()


# Pré-processamento dos dados
input_characters = sorted(list(set("".join(input_texts))))
target_characters = sorted(list(set("".join(target_texts))))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


# Função para decodificar a sequência
def decode_sequence(input_seq):
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""
    output_tokens = np.zeros((1, max_decoder_seq_length, num_decoder_tokens))
    while not stop_condition:
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        h, c = np.zeros((1, num_decoder_tokens)), np.zeros((1, num_decoder_tokens))
        states_value = [h, c]

    return decoded_sentence

# Carregar o modelo e o vetorizar
model_path = "model.pkl"
vectorizer_path = "vectorizer.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
else:
    app.logger.error(f'Model file not found: {model_path}')
    model = MultinomialNB()  # or handle accordingly

if os.path.exists(vectorizer_path):
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
else:
    app.logger.error(f'Vectorizer file not found: {vectorizer_path}')
    vectorizer = TfidfVectorizer()  # or handle accordingly

@app.route('/', methods=['GET'])
def home():
    app.logger.info('Home endpoint accessed')
    return "Bem-vindo ao chatbot!"

@app.route('/chat', methods=['GET'])
def chat():
    return render_template('chat.html')

@app.route('/chat_message', methods=['POST'])
def chat_message():
    user_message = request.json.get("message", "").lower()
    X = vectorizer.transform([user_message])
    response = model.predict(X)[0]
    app.logger.info(f'Chat message received: {user_message}')
    return jsonify({"response": response})

@app.route('/add_response', methods=['POST'])
@auth.login_required
def add_response():
    try:
        data = request.get_json()
        if not data or 'message' not in data or 'response' not in data:
            raise ValueError("Invalid input data")
        
        message = data['message']
        response = data['response']
        
        # Aqui você pode adicionar a lógica para salvar a mensagem e a resposta
        # Por exemplo, salvar em um banco de dados ou atualizar um modelo
        
        return jsonify({"message": "Resposta atualizada com sucesso!"}), 200
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({"error": "Failed to decode JSON object", "message": str(e)}), 400

@app.route('/responses', methods=['GET'])
def list_responses():
    responses = Response.query.all()
    app.logger.info('List responses endpoint accessed')
    return jsonify([{"message": r.message, "response": r.response} for r in responses])

@app.route('/update_response', methods=['PUT'])
@auth.login_required
def update_response():
    message = request.json.get('message', '').lower()
    tokens = word_tokenize(message, language='portuguese')
    filtered_tokens = [word for word in tokens if word not in stopwords.words('portuguese')]
    filtered_message = ' '.join(filtered_tokens)
    new_response = request.json.get('response', '')
    response_entry = Response.query.filter_by(message=filtered_message).first()
    if response_entry:
        response_entry.response = new_response
        db.session.commit()
        with app.app_context():
            global model
            model = train_model()
        app.logger.info(f'Update response endpoint accessed with message: {message}')
        return jsonify({"message": "Resposta atualizada com sucesso!"})
    else:
        return jsonify({"message": "Mensagem não encontrada."}), 404

@app.route('/delete_response', methods=['DELETE'])
@auth.login_required
def delete_response():
    message = request.json.get('message', '').lower()
    tokens = word_tokenize(message, language='portuguese')
    filtered_tokens = [word for word in tokens if word not in stopwords.words('portuguese')]
    filtered_message = ' '.join(filtered_tokens)
    response_entry = Response.query.filter_by(message=filtered_message).first()
    if response_entry:
        db.session.delete(response_entry)
        db.session.commit()
        with app.app_context():
            global model
            model = train_model()
        app.logger.info(f'Delete response endpoint accessed with message: {message}')
        return jsonify({"message": "Resposta deletada com sucesso!"})
    else:
        return jsonify({"message": "Mensagem não encontrada."}), 404

@app.route('/train', methods=['POST'])
def train():
    new_input_texts = request.json.get("input_texts", [])
    new_target_texts = request.json.get("target_texts", [])

    if not new_input_texts or not new_target_texts:
        return jsonify({"error": "Invalid input"}), 400

    # Carregar o vetorizar existente
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # Vetorizar os novos textos usando o vetorizar existente
    X_new = vectorizer.transform(new_input_texts)
    y_new = new_target_texts

    # Carregar as classes existentes do modelo
    if hasattr(model, 'classes_'):
        classes = model.classes_
    else:
        classes = np.unique(y_new)

    # Treinar o modelo com os novos dados
    model.partial_fit(X_new, y_new, classes=classes)

    # Salvar o modelo atualizado e o vetorizar
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    return jsonify({"message": "Model updated successfully"})

@app.route('/your_endpoint', methods=['POST'])
@auth.login_required
def your_endpoint():
    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")
        if data is None:
            raise ValueError("No JSON data received")
    except (ValueError, json.JSONDecodeError) as e:
        app.logger.error(f"Error decoding JSON: {e}")
        return jsonify({"error": "Failed to decode JSON object", "message": str(e)}), 400

    # Process the valid JSON data

    return jsonify({"success": True})

@app.route('/spec')
def spec():
    app.logger.info('Spec endpoint accessed')
    return jsonify(swagger(app))

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            raise ValueError("Invalid input data")
        
        message = data['message']
        message_tfidf = vectorizer.transform([message])
        
        # Ensure the number of features matches the model's expectation
        if message_tfidf.shape[1] != model.n_features_in_:
            raise ValueError(f"Input data has {message_tfidf.shape[1]} features, but the model expects {model.n_features_in_} features.")
        
        prediction = model.predict(message_tfidf)
        return jsonify({"prediction": prediction.tolist()})
    except ValueError as e:
        app.logger.error(f"ValueError: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "An unexpected error occurred", "message": str(e)}), 500

# Vetorizar os textos
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(input_texts)
y = target_texts

# Treinar o modelo
model = MultinomialNB()
model.fit(X, y)

# Salvar o vetorizador e o modelo
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

SWAGGER_URL = '/swagger'
API_URL = '/spec'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Chatbot API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

if __name__ == "__main__":
    app.run(debug=True)