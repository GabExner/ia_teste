import json
import logging
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

@app.route('/', methods=['GET'])
def home():
    app.logger.info('Home endpoint accessed')
    return "Bem-vindo ao chatbot!"

@app.route('/chat', methods=['GET'])
def chat():
    return render_template('chat.html')

@app.route('/chat_message', methods=['POST'])
def chat_message():
    user_message = request.json.get('message', '').lower()
    tokens = word_tokenize(user_message, language='portuguese')
    filtered_tokens = [word for word in tokens if word not in stopwords.words('portuguese')]
    filtered_message = ' '.join(filtered_tokens)
    response_entry = Response.query.filter_by(message=filtered_message).first()
    if response_entry:
        response = response_entry.response
    else:
        response = model.predict([filtered_message])[0]
    app.logger.info(f'Chat message received: {user_message}')
    return jsonify({"response": response})

@app.route('/add_response', methods=['POST'])
@auth.login_required
def add_response():
    new_message = request.json.get('message', '').lower()
    tokens = word_tokenize(new_message, language='portuguese')
    filtered_tokens = [word for word in tokens if word not in stopwords.words('portuguese')]
    filtered_message = ' '.join(filtered_tokens)
    new_response = request.json.get('response', '')
    response_entry = Response(message=filtered_message, response=new_response)
    db.session.add(response_entry)
    db.session.commit()
    with app.app_context():
        global model
        model = train_model()
    app.logger.info(f'Add response endpoint accessed with message: {new_message}')
    return jsonify({"message": "Resposta adicionada com sucesso!"})

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
@auth.login_required
def train():
    with app.app_context():
        global model
        model = train_model()
    app.logger.info('Model retrained')
    return jsonify({"message": "Modelo re-treinado com sucesso!"})

@app.route('/spec')
def spec():
    app.logger.info('Spec endpoint accessed')
    return jsonify(swagger(app))

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