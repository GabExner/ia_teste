import unittest
from app import app, db, Response

class ChatbotAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "Bem-vindo ao chatbot!")

    def test_add_response(self):
        # Limpar o banco de dados antes de adicionar uma nova resposta
        with app.app_context():
            db.session.query(Response).delete()
            db.session.commit()

        response = self.app.post('/add_response', json={
            'message': 'olá',
            'response': 'Olá, como posso ajudar?'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Resposta adicionada com sucesso!', response.data.decode())

    def test_chat(self):
        # Limpar o banco de dados antes de adicionar uma nova resposta
        with app.app_context():
            db.session.query(Response).delete()
            db.session.commit()

        self.app.post('/add_response', json={
            'message': 'olá',
            'response': 'Olá, como posso ajudar?'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        response = self.app.post('/chat', json={'message': 'olá'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Olá, como posso ajudar?', response.json['response'])

    def test_list_responses(self):
        # Limpar o banco de dados antes de adicionar uma nova resposta
        with app.app_context():
            db.session.query(Response).delete()
            db.session.commit()

        self.app.post('/add_response', json={
            'message': 'olá',
            'response': 'Olá, como posso ajudar?'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        response = self.app.get('/responses')
        self.assertEqual(response.status_code, 200)
        responses = response.get_json()
        self.assertTrue(any(r['message'] == 'olá' for r in responses))

    def test_update_response(self):
        # Limpar o banco de dados antes de adicionar uma nova resposta
        with app.app_context():
            db.session.query(Response).delete()
            db.session.commit()

        self.app.post('/add_response', json={
            'message': 'olá',
            'response': 'Olá, como posso ajudar?'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        response = self.app.put('/update_response', json={
            'message': 'olá',
            'response': 'Olá, como posso ajudar você hoje?'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Resposta atualizada com sucesso!', response.data.decode())

    def test_delete_response(self):
        # Limpar o banco de dados antes de adicionar uma nova resposta
        with app.app_context():
            db.session.query(Response).delete()
            db.session.commit()

        self.app.post('/add_response', json={
            'message': 'olá',
            'response': 'Olá, como posso ajudar?'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        response = self.app.delete('/delete_response', json={
            'message': 'olá'
        }, headers={'Authorization': 'Basic YWRtaW46cGFzc3dvcmQ='})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Resposta deletada com sucesso!', response.data.decode())

if __name__ == '__main__':
    unittest.main()