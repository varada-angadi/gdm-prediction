from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import bcrypt
import sqlite3
from tensorflow import keras
import os
import pandas as pd
from datetime import datetime

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "gdm_model.h5")

app = Flask(__name__, static_folder=FRONTEND_DIR)
app.secret_key = 'your-secret-key-change-in-production-12345678'
CORS(app, supports_credentials=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)

with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'feature_names.pkl'), 'rb') as f:
    feature_names = pickle.load(f)

print("Model loaded successfully!")

# ----------------------------
# DATABASE INITIALIZATION
# ----------------------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  name TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER NOT NULL,
                  timestamp TEXT NOT NULL,
                  age REAL,
                  bmi REAL,
                  family_history INTEGER,
                  pcos INTEGER,
                  previous_gdm INTEGER,
                  physical_activity INTEGER,
                  diet_quality INTEGER,
                  gestational_week INTEGER,
                  weight_gain REAL,
                  previous_pregnancies INTEGER,
                  blood_pressure_systolic INTEGER,
                  resting_heart_rate INTEGER,
                  risk_level TEXT,
                  risk_percentage REAL,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# FRONTEND ROUTES
# ----------------------------
@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_risk_level(probability):
    if probability >= 0.7:
        return "High"
    elif probability >= 0.35:
        return "Moderate"
    else:
        return "Low"

def save_to_database(user_id, input_data, result_data):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                     (user_id, timestamp, age, bmi, family_history, pcos, previous_gdm,
                      physical_activity, diet_quality, gestational_week, weight_gain,
                      previous_pregnancies, blood_pressure_systolic, resting_heart_rate,
                      risk_level, risk_percentage)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (user_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   input_data['age'], input_data['bmi'], input_data['family_history'],
                   input_data['pcos'], input_data['previous_gdm'],
                   input_data['physical_activity'], input_data['diet_quality'],
                   input_data['gestational_week'], input_data['weight_gain'],
                   input_data['previous_pregnancies'], input_data['blood_pressure_systolic'],
                   input_data['resting_heart_rate'], result_data['risk_level'],
                   result_data['risk_percentage']))
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB save error:", e)

# ----------------------------
# API ROUTES
# ----------------------------
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        email, password, name = data['email'], data['password'], data['name']
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (email, password, name) VALUES (?, ?, ?)',
                  (email, hashed, name))
        conn.commit()
        conn.close()
        return jsonify({'message': 'User created successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already exists'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email, password = data['email'], data['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, password, name FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
            return jsonify({'message': 'Login successful', 'user_id': user[0], 'name': user[2], 'email': email}), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_id = data.get('user_id')

        input_dict = {feature: float(data[feature]) if feature not in ['family_history','pcos','previous_gdm','physical_activity','diet_quality','gestational_week','previous_pregnancies','blood_pressure_systolic','resting_heart_rate','user_id'] else int(data[feature]) for feature in feature_names}

        input_array = np.array([[input_dict[f] for f in feature_names]])
        input_scaled = scaler.transform(input_array)

        prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
        risk_level = get_risk_level(prediction_proba)
        risk_percentage = round(float(prediction_proba)*100, 1)

        result_data = {'risk_level': risk_level, 'risk_percentage': risk_percentage}

        if user_id:
            save_to_database(user_id, input_dict, result_data)

        return jsonify(result_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
