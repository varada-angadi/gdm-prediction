from flask import Flask, request, jsonify, session, send_from_directory
from flask_cors import CORS
import numpy as np
import pickle
import bcrypt
import sqlite3
from tensorflow import keras
import os
import pandas as pd
from datetime import datetime

model_path = os.path.join(os.path.dirname(__file__), 'gdm_model.h5')
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production-12345678'
CORS(app, supports_credentials=True)

##########################################
# FRONTEND ROUTES
##########################################

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))

@app.route('/')
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route('/index.html')
def index_page():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route('/login.html')
def login_page():
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route('/signup.html')
def signup_page():
    return send_from_directory(FRONTEND_DIR, "signup.html")

@app.route('/prediction.html')
def prediction_page():
    return send_from_directory(FRONTEND_DIR, "prediction.html")

@app.route('/result.html')
def result_page():
    return send_from_directory(FRONTEND_DIR, "result.html")

@app.route('/history.html')
def history_page():
    return send_from_directory(FRONTEND_DIR, "history.html")

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

##########################################
# MODEL LOADING
##########################################

print("Loading model...")
model = keras.models.load_model(model_path)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("Model loaded successfully!")

##########################################
# DATABASE SETUP
##########################################

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  name TEXT NOT NULL)''')
    
    # Predictions history table
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

##########################################
# HELPER FUNCTIONS
##########################################

def save_to_csv(input_data, result_data, user_email=None):
    """Save prediction data to CSV file for all users"""
    try:
        csv_file = 'all_predictions.csv'
        
        record = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'User Email': user_email if user_email else 'Anonymous',
            'Age': input_data['age'],
            'BMI': input_data['bmi'],
            'Family History': 'Yes' if input_data['family_history'] == 1 else 'No',
            'PCOS': 'Yes' if input_data['pcos'] == 1 else 'No',
            'Previous GDM': 'Yes' if input_data['previous_gdm'] == 1 else 'No',
            'Physical Activity': input_data['physical_activity'],
            'Diet Quality': input_data['diet_quality'],
            'Gestational Week': input_data['gestational_week'],
            'Weight Gain (kg)': input_data['weight_gain'],
            'Previous Pregnancies': input_data['previous_pregnancies'],
            'Blood Pressure Systolic': input_data['blood_pressure_systolic'],
            'Resting Heart Rate': input_data['resting_heart_rate'],
            'Risk Level': result_data['risk_level'],
            'Risk Percentage': result_data['risk_percentage']
        }
        
        df_new = pd.DataFrame([record])
        
        if os.path.exists(csv_file):
            df_existing = pd.read_csv(csv_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_file, index=False)
        else:
            df_new.to_csv(csv_file, index=False)
        
        print(f"✅ CSV: Saved to {csv_file}")
        return True
    except Exception as e:
        print(f"❌ CSV Error: {str(e)}")
        return False

def save_to_excel(input_data, result_data, user_email=None):
    """Save prediction data to Excel file"""
    try:
        excel_file = 'prediction_records.xlsx'
        
        record = {
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'User Email': user_email if user_email else 'Anonymous',
            'Age': input_data['age'],
            'BMI': input_data['bmi'],
            'Family History': 'Yes' if input_data['family_history'] == 1 else 'No',
            'PCOS': 'Yes' if input_data['pcos'] == 1 else 'No',
            'Previous GDM': 'Yes' if input_data['previous_gdm'] == 1 else 'No',
            'Physical Activity': input_data['physical_activity'],
            'Diet Quality': input_data['diet_quality'],
            'Gestational Week': input_data['gestational_week'],
            'Weight Gain (kg)': input_data['weight_gain'],
            'Previous Pregnancies': input_data['previous_pregnancies'],
            'Blood Pressure Systolic': input_data['blood_pressure_systolic'],
            'Resting Heart Rate': input_data['resting_heart_rate'],
            'Risk Level': result_data['risk_level'],
            'Risk Percentage': result_data['risk_percentage']
        }
        
        df_new = pd.DataFrame([record])
        
        if os.path.exists(excel_file):
            df_existing = pd.read_excel(excel_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(excel_file, index=False)
        else:
            df_new.to_excel(excel_file, index=False)
        
        print(f"✅ Excel: Saved to {excel_file}")
        return True
    except Exception as e:
        print(f"❌ Excel Error: {str(e)}")
        return False

def save_to_database(user_id, input_data, result_data):
    """Save prediction to database"""
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
        print(f"✅ DATABASE: Saved for user_id={user_id}")
        return True
    except Exception as e:
        print(f"❌ DATABASE Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_risk_level(probability):
    if probability >= 0.70:
        return "High"
    elif probability >= 0.35:
        return "Moderate"
    else:
        return "Low"

def get_top_risk_factors(input_data, prediction_proba):
    reference = {
        'age': 25, 'bmi': 22, 'family_history': 0, 'pcos': 0,
        'previous_gdm': 0, 'physical_activity': 3, 'diet_quality': 2,
        'gestational_week': 16, 'weight_gain': 8, 'previous_pregnancies': 1,
        'blood_pressure_systolic': 115, 'resting_heart_rate': 70
    }

    weights = {
        'previous_gdm': 5.0, 'bmi': 4.5, 'family_history': 4.0, 'pcos': 3.5,
        'age': 3.0, 'weight_gain': 2.5, 'physical_activity': 2.0,
        'diet_quality': 2.0, 'blood_pressure_systolic': 1.5,
        'previous_pregnancies': 1.2, 'resting_heart_rate': 0.8,
        'gestational_week': 0.5
    }

    risk_contributions = {}

    for feature in feature_names:
        value = input_data[feature]
        ref_value = reference[feature]

        if feature in ['family_history', 'pcos', 'previous_gdm']:
            contribution = weights[feature] if value == 1 else 0
        elif feature in ['physical_activity', 'diet_quality']:
            contribution = (ref_value - value) * weights[feature]
        else:
            contribution = abs(value - ref_value) * weights[feature] / ref_value

        risk_contributions[feature] = max(0, contribution)

    sorted_factors = sorted(risk_contributions.items(), key=lambda x: x[1], reverse=True)

    top_factors = []
    for feature, contribution in sorted_factors[:3]:
        if contribution > 0.1:
            factor_name = feature.replace('_', ' ').title()
            value = input_data[feature]
            message = f"{factor_name}: {value}"
            top_factors.append(message)

    return top_factors if top_factors else ["Overall health profile"]

def get_recommendations(risk_level, top_factors, input_data):
    recommendations = []

    if risk_level == "High":
        recommendations.append("Consult with your obstetrician immediately for comprehensive evaluation.")
    elif risk_level == "Moderate":
        recommendations.append("Schedule an appointment to discuss your risk factors with your healthcare provider.")
    else:
        recommendations.append("Maintain your regular prenatal care schedule and continue healthy habits.")

    if input_data['bmi'] > 25:
        recommendations.append("Focus on balanced nutrition with controlled portions and nutrient-dense foods.")

    if input_data['physical_activity'] < 2:
        recommendations.append("Increase physical activity gradually - try walking, prenatal yoga, or swimming.")

    if input_data['diet_quality'] < 2:
        recommendations.append("Improve diet quality by incorporating more whole grains, vegetables, and lean proteins.")

    if input_data['blood_pressure_systolic'] > 120:
        recommendations.append("Monitor your blood pressure regularly and limit sodium intake.")

    recommendations.append("Stay well-hydrated throughout the day (aim for 8-10 glasses of water).")
    recommendations.append("Ensure adequate rest with 7-9 hours of quality sleep each night.")
    recommendations.append("Practice stress management techniques like meditation or deep breathing exercises.")

    return recommendations[:8]

##########################################
# API ROUTES
##########################################

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        email = data['email']
        password = data['password']
        name = data['name']

        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = sqlite3.connect('users.db')
        c = conn.cursor()

        try:
            c.execute('INSERT INTO users (email, password, name) VALUES (?, ?, ?)',
                      (email, hashed, name))
            conn.commit()
            print(f"✅ User registered: {email}")
            return jsonify({'message': 'User created successfully'}), 201
        except sqlite3.IntegrityError:
            return jsonify({'error': 'Email already exists'}), 400
        finally:
            conn.close()

    except Exception as e:
        print(f"❌ Signup error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        email = data['email']
        password = data['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, password, name FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
            # Return user data to be stored in localStorage on frontend
            print(f"✅ Login successful: {email} (ID: {user[0]})")
            return jsonify({
                'message': 'Login successful',
                'user_id': user[0],
                'name': user[2],
                'email': email
            }), 200
        else:
            print(f"❌ Login failed: {email}")
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract user_id from request body (sent from frontend)
        user_id = data.get('user_id')
        print(f"\n{'='*60}")
        print(f"PREDICTION REQUEST")
        print(f"User ID from request: {user_id}")
        print(f"{'='*60}")

        input_dict = {feature: float(data[feature]) if feature not in
                      ['family_history', 'pcos', 'previous_gdm',
                       'physical_activity', 'diet_quality',
                       'gestational_week', 'previous_pregnancies',
                       'blood_pressure_systolic', 'resting_heart_rate', 'user_id']
                      else int(data[feature])
                      for feature in feature_names}

        input_array = np.array([[input_dict[f] for f in feature_names]])
        input_scaled = scaler.transform(input_array)

        prediction_proba = model.predict(input_scaled, verbose=0)[0][0]
        risk_percentage = float(prediction_proba * 100)
        risk_level = get_risk_level(prediction_proba)
        top_factors = get_top_risk_factors(input_dict, prediction_proba)
        recommendations = get_recommendations(risk_level, top_factors, input_dict)

        result_data = {
            'risk_level': risk_level,
            'risk_percentage': round(risk_percentage, 1),
            'top_factors': top_factors,
            'recommendations': recommendations
        }

        # Get user email and save to database
        user_email = None
        if user_id:
            try:
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('SELECT email FROM users WHERE id = ?', (user_id,))
                user = c.fetchone()
                if user:
                    user_email = user[0]
                    print(f"User email: {user_email}")
                conn.close()
                
                # Save to database
                save_to_database(user_id, input_dict, result_data)
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  No user_id provided - saving as Anonymous")

        # Save to CSV and Excel
        save_to_csv(input_dict, result_data, user_email)
        save_to_excel(input_dict, result_data, user_email)

        print(f"{'='*60}\n")
        return jsonify(result_data), 200

    except Exception as e:
        print(f"❌ PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/history/<int:user_id>', methods=['GET'])
def get_history(user_id):
    """Get prediction history for a specific user"""
    try:
        print(f"\n{'='*60}")
        print(f"HISTORY REQUEST for user_id={user_id}")
        print(f"{'='*60}")
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('''SELECT id, timestamp, age, bmi, family_history, pcos, previous_gdm,
                            physical_activity, diet_quality, gestational_week, weight_gain,
                            previous_pregnancies, blood_pressure_systolic, resting_heart_rate,
                            risk_level, risk_percentage
                     FROM predictions 
                     WHERE user_id = ?
                     ORDER BY timestamp DESC''', (user_id,))
        
        predictions = c.fetchall()
        conn.close()
        
        print(f"✅ Found {len(predictions)} prediction(s)")
        
        history = []
        for pred in predictions:
            history.append({
                'id': pred[0],
                'timestamp': pred[1],
                'age': pred[2],
                'bmi': pred[3],
                'family_history': pred[4],
                'pcos': pred[5],
                'previous_gdm': pred[6],
                'physical_activity': pred[7],
                'diet_quality': pred[8],
                'gestational_week': pred[9],
                'weight_gain': pred[10],
                'previous_pregnancies': pred[11],
                'blood_pressure_systolic': pred[12],
                'resting_heart_rate': pred[13],
                'risk_level': pred[14],
                'risk_percentage': pred[15]
            })
        
        print(f"{'='*60}\n")
        return jsonify({'history': history}), 200
    
    except Exception as e:
        print(f"❌ HISTORY ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    print(f"✅ User logged out")
    return jsonify({'message': 'Logged out successfully'}), 200

##########################################
# RUN APP
##########################################

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print("\n" + "="*60)
    print("🚀 MATERNAGUARD SERVER STARTING")
    print("="*60)
    print(f"Server Running on http://0.0.0.0:{port}")
    print("="*60 + "\n")

    app.run(host="0.0.0.0", port=port)