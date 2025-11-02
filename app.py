# app.py - Emotion detection web app with fallback predictor
from flask import Flask, render_template, request, jsonify, send_from_directory
import os, sqlite3, io, base64, re
from datetime import datetime
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static','uploads')

EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Try loading a Keras model; if unavailable, define a fallback predictor
MODEL_PATH = 'emotion_model.h5'
use_real_model = False
model = None
try:
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    use_real_model = True
    print("Loaded real Keras model.")
except Exception as e:
    print("Could not load real Keras model, using fallback predictor. Error:", e)
    # fallback predictor: simple heuristic based on image brightness
    class DummyModel:
        def predict(self, arr):
            # arr is (1,48,48,1) grayscale [0,1]
            mean = float(arr.mean())
            # map mean brightness to a pseudo-probability distribution
            # darker -> 'Sad' or 'Angry'; brighter -> 'Happy' or 'Surprise'
            probs = np.zeros((1,7))
            if mean < 0.25:
                probs[0,0] = 0.5  # Angry
                probs[0,4] = 0.5  # Sad
            elif mean < 0.45:
                probs[0,2] = 0.4  # Fear
                probs[0,4] = 0.6  # Sad
            elif mean < 0.65:
                probs[0,6] = 0.5  # Neutral
                probs[0,5] = 0.5  # Surprise
            else:
                probs[0,3] = 0.85 # Happy
                probs[0,5] = 0.15 # Surprise
            return probs
    model = DummyModel()

# Initialize DB
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, image_path TEXT, emotion TEXT, date TIMESTAMP)")
    conn.commit()
    conn.close()

init_db()

def preprocess_pil(img_pil):
    # convert to grayscale 48x48 and scale to [0,1]
    img = img_pil.convert('L').resize((48,48))
    arr = np.array(img).astype('float32')/255.0
    arr = arr.reshape((1,48,48,1))
    return arr

@app.route('/')
def index():
    return render_template('index.html', student_name='Adegbola Oluwabusolami', matric='23CD034295')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Anonymous')
    file = request.files.get('file')
    img_pil = None
    saved_path = None
    if file:
        filename = re.sub(r'[^0-9a-zA-Z._-]','_', file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        img_pil = Image.open(save_path)
        saved_path = save_path
    else:
        b64 = request.form.get('webcam')
        if b64:
            header, encoded = (b64.split(',',1) if ',' in b64 else (None,b64))
            data = base64.b64decode(encoded)
            img_pil = Image.open(io.BytesIO(data))
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], f'webcam_{ts}.png')
            img_pil.save(save_path)
            saved_path = save_path
    if img_pil is None:
        return jsonify({'error':'No image provided'}), 400
    arr = preprocess_pil(img_pil)
    preds = model.predict(arr)
    emotion = EMOTIONS[int(np.argmax(preds))]
    # store in db
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name, image_path, emotion, date) VALUES (?, ?, ?, ?)", (name, saved_path, emotion, datetime.now()))
    conn.commit()
    conn.close()
    return jsonify({'emotion': emotion, 'image_path': saved_path})

@app.route('/db_latest')
def db_latest():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT name, image_path, emotion, date FROM users ORDER BY id DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    results = []
    for r in rows:
        results.append({'name': r[0], 'image_path': r[1], 'emotion': r[2], 'date': r[3]})
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
