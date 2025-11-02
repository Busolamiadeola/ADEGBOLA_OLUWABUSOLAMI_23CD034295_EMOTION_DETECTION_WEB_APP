Emotion Detection Web App - FINAL PACKAGE
Student: Adegbola Oluwabusolami
Matric: 23CD034295
Folder: ADEGBOLA_OLUWABUSOLAMI_23CD034295_EMOTION_DETECTION_WEB_APP

This version includes:
- Modern UI with webcam (auto-start) + file upload
- Pre-included emotion_model.h5 (small) and fallback predictor so the app runs even if TensorFlow won't install on the host
- SQLite database 'database.db' with sample entries and images under static/uploads
- Instructions below for running locally and deployment to Render

Run locally:
1. python -m venv venv
2. source venv/bin/activate   (or venv\Scripts\activate on Windows)
3. pip install -r requirements.txt
4. python app.py
5. Open http://127.0.0.1:5000

Deploy to Render (short):
- Create GitHub repo, push this folder, then create a new Web Service on Render connecting the repo.
- Set Start Command: gunicorn app:app --bind 0.0.0.0:$PORT
- If TensorFlow installation fails on Render, the app still runs using a fallback predictor, but for best accuracy host a trained model elsewhere or use Git LFS.
