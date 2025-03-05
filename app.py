import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

# 1. Muat model Keras dari file .h5
# Pastikan path 'model.h5' sesuai lokasi model Anda.
model = tf.keras.models.load_model('final_model.h5')
print("Model loaded successfully")

# Jika model punya 4 kelas (baik, rusak ringan, dsb.)
CLASS_NAMES = ['baik', 'rusak_ringan', 'rusak_sedang', 'rusak_berat']

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "Server is running."

@app.route('/predict-base64', methods=['POST'])
@cross_origin(origin='*')
def predict_base64():
    """
    Menerima JSON: { "image": "data:image/jpeg;base64,..." }
    Mengembalikan: { "predictedClass": "<kelas>" }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400

        # 1. Ambil base64 (misal "data:image/jpeg;base64,/9j/4AAQ...")
        base64_image = data['image']

        # Jika ada prefix "data:image/xxx;base64,", kita buang prefix-nya
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        # 2. Decode base64 -> bytes
        image_bytes = base64.b64decode(base64_image)

        # 3. Buka dengan PIL
        image = Image.open(io.BytesIO(image_bytes))

        # 4. Preprocessing (resize, normalize, dsb.)
        #    Sesuaikan dengan ukuran input model Anda.
        image = image.resize((150, 150))
        image_array = np.array(image, dtype='float32') / 255.0
        # shape awal: (150,150,3)
        # tambahkan dimensi batch -> (1,150,150,3)
        image_array = np.expand_dims(image_array, axis=0)

        # 5. Prediksi
        prediction = model.predict(image_array)  # shape: (1, 4)
        # Argmax
        max_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[max_index]

        # 6. Return JSON
        return jsonify({"predictedClass": predicted_class})

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "Prediction error"}), 500

if __name__ == '__main__':
    # Jalankan Flask di port 8080 (default)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
