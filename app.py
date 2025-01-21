# from flask import Flask, request, jsonify, render_template
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import matplotlib.pyplot as plt
# import io
# from PIL import Image


# # Inisialisasi aplikasi Flask
# app = Flask(__name__)

# @app.route("/")
# def root():
#     return render_template("index.html")



# # Folder untuk menyimpan sementara file gambar yang diunggah
# UPLOAD_FOLDER = './uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Muat model yang telah disimpan
# MODEL_PATH = 'my_model.h5'  # Ganti dengan path model Anda
# model = load_model(MODEL_PATH)

# # Daftar kelas sesuai dengan model Anda
# class_names = {0: 'Bukan Daun', 1: 'Healty', 2: 'Penyakit Cescospora', 3: 'Penyakit Kuning', 4: 'Penyakit Mozaik', 5: 'Penyakit Tungau'}

# def preprocess_image(image_path):
#     """
#     Preproses gambar untuk dimasukkan ke model.
#     """
#     img = load_img(image_path, target_size=(128, 128))  # Ubah ukuran gambar
#     img_array = img_to_array(img) / 255.0  # Normalisasi piksel
#     img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
#     return img_array

# def predict_image(image_path):
#     """
#     Fungsi untuk memprediksi kelas gambar.
#     """
#     # Preproses gambar
#     img_array = preprocess_image(image_path)

#     # Prediksi dengan model
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions[0])

#     # Dapatkan nama kelas
#     predicted_class_name = class_names.get(predicted_class, 'Unknown')

#     # Hasil prediksi
#     return predicted_class_name, predictions[0].tolist()


# @app.route('/classify', methods=['POST'])
# def classify():
#     """
#     Endpoint untuk mengklasifikasikan gambar.
#     """
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400

#     # Simpan file ke folder uploads
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)

#     # Prediksi gambar
#     try:
#         predicted_class_name, prediction_probabilities = predict_image(file_path)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

#     # Hapus file setelah diproses
#     os.remove(file_path)

#     # Return hasil prediksi
#     return jsonify({
#         'predicted_class': predicted_class_name,
#         'prediction_probabilities': prediction_probabilities
#     })


# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import os
import onnx
import onnxruntime as ort
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

@app.route("/")
def root():
    return render_template("index.html")

# Folder untuk menyimpan sementara file gambar yang diunggah
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model ONNX
MODEL_PATH = 'my_model.onnx'  # Ganti dengan path model ONNX Anda
onnx_model = onnx.load(MODEL_PATH)
onnx.checker.check_model(onnx_model)
session = ort.InferenceSession(MODEL_PATH)

# Daftar kelas sesuai dengan model Anda
class_names = {0: 'Bukan Daun', 1: 'Healty', 2: 'Penyakit Cescospora', 3: 'Penyakit Kuning', 4: 'Penyakit Mozaik', 5: 'Penyakit Tungau'}

def preprocess_image(image_path):
    """
    Preproses gambar untuk dimasukkan ke model.
    """
    img = load_img(image_path, target_size=(128, 128))  # Ubah ukuran gambar
    img_array = img_to_array(img) / 255.0  # Normalisasi piksel
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    return img_array

def predict_image(image_path):
    """
    Fungsi untuk memprediksi kelas gambar.
    """
    # Preproses gambar
    img_array = preprocess_image(image_path)

    # Prediksi dengan model ONNX
    inputs = {session.get_inputs()[0].name: img_array.astype(np.float32)}
    predictions = session.run(None, inputs)
    predicted_class = np.argmax(predictions[0])

    # Dapatkan nama kelas
    predicted_class_name = class_names.get(predicted_class, 'Unknown')

    # Hasil prediksi
    return predicted_class_name, predictions[0].tolist()

@app.route('/classify', methods=['POST'])
def classify():
    """
    Endpoint untuk mengklasifikasikan gambar.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Simpan file ke folder uploads
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Prediksi gambar
    try:
        predicted_class_name, prediction_probabilities = predict_image(file_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Hapus file setelah diproses
    os.remove(file_path)

    # Return hasil prediksi
    return jsonify({
        'predicted_class': predicted_class_name,
        'prediction_probabilities': prediction_probabilities
    })


if __name__ == '_main_':
    app.run(debug=True)