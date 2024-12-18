from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import LambdaCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stdout = sys.__stdout__  
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
CORS(app)

class PlantRecommendationCNN:
    def __init__(self):
        self.data_buah = self.load_data()
        self.model = self.create_model()
        self.train_model()

    def load_data(self):
        data = {
            "Nama Buah": [
                'Apel', 'Jeruk', 'Mangga', 'Pisang', 'Anggur', 'Semangka',
                'Melon', 'Jambu Biji', 'Durian', 'Nanas', 'Alpukat',
                'Kelengkeng', 'Sirsak', 'Rambutan', 'Ceri', 'Belimbing',
                'Stroberi', 'Lemon', 'Pepaya', 'Kiwi', 'Markisa',
                'Cempedak', 'Salak', 'Sukun', 'Manggis', 'Duku',
                'Buah Naga', 'Zaitun', 'Delima', 'Kesemek', 'Kersen',
                'Jambu Air', 'Ara (Fig)', 'Ciplukan', 'Kurma',
                'Blueberry', 'Blackberry', 'Plum', 'Mulberry',
                'Grapefruit', 'Persik (Peach)', 'Aprikot',
                'Kakao', 'Kacang Tanah', 'Cokelat', 'Tomat',
                'Cabai Merah', 'Pare'
            ],
            "Cocok": [
                80, 70, 85, 75, 90, 60, 65, 78, 88, 72,
                77, 83, 68, 76, 81, 74, 62, 71, 73, 79,
                84, 87, 78, 65, 82, 75, 68, 89, 80, 66,
                72, 78, 85, 74, 88, 63, 65, 84, 70, 76,
                79, 82, 72, 68, 89, 71, 75, 67
            ],
            "Toleransi": [
                10, 15, 8, 12, 5, 20, 18, 10, 6, 14,
                9, 7, 13, 10, 11, 12, 18, 14, 12, 9,
                8, 7, 11, 20, 10, 12, 15, 5, 9, 17,
                13, 11, 7, 12, 5, 18, 17, 10, 15, 12,
                9, 8, 14, 16, 5, 14, 13, 16
            ]
        }
        dataset = pd.DataFrame(data)
        print(f"Dataset dimuat:\n{dataset.head()}")
        return dataset

    def create_model(self):
        model = Sequential([
            Conv2D(32, (2, 2), activation='relu', input_shape=(5, 5, 1)),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
        return model

    def train_model(self):
        data = []
        labels = []

        for suhu in range(0, 51, 5):
            for kelembaban in range(0, 101, 10):
                for kecepatan_angin in range(0, 31, 5):
                    grid = np.zeros((5, 5, 1))  
                    grid[0, 0, 0] = suhu / 50  
                    grid[0, 1, 0] = kelembaban / 100  
                    grid[0, 2, 0] = kecepatan_angin / 30 
                    data.append(grid)
                    labels.append(suhu + kelembaban - kecepatan_angin)

        data = np.array(data)
        labels = np.array(labels)

        # Train the model
        custom_callback = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: print(f"Epoch {epoch + 1} dimulai"),
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch + 1} selesai"),
        )

        self.model.fit(data, labels, epochs=10, batch_size=32, verbose=0, callbacks=[custom_callback])
        print("Model trained successfully.")

    def predict_plant_suitability(self, suhu_input, kelembaban_input, kecepatan_angin_input):
        input_data = np.array([
            [suhu_input / 50, kelembaban_input / 100, kecepatan_angin_input / 30, 0, 0],
            [suhu_input / 50, kelembaban_input / 100, kecepatan_angin_input / 30, 0, 0],
            [suhu_input / 50, kelembaban_input / 100, kecepatan_angin_input / 30, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]).reshape(1, 5, 5, 1)

        fuzzy_result = self.model.predict(input_data)[0][0]
        print(f"Hasil prediksi model: {fuzzy_result}")

        hasil_kecocokan = {}
        for _, row in self.data_buah.iterrows():
            cocok = row["Cocok"]
            toleransi = row["Toleransi"]
            min_cocok = cocok - toleransi

            if fuzzy_result >= cocok:
                persentasi_kecocokan = 100
            elif fuzzy_result < min_cocok * 0.9:
                persentasi_kecocokan = 0
            else:
                persentasi_kecocokan = ((fuzzy_result - min_cocok) / toleransi) * 100

            hasil_kecocokan[row["Nama Buah"]] = persentasi_kecocokan

        return hasil_kecocokan

recommender = PlantRecommendationCNN()

@app.route('/apibuah', methods=['POST'])
def recommend():
    data = request.get_json()
    print(f"Data request diterima: {data}")

    if not data:
        return jsonify({"error": "Invalid request, JSON data required"}), 400

    try:
        suhu = float(data['suhu'])
        kelembaban = float(data['kelembaban'])
        kecepatan_angin = float(data['kecepatan_angin'])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input data: {e}"}), 400

    hasil_kecocokan = recommender.predict_plant_suitability(suhu, kelembaban, kecepatan_angin)

    rekomendasi = {plant: suitability for plant, suitability in hasil_kecocokan.items() if suitability > 0}
    rekomendasi_sorted = sorted(rekomendasi.items(), key=lambda x: x[1], reverse=True)

    if not rekomendasi_sorted:
        return jsonify({"recommendations": "Tidak ada rekomendasi tanaman saat ini."})

    return jsonify({"recommendations": rekomendasi_sorted})

if __name__ == '__main__':
    app.run(debug=True)
