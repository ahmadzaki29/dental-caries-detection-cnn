# ===== IMPORTS & LIBRARY =====
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import os
import cv2
import base64
from datetime import datetime
from collections import Counter

# ===== KONFIGURASI FLASK =====
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')

# Memastikan folder upload ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===== PEMUATAN MODEL CNN =====
# Muat model yang sudah dilatih
model_path = 'dataset_gigi_cnn_tl_real.keras'

# Definisikan custom objects untuk memuat model
custom_objects = {
    'MobileNetV3Small': MobileNetV3Small
}

# Verifikasi file model
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    exit(1)
else:
    print(f"Model file found: {model_path}")
    print(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

# Muat model dengan custom objects
model = load_model(model_path, custom_objects=custom_objects)

# Label kelas untuk model 
model_class_labels = ['bukan-gigi', 'caries', 'no-caries']

# Verifikasi model berhasil dimuat
print("Model loaded successfully")
print(f"Model output shape: {model.output_shape}")
print(f"Number of classes: {model.output_shape[-1]}")
print(f"Class labels: {model_class_labels}")

# ===== FUNGSI TEST MODEL =====
# Test model dengan dummy data untuk memverifikasi output
def test_model_prediction():
    """Test model dengan dummy data untuk memverifikasi output"""
    try:
        # Buat dummy image yang mirip dengan data training
        dummy_input = np.random.random((1, 224, 224, 3))
        dummy_input = preprocess_input(dummy_input)
        test_prediction = model.predict(dummy_input, verbose=0)
        
        print("=== MODEL TEST ===")
        print(f"Test prediction shape: {test_prediction.shape}")
        print(f"Test prediction sum: {np.sum(test_prediction):.4f}")
        print(f"Test prediction values: {test_prediction[0]}")
        
        # Periksa apakah prediksi masuk akal
        if np.sum(test_prediction[0]) < 0.9 or np.sum(test_prediction[0]) > 1.1:
            print("Warning: Total probability is not close to 1.0")
            return False
        
        # Periksa apakah ada prediksi yang masuk akal
        max_prob = np.max(test_prediction[0])
        if max_prob < 0.1:
            print("Warning: Maximum probability is too low")
            return False
        
        print("Model test passed!")
        return True
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

# Jalankan test
test_model_prediction()  

# ===== DATABASE INFORMASI MEDIS =====
# Info default untuk hasil prediksi
info_default = {
    'caries': {
        '100': {
            'gejala': 'Kerusakan gigi yang sangat jelas, lubang hitam yang dalam, nyeri hebat saat mengunyah, sensitivitas ekstrim terhadap suhu, bau mulut yang tidak sedap.',
            'penyebab': 'Kerusakan email gigi yang parah, infeksi bakteri yang sangat aktif, penumpukan plak dan tartar dalam jangka panjang, higiene mulut yang sangat buruk.',
            'pengobatan': 'Tindakan perawatan saluran akar (root canal) segera, kemungkinan crown atau mahkota gigi, antibiotik jika ada infeksi, konsultasi dokter gigi spesialis endodontis.'
        },
        '90': {
            'gejala': 'Lubang gigi yang terlihat jelas, nyeri sedang hingga berat saat makan, sensitivitas tinggi terhadap makanan manis dan dingin, perubahan warna gigi yang jelas.',
            'penyebab': 'Penumpukan plak yang signifikan, konsumsi makanan asam dan manis berlebihan, kurangnya perawatan gigi rutin, erosi email gigi.',
            'pengobatan': 'Penambalan gigi (filling) segera, pembersihan karang gigi, perubahan pola makan, konsultasi rutin dengan dokter gigi.'
        },
        '80': {
            'gejala': 'Bintik putih atau coklat pada gigi, nyeri ringan saat makan makanan manis, sensitivitas terhadap makanan dingin, permukaan gigi terasa kasar.',
            'penyebab': 'Awal pembentukan plak, demineralisasi email gigi, konsumsi minuman bersoda, kurangnya asupan fluoride.',
            'pengobatan': 'Aplikasi fluoride topikal, perawatan remineralisasi, perbaikan pola makan, peningkatan kebersihan mulut.'
        },
        '70': {
            'gejala': 'Perubahan warna minimal pada gigi, sensitivitas ringan yang tidak konsisten, tidak ada nyeri yang signifikan.',
            'penyebab': 'Tahap awal pembentukan plak, pola makan yang mulai tidak sehat, frekuensi menyikat gigi yang kurang.',
            'pengobatan': 'Peningkatan rutinitas kebersihan mulut, penggunaan pasta gigi dengan fluoride, konsultasi pencegahan dengan dokter gigi.'
        }
    },
    'no-caries': {
        '100': {
            'gejala': 'Gigi putih bersih, tidak ada tanda-tanda kerusakan, email gigi kuat dan mengkilap.',
            'penyebab': 'Perawatan gigi yang sangat baik, pola makan sehat seimbang, rutinitas kebersihan mulut yang optimal.',
            'pengobatan': 'Pertahankan rutinitas kebersihan gigi yang baik, pemeriksaan rutin 6 bulan sekali ke dokter gigi.'
        },
        '90': {
            'gejala': 'Gigi dalam kondisi baik, sedikit perubahan warna alami, tidak ada keluhan.',
            'penyebab': 'Perawatan gigi yang baik, konsumsi makanan sehat, menyikat gigi teratur.',
            'pengobatan': 'Lanjutkan perawatan rutin, pertimbangkan pembersihan gigi profesional tahunan.'
        },
        '80': {
            'gejala': 'Kondisi gigi cukup baik, mungkin ada sedikit noda permukaan, tidak ada masalah struktural.',
            'penyebab': 'Perawatan gigi yang cukup baik, beberapa kebiasaan diet yang perlu diperbaiki.',
            'pengobatan': 'Tingkatkan kebersihan mulut, kurangi konsumsi makanan/minuman yang mewarnai gigi.'
        },
        '70': {
            'gejala': 'Gigi secara umum sehat, mungkin ada sedikit plak atau noda.',
            'penyebab': 'Perawatan dasar yang dilakukan, namun bisa ditingkatkan.',
            'pengobatan': 'Tingkatkan frekuensi menyikat gigi, gunakan obat kumur, perbaiki teknik menyikat gigi.'
        }
    },
    'bukan-gigi': {
        'pesan': 'Gambar yang ditampilkan bukan merupakan gambar gigi. Silakan ambil gambar gigi dengan posisi yang tepat, pencahayaan yang baik, dan fokus yang jelas untuk pemeriksaan kesehatan gigi.'
    }
}

# ===== FUNGSI PENGOLAHAN GAMBAR =====
# Fungsi untuk mengolah gambar
def prepare_image(file_path):
    try:
        # Muat dan resize gambar ke ukuran input MobileNetV3
        img = load_img(file_path, target_size=(224, 224))
        # Konversi ke array
        img_array = img_to_array(img)
        # Terapkan preprocessing MobileNetV3
        img_array = preprocess_input(img_array)
        # Tambahkan dimensi batch
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error in prepare_image: {str(e)}")
        return None

# ===== KONFIGURASI FOLDER PENYIMPANAN =====
# Folder penyimpanan untuk foto
save_folder = r'D:\Web\deteksi_gigi_cnn_tl_3_label\camera\photos'  
os.makedirs(save_folder, exist_ok=True)

# ===== FUNGSI PREDIKSI UPLOAD GAMBAR =====
def predict_caries(img_array):
    """
    Prediksi apakah gambar adalah Caries, No Caries, atau Bukan Gigi
    """
    try:
        # Prediksi dengan model
        predictions = model.predict(img_array, verbose=0)
        
        # Periksa apakah prediksi valid
        if np.sum(predictions[0]) == 0:
            print("Warning: All predictions are zero!")
            return 'no_caries', 0.0
        
        prediction_class = np.argmax(predictions)
        result = model_class_labels[prediction_class]
        probability = float(predictions[0][prediction_class] * 100)
        
        # Threshold confidence yang tinggi untuk model dengan akurasi 93%
        if result == 'bukan-gigi':
            # Threshold tinggi untuk "bukan-gigi" karena model sangat akurat
            if probability < 80:
                return 'no-caries', probability
        elif result == 'caries':
            # Threshold tinggi untuk caries karena model sangat akurat
            if probability < 85:
                return 'no-caries', probability
        elif result == 'no-caries':
            # Threshold tinggi untuk no-caries karena model sangat akurat
            if probability < 90:
                return 'caries', probability
        
        return result, probability
        
    except Exception as e:
        print(f"Error in predict_caries: {str(e)}")
        return 'no_caries', 0.0

# ===== FUNGSI PREDIKSI KAMERA REAL-TIME =====
def predict_caries_camera(img_array):
    """
    Prediksi untuk kamera real-time dengan threshold tinggi untuk model akurat
    """
    try:
        # Prediksi dengan model
        predictions = model.predict(img_array, verbose=0)
        
        # Periksa apakah prediksi valid
        if np.sum(predictions[0]) == 0:
            return 'no_caries', 0.0
        
        prediction_class = np.argmax(predictions)
        result = model_class_labels[prediction_class]
        probability = float(predictions[0][prediction_class] * 100)
        
        # Threshold tinggi untuk real-time karena model sangat akurat (93%)
        if result == 'bukan-gigi':
            # Threshold tinggi untuk "bukan-gigi"
            if probability < 75:
                return 'no-caries', probability
        elif result == 'caries':
            # Threshold tinggi untuk caries
            if probability < 80:
                return 'no-caries', probability
        elif result == 'no-caries':
            # Threshold tinggi untuk no-caries
            if probability < 85:
                return 'caries', probability
        
        return result, probability
        
    except Exception as e:
        print(f"Error in predict_caries_camera: {str(e)}")
        return 'no_caries', 0.0

# ===== ROUTE HALAMAN UTAMA =====
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ===== ROUTE PREDIKSI UPLOAD GAMBAR =====
@app.route('/prediksi', methods=["GET", "POST"])
def prediksi():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
            
        # Validasi tipe file
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not '.' in file.filename or \
           file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400
            
        # Validasi ukuran file (maksimal 5MB)
        if len(file.read()) > 5 * 1024 * 1024:  # 5MB dalam bytes
            return jsonify({"error": "File too large"}), 400
        file.seek(0)  # Reset pointer file
        
        if file:
            try:
                # Simpan file yang diupload
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                # Prediksi gambar
                img = prepare_image(file_path)
                if img is None:
                    return render_template('prediksi.html', error="Gagal memproses gambar")

                # Prediksi Caries atau No Caries
                result, probability = predict_caries(img)

                # Ambil info tentang hasil prediksi berdasarkan level akurasi
                accuracy_level = None
                if result == 'bukan-gigi':
                    # Untuk "bukan-gigi" tidak perlu accuracy level
                    info = info_default.get(result, {})
                else:
                    # Untuk caries dan no-caries, tentukan accuracy level
                    if probability >= 95:
                        accuracy_level = '100'
                    elif probability >= 85:
                        accuracy_level = '90'
                    elif probability >= 75:
                        accuracy_level = '80'
                    else:
                        accuracy_level = '70'
                    info = info_default.get(result, {}).get(accuracy_level, None)
                # Dapatkan label level (konsisten dengan kamera real time)
                level_label = get_level_label(result, probability)
                # Dapatkan nama display yang user-friendly
                display_name = get_display_name(result)

                # URL gambar yang diupload
                image_url = url_for('static', filename='uploads/' + file.filename)

                return render_template(
                    "prediksi.html",
                    prediction=result,
                    prediction_display=display_name,
                    probability=probability,
                    image_url=image_url,
                    info=info,
                    accuracy_level=accuracy_level,
                    level_label=level_label
                )
            except Exception as e:
                print(f"Error in prediksi: {str(e)}")
                return render_template('prediksi.html', error="Terjadi kesalahan saat memproses gambar")

    return render_template('prediksi.html', prediction=None)

# ===== ROUTE CEK KAMERA =====
@app.route('/check_camera')
def check_camera():
    available_cameras = []
    for idx in range(3):  # Cek kamera 0, 1, dan 2
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(idx)
                cap.release()
        except:
            continue
    return jsonify({
        "status": "success",
        "available_cameras": available_cameras,
        "message": "Kamera tersedia" if available_cameras else "Tidak ada kamera yang ditemukan"
    })

@app.route('/camera')
def camera():
    return render_template('camera.html')

# ===== FUNGSI HELPER LABEL =====
def get_display_name(result):
    """
    Mengkonversi nama label internal ke nama yang ditampilkan ke user
    """
    display_names = {
        'caries': 'Caries (Karies)',
        'no-caries': 'No Caries (Tidak Ada Karies)',
        'bukan-gigi': 'Bukan Gigi'
    }
    return display_names.get(result, result)

def get_level_label(result, probability):
    if result == 'caries':
        if probability >= 90:
            return 'Karies Serius'
        elif probability >= 80:
            return 'Karies Sedang'
        elif probability >= 70:
            return 'Karies Ringan'
        else:
            return 'Karies Awal'
    elif result == 'no-caries':
        if probability >= 90:
            return 'Gigi Sangat Sehat'
        elif probability >= 80:
            return 'Gigi Sehat'
        elif probability >= 70:
            return 'Gigi Cukup Sehat'
        else:
            return 'Gigi Kurang Sehat'
    elif result == 'bukan-gigi':
        return 'Bukan Gigi'

def get_level_label_camera(result, probability):
    """
    Level label yang disederhanakan untuk kamera real-time
    """
    if result == 'caries':
        if probability >= 85:
            return 'Karies'
        else:
            return 'Karies'
    elif result == 'no-caries':
        if probability >= 85:
            return 'No Karies'
        else:
            return 'No Karies'
    elif result == 'bukan-gigi':
        if probability >= 50:
            return 'Bukan Gigi'
        else:
            return 'Bukan Gigi'
    else:
        return 'Bukan Gigi'

# ===== FUNGSI GENERATE VIDEO STREAMING =====
# Fungsi untuk menghasilkan video feed langsung
def generate_frames():
    # Coba index 0, 1, 2 secara berurutan
    camera = None
    for idx in [0, 1, 2]:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                camera = cam
                break
            cam.release()
    if camera is None or not camera.isOpened():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Kamera tidak ditemukan", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_frame)
            error_frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + error_frame_bytes + b'\r\n')
    
    frame_count = 0
    last_result = None
    last_prob = None
    # Tambahkan smoothing untuk mengurangi fluktuasi
    result_history = []
    prob_history = []
    history_size = 5  # Jumlah frame untuk smoothing (lebih besar untuk stabilitas)
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                frame_count += 1
                h, w, _ = frame.shape
                
                # Resize frame untuk streaming ringan (perbaikan: ukuran lebih besar)
                frame = cv2.resize(frame, (640, 480))
                h, w, _ = frame.shape
                
                # Ukuran kotak deteksi 45% dari frame, square untuk konsistensi dengan model
                box_size = int(min(w, h) * 0.45)  # 45% dari dimensi terkecil untuk fokus per gigi
                box_w, box_h = box_size, box_size
                x1 = (w - box_w) // 2
                y1 = (h - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                
                # Prediksi setiap frame untuk akurasi lebih baik (perbaikan: frekuensi lebih tinggi)
                if frame_count % 1 == 0 or last_result is None:
                    crop = frame[y1:y2, x1:x2]
                    # Konversi BGR ke RGB (OpenCV menggunakan BGR, PIL menggunakan RGB)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    # Resize langsung ke ukuran model tanpa resize ganda
                    crop_resized = cv2.resize(crop_rgb, (224, 224))
                    img_array = img_to_array(crop_resized)
                    img_array = preprocess_input(img_array)
                    img_array = np.expand_dims(img_array, axis=0)
                    # Prediksi Caries atau No Caries (menggunakan fungsi khusus kamera)
                    current_result, current_prob = predict_caries_camera(img_array)
                    
                    # Smoothing prediksi untuk mengurangi fluktuasi
                    result_history.append(current_result)
                    prob_history.append(current_prob)
                    
                    # Batasi ukuran history
                    if len(result_history) > history_size:
                        result_history.pop(0)
                        prob_history.pop(0)
                    
                    # Ambil hasil yang paling sering muncul
                    if len(result_history) >= history_size:
                        result_counter = Counter(result_history)
                        last_result = result_counter.most_common(1)[0][0]
                        # Rata-rata probability untuk hasil yang dominan
                        dominant_results = [prob for prob, res in zip(prob_history, result_history) if res == last_result]
                        last_prob = sum(dominant_results) / len(dominant_results) if dominant_results else current_prob
                    else:
                        last_result = current_result
                        last_prob = current_prob
                
                result = last_result
                probability = last_prob
                
                # Warna kotak berdasarkan hasil
                if result == 'caries':
                    color = (0, 0, 255)  # Merah untuk karies
                    bg_color = (0, 0, 255)
                elif result == 'no-caries':
                    color = (0, 200, 0)  # Hijau untuk no karies
                    bg_color = (0, 200, 0)
                elif result == 'bukan-gigi':
                    color = (255, 165, 0)  # Orange untuk bukan gigi
                    bg_color = (255, 165, 0)
                else:
                    color = (128, 128, 128)  # Abu-abu untuk default
                    bg_color = (128, 128, 128)
                
                # Label level untuk kamera real-time (disederhanakan)
                level_label = get_level_label_camera(result, probability)
                
                # Gambar kotak dengan ketebalan yang lebih jelas
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Label di atas kotak dengan background yang lebih jelas
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                label = f'{level_label}: {probability:.1f}%'
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                label_x = x1 + (box_w - label_w) // 2
                label_y = y1 - 20 if y1 - 20 > label_h else y1 + label_h + 10
                
                # Background label yang lebih besar
                cv2.rectangle(frame, (label_x - 10, label_y - label_h - 10), 
                             (label_x + label_w + 10, label_y + 10), bg_color, -1)
                cv2.putText(frame, label, (label_x, label_y), font, font_scale, 
                           (255,255,255), thickness, cv2.LINE_AA)
                
                # Encode frame dengan kualitas yang lebih baik (perbaikan: kualitas 85%)
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error in generate_frames: {str(e)}")
                continue
    camera.release()

# ===== ROUTE VIDEO STREAMING =====
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== ROUTE AMBIL FOTO =====
@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    try:
        data = request.get_json()

        # Ambil data gambar dari request (format base64)
        image_data = data.get('image')

        if image_data:
            # Hapus prefix (data:image/png;base64,)
            image_data = image_data.split(",")[1]

            # Decode data gambar dari base64
            image_bytes = base64.b64decode(image_data)

            # Generate nama file unik berdasarkan waktu saat ini
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_path = os.path.join(save_folder, f"photo_{timestamp}.png")

            # Simpan gambar ke disk
            with open(file_path, 'wb') as file:
                file.write(image_bytes)

            # Baca ulang gambar asli
            frame = cv2.imread(file_path)
            if frame is None:
                return jsonify({"status": "error", "message": "Gagal membaca gambar"})

            h, w, _ = frame.shape
            box_w, box_h = int(w * 0.45), int(h * 0.45)  # 45% untuk fokus per gigi
            x1 = (w - box_w) // 2
            y1 = (h - box_h) // 2
            x2 = x1 + box_w
            y2 = y1 + box_h
            crop = frame[y1:y2, x1:x2]
            # Konversi BGR ke RGB (OpenCV menggunakan BGR, PIL menggunakan RGB)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (224, 224))
            img_array = img_to_array(crop_resized)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            # Prediksi Caries atau No Caries
            result, probability = predict_caries(img_array)

            # Ambil info tentang hasil prediksi berdasarkan level akurasi
            accuracy_level = None
            if result == 'bukan-gigi':
                # Untuk "bukan-gigi" tidak perlu accuracy level
                info = info_default.get(result, {})
            else:
                # Untuk caries dan no-caries, tentukan accuracy level
                if probability >= 95:
                    accuracy_level = '100'
                elif probability >= 85:
                    accuracy_level = '90'
                elif probability >= 75:
                    accuracy_level = '80'
                else:
                    accuracy_level = '70'
                info = info_default.get(result, {}).get(accuracy_level, None)
            # Dapatkan nama display yang user-friendly
            display_name = get_display_name(result)

            # Respon dengan sukses dan hasil prediksi
            return jsonify({
                "status": "success",
                "message": "Photo saved successfully.",
                "prediction": result,
                "prediction_display": display_name,
                "probability": probability,
                "info": info,
                "accuracy_level": accuracy_level,
                "saved_filename": f"photo_{timestamp}.png"
            })

        return jsonify({"status": "error", "message": "No image data found"})
    except Exception as e:
        print(f"Error in capture_photo: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

# ===== ROUTE UNDUH FOTO =====
@app.route('/download_photo/<filename>')
def download_photo(filename):
    # Mengizinkan unduh file dari folder photos
    return send_from_directory(save_folder, filename, as_attachment=True)

# ===== EKSEKUSI UTAMA =====
if __name__ == "__main__":
    app.run(debug=True)
