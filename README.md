# SISTEM DETEKSI KARIES GIGI MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK (CNN)

## Deskripsi
Aplikasi web untuk deteksi kondisi gigi menggunakan model CNN dengan transfer learning MobileNetV3. Sistem ini dapat mengklasifikasikan gambar menjadi 3 kategori: bukan-gigi, caries, dan no-caries dengan akurasi tinggi.

## Struktur Arsitektur Aplikasi

### Backend (Flask)
- `app.py` - Aplikasi utama Flask 
- Model CNN - `dataset_gigi_cnn_tl_real.keras` 
- Transfer Learning - MobileNetV3Small 
- Computer Vision - OpenCV untuk pemrosesan gambar dan kamera

### Frontend (HTML/CSS/JS)
- 5 Template HTML - Responsif dengan Bootstrap 
- Static Assets - CSS, JavaScript, dan folder uploads
- Responsive Design - Mobile-first approach

### AI Model
- Arsitektur: MobileNetV3Small (transfer learning)
- Input: 224x224 pixels RGB
- Output: 3 kelas klasifikasi dengan probability scores
- Preprocessing: MobileNetV3 standard preprocessing

## Fitur Utama

### 1. Prediksi via Upload Gambar
- Upload gambar (JPG/PNG/JPEG)
- Analisis otomatis dengan model CNN
- Hasil prediksi dengan tingkat kepercayaan
- Informasi medis lengkap (gejala, penyebab, pengobatan)
- Level klasifikasi detail berdasarkan probability

### 2. Deteksi Real-time via Kamera
- Deteksi langsung dengan OpenCV
- Kotak deteksi berwarna dinamis (45% frame size)
- Pengambilan foto dengan hasil prediksi
- Penyimpanan otomatis di folder `camera/photos/`
- Frame smoothing untuk stabilitas hasil

### 3. Sistem Klasifikasi 3 Label
- `bukan-gigi` - Gambar bukan gigi
- `caries` - Gigi dengan karies/kerusakan
- `no-caries` - Gigi sehat tanpa karies

## Teknologi yang Digunakan

### Python Libraries
- Flask - Web framework
- TensorFlow/Keras - Deep learning
- OpenCV - Computer vision
- NumPy - Numeric computing
- PIL - Image processing

### Frontend
- HTML/CSS - Struktur dan styling
- JavaScript - Interaktivitas
- Bootstrap - UI framework responsif

## Struktur File

deteksi_gigi_cnn_tl_3_label/
├── app.py                          # Aplikasi Flask utama 
├── dataset_gigi_cnn_tl_real.keras # Model CNN yang sudah dilatih 
├── cnn_tl_real.ipynb             # Notebook training model 
├── templates/                     # Template HTML
│   ├── base.html                 # Layout utama
│   ├── home.html                 # Landing page
│   ├── prediksi.html             # Upload & prediction
│   ├── camera.html               # Real-time detection
│   └── about.html                # Informasi aplikasi
├── static/                        # File statis
│   ├── css/                      # Custom styling
│   └── uploads/                  # User uploads
└── camera/                        # Folder penyimpanan foto kamera
    └── photos/                    # Captured photos


## Cara Menjalankan

### 1. Jalankan Aplikasi
python app.py

### 2. Akses Aplikasi
- Home: `http://localhost:5000/`
- Prediksi: `http://localhost:5000/prediksi`
- Kamera: `http://localhost:5000/camera`
- About: `http://localhost:5000/about`

## Konfigurasi Model

### Threshold Confidence 
*Upload Gambar:*
- `bukan-gigi`: Threshold 80%
- `caries`: Threshold 85%
- `no-caries`: Threshold 90%

*Real-time Kamera:*
- `bukan-gigi`: Threshold 75%
- `caries`: Threshold 80%
- `no-caries`: Threshold 85%

### Level Klasifikasi Detail
- Caries: Serius (90-100%), Sedang (80-89%), Ringan (70-79%), Awal (<70%)
- No Caries: Sangat Sehat (90-100%), Sehat (80-89%), Cukup Sehat (70-79%), Kurang Sehat (<70%)

### Warna Indikator
- Merah - Caries (Karies)
- Hijau - No Caries (Gigi Sehat)
- Orange - Bukan Gigi

## Performance Features

### Model Optimization
- Verifikasi model saat startup
- Test prediction dengan dummy data
- Custom objects untuk MobileNetV3
- Error handling yang robust

### Real-time Processing
- Frame smoothing (5 frame history)
- JPEG encoding dengan kualitas 85%
- Adaptive thresholds untuk responsivitas
- Camera detection otomatis (index 0,1,2)

## API Endpoints

- `/` - Home page
- `/prediksi` - Upload dan prediksi gambar
- `/camera` - Real-time detection
- `/video_feed` - Video stream kamera
- `/capture_photo` - Pengambilan foto
- `/check_camera` - Cek ketersediaan kamera
- `/download_photo/<filename>` - Download foto