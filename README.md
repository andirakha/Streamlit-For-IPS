# 🎓 Sistem Lokasi Kampus Pintar (Indoor Positioning System)

Aplikasi web berbasis **Streamlit** untuk melacak posisi di dalam ruangan (*Indoor Positioning System*) di area kampus. Sistem ini menggunakan teknik *WiFi Fingerprinting* yang secara otomatis memindai sinyal Access Point (RSSI) di sekitar pengguna dan memprediksi lokasi ruangan secara *real-time* menggunakan algoritma **K-Nearest Neighbors (KNN)**.

---

## ✨ Fitur Utama

- 🔍 **Pemindaian WiFi Otomatis:** Terintegrasi dengan perangkat keras (adapter WiFi) melalui pustaka `pywifi` untuk memindai kekuatan sinyal (RSSI) dari Access Point target di sekitar secara otomatis (durasi ±12 detik).
- 📍 **Prediksi Lokasi Cerdas:** Menggunakan model Machine Learning KNN (`knn_model_radiomap.pkl`) dan Scaler (`scaler_radiomap.pkl`) untuk mencocokkan pola sinyal dan menentukan posisi saat ini dengan tingkat kepercayaan (*confidence level*).
- 📶 **Monitoring Sinyal Real-Time:** Menampilkan tabel interaktif berisi metrik kualitas sinyal (dalam satuan dBm) dari tiap Access Point yang terdeteksi.
- 🔒 **Aman & Efisien:** Pemrosesan prediksi dilakukan sepenuhnya secara lokal (di perangkat pengguna).
- 🏢 **Dukungan Area Terpetakan:** Saat ini model dilatih untuk mendeteksi area Gedung R FT UNTIRTA.

---

## 📂 Struktur Direktori

```text
├── .streamlit/               # Konfigurasi UI bawaan Streamlit
├── venv/                     # Virtual Environment (diabaikan oleh Git)
├── knn_model_radiomap.pkl    # Pre-trained model K-Nearest Neighbors
├── scaler_radiomap.pkl       # Pre-trained scaler untuk normalisasi data RSSI
├── main.py                   # Source code utama antarmuka dan logika aplikasi
└── requirements.txt          # Daftar dependencies library Python
```

---

## 🛠️ Persyaratan Sistem

- **Python:** Versi 3.8 atau lebih baru.
- **Perangkat Keras:** Laptop atau PC yang memiliki Adapter WiFi aktif (wajib ada untuk fitur `pywifi` melakukan _scanning_ area).
- **Sistem Operasi:** Windows / macOS / Linux.

---

## 🚀 Panduan Instalasi

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi di komputer lokal Anda:

### 1. Clone Repository

```text
git clone [https://github.com/andirakha/Streamlit-For-IPS.git](https://github.com/andirakha/Streamlit-For-IPS.git)
cd Streamlit-For-IPS
```

### 2. Buat dan Aktifkan Virtual Environment

Sangat disarankan menggunakan _virtual environment_ agar pustaka yang diinstal tidak bentrok dengan proyek lain.

- **Windows**

```text
python -m venv venv
.\venv\Scripts\activate
```

- **macOS / Linux**

```text
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Library (Dependencies)

Instal semua pustaka yang terdaftar pada `requirements.txt`:

```text
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

Eksekusi file utama menggunakan Streamlit:

```text
streamlit run main.py
```

Aplikasi akan otomatis terbuka pada browser di alamat `http://localhost:8501`.

---

## ⚙️ Cara Kerja Sistem (Metodologi)

1. **Inisialisasi:** Saat aplikasi berjalan, sistem memuat _Pre-trained Model_ (KNN) dan _Scaler_ ke dalam memori.
2. **Pemindaian Sinyal (Scanning):** Ketika pengguna menekan tombol **"Pindai Lokasi**", modul `pywifi` akan mengaktifkan adapter WiFi perangkat untuk memindai jaringan di sekitar selama 12 detik.
3. **Penyaringan (Filtering):** Sistem mencocokkan MAC _Address_ (BSSID) yang tertangkap dengan daftar Target _Access Point_ kampus.
4. **Preprocessing:** Nilai RSSI yang didapat (atau nilai _default_ -100 dBm jika AP tidak terjangkau) akan disusun menjadi _feature vector_ dan dinormalisasi oleh `scaler_radiomap.pkl`.
5. **Prediksi:** Model KNN mengklasifikasikan pola array RSSI tersebut untuk menghasilkan label ruangan terdekat beserta persentase _confidence_.

---

## 📚 Teknologi Utama yang Digunakan

- **Streamlit:** Pembuatan antarmuka web interaktif (_Frontend_/_Dashboard_).
- **PyWiFi:** Modul kontrol antarmuka nirkabel untuk pemindaian _Access Point_.
- **Scikit-Learn:** Implementasi algoritma _K-Nearest Neighbors_ dan _Data Scaling_.
- **Pandas & NumPy:** Transformasi dan manipulasi _dataframe_ (_array vector_).



