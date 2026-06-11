import streamlit as st
import time
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime
import pywifi

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Sistem Lokasi Kampus Pintar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inisialisasi Session State
if 'wifi_networks' not in st.session_state:
    st.session_state.wifi_networks = None

if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Konfigurasi Filter MAC Address (BSSID)
TARGET_APS = {
    "E2:5D:54:8D:CC:D7": "Teras R2.8 (2.4 GHz)",
    "E0:5D:54:8D:CC:D7": "Teras R2.8 (5 GHz)",
    "E2:5D:54:8D:CA:1F": "Teras R2.13 (2.4 GHz)",
    "E0:5D:54:8D:CA:1F": "Teras R2.13 (5 GHz)",
    "D6:31:27:87:7E:5B": "R2.2 (2.4 GHz)",
    "D4:31:27:87:7E:5B": "R2.2 (5 GHz)",
    "C0:A4:76:A6:72:0B": "R2.4 (2.4 GHz)",
    "C2:A4:76:86:72:0B": "R2.4 (5 GHz)",
    "E2:5D:54:8D:CA:0B": "R2.6 (2.4 GHz)",
    "E0:5D:54:8D:CA:0B": "R2.6 (5 GHz)",
    "C0:A4:76:A6:7B:B1": "R2.12 (2.4 GHz)",
    "C2:A4:76:86:7B:B1": "R2.12 (5 GHz)"
}
TARGET_APS = {bssid.lower(): lokasi for bssid, lokasi in TARGET_APS.items()}

@st.cache_resource
def load_ml_components():
    # Memuat model dan scaler menggunakan joblib.
    model_path = 'knn_model_radiomap.pkl'    
    scaler_path = 'scaler_radiomap.pkl'  
    
    model = None
    scaler = None
    
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        else:
            st.error(f"File {model_path} atau {scaler_path} tidak ditemukan! (Pastikan file ada di direktori yang sama)")
            return None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# Load model ke memori
ml_model, ml_scaler = load_ml_components()

# Fungsi untuk memindai jaringan WiFi dan menyiapkan data untuk prediksi
def get_campus_networks(scan_duration_seconds):

    # Inisialisasi daftar dengan semua target AP bernilai -100 (Tidak Terdeteksi)
    networks_dict = {
        bssid: {
            "bssid": bssid,
            "lokasi_ap": lokasi,
            "rssi": -100,
            "ssid": "Tidak Terjangkau"
        }
        for bssid, lokasi in TARGET_APS.items()
    }
    
    try:
        wifi = pywifi.PyWiFi()
        if len(wifi.interfaces()) == 0:
            st.error("Tidak ada antarmuka WiFi yang ditemukan pada perangkat ini.")
            return list(networks_dict.values())
        
        iface = wifi.interfaces()[0]
        
        # Memulai pemindaian
        iface.scan()
        
        # Jeda waktu agar adapter WiFi selesai memindai lingkungan sekitar
        time.sleep(scan_duration_seconds)
        
        # Mengambil hasil pindaian
        scan_results = iface.scan_results()
        
        if not scan_results:
            st.warning("Pemindaian selesai, tetapi tidak ada jaringan yang terdeteksi.")
            return list(networks_dict.values())

        for profile in scan_results:
            # Mengambil BSSID dan menjadikannya huruf kecil
            current_bssid = profile.bssid.lower() if profile.bssid else ""
            
            # Menggunakan rsplit untuk memotong string dari titik dua paling kanan
            if ":" in current_bssid:
                parts = current_bssid.rsplit(":", 1)
                current_bssid = "".join(parts)
            
            # Hanya proses jika bssid ada di daftar TARGET_APS
            if current_bssid in networks_dict:
                # pywifi.profile.signal sudah dalam format dBm
                rssi = profile.signal
                
                # Perbarui jika rssi dari scan lebih baik dari yang tersimpan
                if networks_dict[current_bssid]["rssi"] < rssi:
                    networks_dict[current_bssid]["rssi"] = rssi
                    
                    # Penanganan decoding SSID yang robust
                    ssid_str = profile.ssid
                    try:
                        ssid_str = profile.ssid.encode('raw_unicode_escape').decode('utf-8', 'ignore')
                    except UnicodeDecodeError:
                        try:
                            ssid_str = profile.ssid.encode('latin-1').decode('utf-8', 'ignore')
                        except Exception:
                            ssid_str = str(profile.ssid)
                    except AttributeError:
                        ssid_str = "<Hidden Network>"
                        
                    networks_dict[current_bssid]["ssid"] = ssid_str if ssid_str else "<Hidden Network>"

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memindai WiFi: {e}")
        
    # Ubah format dictionary menjadi list untuk dikembalikan ke UI/Model
    return list(networks_dict.values())

# Fungsi untuk memprediksi lokasi berdasarkan data jaringan yang dipindai
def predict_location(networks_data, model, scaler):
    
    if model is None or scaler is None:
        return {"location": "Error: Model tidak dimuat", "confidence": 0.0, "timestamp": datetime.now().strftime("%H:%M:%S")}

    # 1. Ekstrak RSSI dalam urutan yang TEPAT sesuai TARGET_APS
    feature_vector = []
    network_lookup = {net['bssid']: net['rssi'] for net in networks_data}
    
    for bssid in TARGET_APS.keys():
        rssi_val = network_lookup.get(bssid, -100)
        feature_vector.append(rssi_val)

    # 2. Reshape menjadi array 2D
    X_input = np.array(feature_vector).reshape(1, -1)

    # 3. Scaling Data
    try:
        X_scaled = scaler.transform(X_input)
    except Exception as e:
        return {"location": f"Error Scaling: {e}", "confidence": 0.0, "timestamp": datetime.now().strftime("%H:%M:%S")}

    # 4. Prediksi
    try:
        predicted_class = model.predict(X_scaled)[0]
    except Exception as e:
        return {"location": f"Error Prediksi: {e}", "confidence": 0.0, "timestamp": datetime.now().strftime("%H:%M:%S")}
    
    # 5. Hitung Confidence (Probabilitas)
    confidence = 1.0 
    try:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities))
    except Exception:
        pass 

    return {
        "location": str(predicted_class),
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

def reset_app():
    st.session_state.wifi_networks = None
    st.session_state.prediction_result = None

# Header
col_logo, col_header = st.columns([0.06, 0.9], vertical_alignment="center")
with col_logo:
    st.header("🏫")
with col_header:
    st.title("Sistem Lokasi Kampus Pintar")
    st.caption("Pelacakan posisi indoor berbasis WiFi untuk Civitas Akademika")

st.divider()

# Layout Utama: 2 Kolom
left_col, right_col = st.columns([2, 1], gap="large")

#  Kolom Kiri: Panel Prediksi 
with left_col:
    with st.container(border=True):
        st.subheader("🌲 Deteksi Lokasi Saya")
        st.write("Tekan tombol di bawah untuk memindai jaringan WiFi kampus dan menentukan posisi Anda.")
        
        col_act_1, col_act_2 = st.columns([3, 1])
        
        do_scan = col_act_1.button("🔍 Pindai Lokasi", type="primary", use_container_width=True)
        do_reset = col_act_2.button("Muat Ulang", use_container_width=True)

        if do_reset:
            reset_app()
            st.rerun()

        if do_scan:
            with st.spinner("Sedang memindai sinyal WiFi Kampus (membutuhkan waktu ±12 detik)..."):
                scanned_data = get_campus_networks(scan_duration_seconds=12)
                st.session_state.wifi_networks = scanned_data
            
            with st.spinner("Mencocokkan fingerprint lokasi..."):
                time.sleep(0.5)
                st.session_state.prediction_result = predict_location(scanned_data, ml_model, ml_scaler)
                st.rerun()

        # Hasil Prediksi
        result = st.session_state.prediction_result
        if result:
            st.divider()
            st.markdown("### 📍 Posisi Terdeteksi")
            
            st.markdown(
                f"""
                <div style="
                    background-color: #d1e7dd; 
                    border: 1px solid #badbcc;
                    color: #0f5132;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h3 style="
                        margin: 0; 
                        padding: 0; 
                        font-family: sans-serif;
                        font-weight: bold;
                    ">
                        {result['location']}
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            m1, m2 = st.columns(2, vertical_alignment= "center")
            with m1:
                st.progress(result['confidence'], text=f"**Tingkat Kepercayaan:** :green[{int(result['confidence'] * 100)}%]")
            with m2:
                st.metric(label="Waktu Pemindaian", value=result['timestamp'])
                
        elif not result:
            st.info("Sistem siap. Silakan mulai pemindaian.")

# Kolom Kanan: Monitoring Jaringan
with right_col:
    with st.container(border=True):
        st.subheader("📶 Sinyal Terdeteksi")
        
        networks = st.session_state.wifi_networks
        
        if networks:
            df = pd.DataFrame(networks)
            df = df.sort_values(by='rssi', ascending=False)

            # Normalisasi untuk visualisasi progress bar (0-100)
            df['Kualitas'] = df['rssi'].apply(lambda x: 100 + x if x > -100 else 0)

            max_pixels = 360  
            row_height = 35   
            needed_height = (len(df) + 1) * row_height
            use_height = min(needed_height, max_pixels)
            
            st.dataframe(
                df[['lokasi_ap', 'Kualitas', 'rssi']],
                use_container_width=True,
                hide_index=True,
                height=use_height,
                column_config={
                    "lokasi_ap": "Lokasi Access Point",
                    "Kualitas": st.column_config.ProgressColumn(
                        "Sinyal",
                        format="%d%%",
                        min_value=0,
                        max_value=69,
                        help="Indikator kekuatan sinyal"
                    ),
                    "rssi": st.column_config.TextColumn(
                        "dBm",
                        help="Nilai RSSI (Received Signal Strength Indicator)"
                    ),
                }
            )
        else:
            st.warning("Belum ada data jaringan.")
            st.caption("Lakukan pemindaian terlebih dahulu.")

st.markdown("---")
c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("##### 🌿 Green Campus")
        st.caption("Hemat energi dengan pelacakan lokasi yang efisien.")

with c2:
    with st.container(border=True):
        st.markdown("##### 🔐 Keamanan")
        st.caption("Data lokasi hanya diproses lokal dan terenkripsi.")

with c3:
    with st.container(border=True):
        st.markdown("##### 🎓 Akademik")
        st.caption("Terintegrasi dengan sistem presensi mahasiswa.")