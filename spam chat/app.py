# spam_detector_app/app.py

import streamlit as st
import joblib
import re
import string

# --- PASTIKAN st.set_page_config() ADA DI SINI, PALING ATAS! ---
# Mengatur konfigurasi halaman Streamlit:
# page_title: Judul yang muncul di tab browser.
# page_icon: Ikon kecil di tab browser.
# layout="wide": Menggunakan lebar penuh halaman browser, bukan hanya bagian tengah.
st.set_page_config(
    page_title="Deteksi Pesan Spam Tingkat Lanjut",
    page_icon="🚫",
    layout="wide"
)
# --- AKHIR DARI KONFIGURASI HALAMAN ---


# --- 1. Muat Model dan Vectorizer yang Telah Disimpan ---
# @st.cache_resource: Dekorator ini memberitahu Streamlit untuk memuat sumber daya
# (seperti model ML) hanya sekali, lalu menyimpannya dalam cache. Ini membuat
# aplikasi jauh lebih cepat dan efisien karena model tidak dimuat ulang setiap interaksi.
@st.cache_resource
def load_resources():
    try:
        loaded_model = joblib.load('spam_model.pkl')
        loaded_vectorizer = joblib.load('vectorizer.pkl')
        return loaded_model, loaded_vectorizer
    except FileNotFoundError:
        st.error("🚨 Error: File model ('spam_model.pkl') atau vectorizer ('vectorizer.pkl') tidak ditemukan.")
        st.error("Pastikan Anda sudah menjalankan script 'train_model.py' di lokal dan mengupload kedua file tersebut ke repositori GitHub Anda.")
        st.stop() # Menghentikan eksekusi aplikasi jika file kritis tidak ditemukan
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model atau vectorizer: {e}")
        st.stop()

# Panggil fungsi untuk memuat model dan vectorizer saat aplikasi dimulai
model, vectorizer = load_resources()

# --- 2. Fungsi Preprocessing Teks ---
# Fungsi ini harus sama persis dengan yang digunakan saat melatih model di train_model.py.
# Konsistensi dalam preprocessing adalah kunci untuk akurasi model.
def preprocess_text(text):
    text = text.lower() # Mengubah semua teks menjadi huruf kecil
    # Menghapus URL (mencocokkan pola http://, https://, atau www.)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Menghapus semua tanda baca dari teks
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Menghapus semua angka
    text = re.sub(r'\d+', '', text)
    # Menghapus spasi berlebih dan spasi di awal/akhir teks
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. Antarmuka Pengguna Streamlit ---

# Header Utama Aplikasi
st.title("🤖 Detektor Pesan SPAM Tingkat Lanjut")
st.markdown("""
    Aplikasi cerdas ini membantu Anda mengidentifikasi pesan yang mencurigakan.
    Cukup masukkan pesan teks, dan model kami akan memprediksi apakah itu **SPAM** atau **HAM (Bukan Spam)**.
""")

st.markdown("---") # Garis pemisah untuk estetika

# Menggunakan kolom untuk tata letak yang lebih baik
col1, col2 = st.columns([2, 1]) # Kolom kiri 2x lebih besar dari kolom kanan

with col1:
    st.subheader("📝 Masukkan Pesan Anda di Sini:")
    message_input = st.text_area(
        "Contoh: 'Selamat! Anda memenangkan hadiah! Klik link ini sekarang!' atau 'Rapat jam 10 pagi besok.'",
        height=200, # Tinggi area teks diperbesar
        help="Tulis pesan yang ingin Anda deteksi di kotak ini.",
        key="main_message_input" # Kunci unik untuk widget
    )

    # Tombol untuk memulai prediksi
    if st.button("🚀 Deteksi Pesan Sekarang!", use_container_width=True): # Tombol akan mengisi lebar container
        if message_input.strip() == "": # Memeriksa apakah input kosong atau hanya spasi
            st.warning("⚠️ Mohon masukkan pesan terlebih dahulu untuk dideteksi!")
        else:
            # Tampilkan indikator loading saat prediksi berlangsung
            with st.spinner('⏳ Menganalisis pesan Anda...'):
                processed_message = preprocess_text(message_input)
                message_vectorized = vectorizer.transform([processed_message])
                prediction = model.predict(message_vectorized)
                prediction_proba = model.predict_proba(message_vectorized)

            st.markdown("### Hasil Deteksi:")
            if prediction[0] == 1: # Jika prediksi adalah 1 (SPAM)
                st.error(f"**🔴 Pesan Ini Teridentifikasi sebagai SPAM!**")
                st.metric(label="Probabilitas SPAM", value=f"{prediction_proba[0][1]*100:.2f}%")
                st.write("---")
                st.warning("🚨 **Peringatan:** Pesan ini memiliki karakteristik yang kuat dari pesan spam. Harap berhati-hati dan jangan mengklik tautan atau memberikan informasi pribadi.")
            else: # Jika prediksi adalah 0 (HAM)
                st.success(f"**🟢 Pesan Ini Teridentifikasi sebagai HAM (Bukan Spam).**")
                st.metric(label="Probabilitas HAM", value=f"{prediction_proba[0][0]*100:.2f}%")
                st.write("---")
                st.info("👍 Pesan ini terlihat aman. Namun, selalu verifikasi pengirim jika Anda merasa ragu.")

with col2:
    st.subheader("💡 Informasi Aplikasi:")
    # Menggunakan expander untuk informasi tambahan
    with st.expander("Bagaimana Cara Kerja Detektor Ini?"):
        st.write("""
            Aplikasi ini menggunakan model Machine Learning sederhana berbasis **Naive Bayes**
            dan **Bag-of-Words (CountVectorizer)**. Prosesnya adalah:
            1.  Pesan Anda dibersihkan (huruf kecil, hapus URL, tanda baca, angka).
            2.  Teks diubah menjadi format numerik yang bisa dimengerti model.
            3.  Model memprediksi apakah pesan itu 'spam' atau 'ham'.
        """)
    with st.expander("Tentang Model yang Digunakan:"):
        st.write("""
            Model ini dilatih dengan **dataset dummy yang sangat kecil**.
            Ini berarti akurasinya mungkin tidak tinggi untuk semua jenis pesan spam di dunia nyata.
            Untuk deteksi yang lebih canggih, diperlukan:
            -   Dataset spam/ham yang jauh lebih besar dan beragam.
            -   Teknik pra-pemrosesan teks yang lebih kompleks (misalnya, stemming, lemmatization).
            -   Model Machine Learning yang lebih canggih (misalnya, SVM, Random Forest, atau model Deep Learning seperti LSTM/Transformer).
        """)
    st.markdown("---")
    st.write("Dibuat dengan ❤️ oleh **[deriffa]** untuk Tugas Deployment.")
    st.write("Terima kasih sudah mencoba!")