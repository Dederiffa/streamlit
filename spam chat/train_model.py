# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib # Pustaka untuk menyimpan/memuat objek Python

# --- Data Dummy untuk Latihan Model ---
# Ini adalah dataset yang SANGAT KECIL dan SEDERHANA.
# Untuk aplikasi di dunia nyata, Anda akan menggunakan dataset yang jauh lebih besar
# dan bervariasi (misalnya dari Kaggle: SMS Spam Collection Dataset).
data = {
    'text': [
        "Selamat! Anda memenangkan undian berhadiah! Klik link ini: bit.ly/undianpalsu", # Spam
        "Halo, apa kabar? Mau makan siang bersama?", # Ham
        "Peringatan! Akun bank Anda disusupi. Segera verifikasi di sini: bank-palsu.com", # Spam
        "Ingatkan rapat jam 3 sore. Jangan lupa laporan bulanan.", # Ham
        "Dapatkan hadiah jutaan rupiah! Balas YA ke 9999 sekarang!", # Spam
        "Bisakah kita jadwalkan ulang panggilan untuk besok?", # Ham
        "Penawaran eksklusif khusus untuk Anda! Waktu terbatas!", # Spam
        "Terima kasih atas pembaruan. Semuanya terlihat bagus.", # Ham
        "Anda memiliki pesan baru. Lihat sekarang juga!", # Spam
        "Hai, apakah Anda sibuk akhir pekan ini?", # Ham
        "Ini kesempatan terakhir Anda untuk klaim hadiah besar!", # Spam
        "Mari kita minum kopi nanti sore?", # Ham
        "Promo pulsa gratis! Cukup isi ulang 5 ribu dapat 100 ribu!", # Spam
        "Mohon konfirmasi identitas Anda dengan mengklik tautan ini.", # Spam
        "Sampai jumpa nanti!", # Ham
        "Pengiriman pesanan Amazon Anda tertunda. Lacak di sini: link-phishing.biz", # Spam
        "Tenggat waktu proyek adalah Senin depan.", # Ham
        "Pangeran Nigeria ingin berbagi kekayaan dengan Anda.", # Spam
        "Hubungi saya segera. Ada hal penting.", # Ham
        "Langganan Netflix Anda telah kedaluwarsa. Perbarui detail pembayaran Anda sekarang!", # Spam
    ],
    'label': [
        'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham',
        'spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'
    ]
}

df = pd.DataFrame(data)

# --- Preprocessing Label ---
# Mengubah label teks ('ham', 'spam') menjadi nilai numerik (0, 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# --- Pemisahan Data (Training dan Testing) ---
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Vectorization (Mengubah Teks menjadi Angka) ---
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# --- Latih Model Machine Learning ---
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# --- Evaluasi Model (Opsional, untuk memeriksa performa) ---
accuracy = model.score(X_test_vectorized, y_test)
print(f"Akurasi Model pada data uji: {accuracy:.2f}")

# --- Simpan Model dan Vectorizer ---
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\nModel 'spam_model.pkl' dan Vectorizer 'vectorizer.pkl' berhasil disimpan.")
print("Sekarang, Anda bisa melanjutkan ke langkah membuat aplikasi Streamlit (app.py) dan deployment.")