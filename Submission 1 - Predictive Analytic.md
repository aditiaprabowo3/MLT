# Laporan Proyek Machine Learning - Aditia Prabowo

## Domain Proyek

Industri wine merupakan salah satu sektor yang terus berkembang dengan permintaan yang tinggi terhadap kualitas produk. Kualitas wine dipengaruhi oleh berbagai faktor kimiawi selama proses produksinya. Dengan memanfaatkan data atribut kimia wine, kita dapat membangun model machine learning untuk memprediksi kualitas wine secara objektif.

**Rubrik/Kriteria Tambahan**:
Proyek ini bertujuan untuk mengembangkan model prediktif yang dapat mengklasifikasikan kualitas wine berdasarkan karakteristik kimianya. Solusi ini akan membantu produsen wine dalam:
- Mengontrol kualitas produksi
- Mengoptimalkan proses fermentasi
- Meningkatkan konsistensi kualitas produk

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
- Bagaimana memprediksi kualitas wine secara akurat berdasarkan parameter kimiawinya?
- Faktor kimia apa saja yang paling berpengaruh terhadap kualitas wine?
- Model machine learning apa yang paling efektif untuk masalah klasifikasi ini?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membangun model prediksi kualitas wine dengan akurasi tinggi
- Mengidentifikasi fitur-fitur paling signifikan yang mempengaruhi kualitas
- Membandingkan performa berbagai algoritma klasifikasi

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Menggunakan algoritma Logistic Regression dan SVM Classifier untuk klasifikasi dengan feature importance analysis
    - Melakukan optimasi hyperparameter untuk mendapatkan model terbaik

## Data Understanding
Dataset yang digunakan berasal dari: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- fixed acidity = Kadar keasaman tetap dalam wine, berpengaruh terhadap rasa 🍇
- volatile acidity = Kadar keasaman yang mudah menguap, bisa mempengaruhi bau tidak sedap 🍶
- citric acid Asam sitrat yang memberikan rasa segar pada wine 🍋
- residual sugar Jumlah gula yang tersisa setelah fermentasi, mempengaruhi rasa manis 🍬
- chlorides Jumlah garam (ion klorida) dalam wine, mempengaruhi rasa 🍸
- free sulfur dioxide SO2 bebas yang berfungsi sebagai pengawet dalam wine 🧪
- total sulfur dioxide Total jumlah SO2 (bebas dan terikat) dalam wine 🧂
- density Kerapatan wine, berkaitan dengan kandungan alkohol dan gula 🔍
- pH Tingkat keasaman atau kebasaan wine 🔵🔴
- sulphates Senyawa sulfat yang digunakan untuk stabilisasi wine ⚗️
- alcohol Persentase kandungan alkohol dalam wine 🍷
- quality Skor kualitas wine berdasarkan penilaian sensorik (biasanya 0-10) 🏅
- Id Nomor identifikasi unik untuk setiap sampel data 🆔

**Rubrik/Kriteria Tambahan (Opsional)**:
- Untuk memahami data lebih dalam saya menggunakan teknik visualisasi data atau exploratory data analysis, saya melakukan Histogram untuk semua variabel numerik pada data den
gan hasil seperti berikut:

![download](https://github.com/user-attachments/assets/12307173-ad98-4a4b-9991-4283a92bb73e)

- **Fixed Acidity**: Distribusi fixed acidity cenderung miring ke kanan. Sebagian besar wine memiliki tingkat fixed acidity sekitar 7–9, namun ada beberapa outlier hingga 15.
- **Volatile Acidity**: Volatile acidity terdistribusi cukup normal, dengan puncak di sekitar 0,5. Semakin tinggi volatile acidity biasanya menurunkan kualitas wine.
- **Citric Acid**: Banyak sampel yang memiliki citric acid rendah atau bahkan nol, menunjukkan bahwa tidak semua wine menggunakan kandungan asam sitrat tinggi.
- **Residual Sugar**: Distribusinya skewed ke kanan (right-skewed). Mayoritas wine memiliki sedikit residual sugar (1–3 gram/liter), namun ada beberapa wine dengan kadar gula sangat tinggi (hingga 15,5).
- **Chlorides**: Chlorides sangat skewed ke kanan juga. Kebanyakan wine memiliki kandungan chlorides rendah (< 0,1), hanya sedikit yang tinggi.
- **Free Sulfur Dioxide & Total Sulfur Dioxide**: Distribusinya juga right-skewed. Mayoritas wine memiliki kandungan sulfur dioxide rendah hingga sedang.
- **Density**: Distribusi density terlihat cukup normal, berpusat di sekitar 0,9967.
- **PH**: pH wine terdistribusi normal, mayoritas berada di antara 3,0–3,5, sesuai dengan keasaman wine pada umumnya.
- **Sulphates**: Distribusi sulphates right-skewed. Sebagian besar wine memiliki sulphates rendah (sekitar 0,5–0,7), dengan beberapa nilai ekstrem.
- **Alcohol**: Distribusi alcohol juga skewed ke kanan. Kebanyakan wine memiliki kadar alkohol sekitar 10–11%, tetapi ada yang hingga hampir 15%.
- **Quality**: Distribusi kualitas wine terpusat di skor 5 dan 6. Sangat sedikit wine yang mendapat nilai 3 atau 8, berarti mayoritas wine di dataset ini berada dalam kategori kualitas sedang.
- **Id**: Kolom Id terdistribusi merata, hanya berfungsi sebagai identifikasi unik tiap wine, tidak mengandung informasi analitis.

## Data Preparation
Pada tahap preparation ini saya menggunakan tekinik penyimpanan data sebelum pemodelan machine learning, Split Data ke Training dan Testing, dan Standardisasi Kolom Numerik

**Rubrik/Kriteria Tambahan (Opsional)**: 
- menyiapkan data sebelum pemodelan machine learning:
  - Memisahkan Fitur dan Target
    Fitur (X) adalah seluruh kolom kecuali kolom target (quality). Target (y) adalah kolom quality yang akan diprediksi.
  - Split Data ke Training dan Testing
    Data dibagi menjadi 80% data latih dan 20% data uji menggunakan train_test_split().
    Parameter stratify=y digunakan untuk menjaga proporsi kelas churn dan non-churn agar tetap seimbang di data train dan test.

    Kenapa split dilakukan sebelum normalisasi?
    Karena kita ingin menghindari data leakage. Jika kita melakukan normalisasi sebelum data dibagi, maka statistik (mean dan std) dari data uji bisa ikut dihitung,       
    sehingga informasi dari masa depan "bocor" ke model. Ini menyebabkan evaluasi model menjadi tidak akurat.
- Standardisasi Kolom Numerik
Fitur numerik distandarisasi menggunakan StandardScaler agar memiliki mean = 0 dan standar deviasi = 1.
Proses standarisasi hanya dilakukan pada data training (fit_transform) dan diterapkan ke data testing (transform) menggunakan scaler yang sama, tanpa menghitung ulang statistik dari data uji.
- Data preparation atau preprocessing sangat penting sebelum melatih model machine learning. Proses ini memastikan bahwa data yang digunakan dalam pelatihan model berkualitas tinggi dan siap untuk menghasilkan model yang optimal.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

Algoritma yang digunakan:
Random Forest:
- Kelebihan: Handles non-linear relationships, feature importance
- Kekurangan: Cenderung overfit jika tidak diatur dengan baik
XGBoost:
- Kelebihan: Powerful untuk data terstruktur, regularisasi bawaan
- Kekurangan: Lebih kompleks, butuh tuning

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Metrik evaluasi yang digunakan
- Accuracy: (TP+TN)/(TP+TN+FP+FN) - Mengukur proporsi prediksi benar
- Precision: TP/(TP+FP) - Kemampuan model tidak memprediksi negatif sebagai positif
- Recall: TP/(TP+FN) - Kemampuan menemukan semua positif
- F1-score: 2(PrecisionRecall)/(Precision+Recall) - Rata-rata harmonik precision-recall

Hasil evaluasi model terbaik Random Forest:
- Accuracy: 0.69
- Precision: 0.82
- Recall: 0.48
- F1-score: 0.61

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

Model ini berhasil memprediksi kualitas wine dengan cukup baik dan masih bisa diperbaiki supaya dapat melakukan penilaian kualitas wine dengan lebih baik.

**---Ini adalah bagian akhir laporan---**
