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
    - Menggunakan 3 model *machine learning* yaitu *Logistic Regression*, *SVM Classifier*, dan *Random Forest* untuk klasifikasi dengan feature importance analysis
    - Melakukan optimasi hyperparameter untuk mendapatkan model terbaik
    - Menggunakan confusion matrix dan f1 score pada masing-masing model *machine learning* untuk menemukan model terbaik berdasarkan akurasi tertinggi.

## Data Understanding
Dataset yang digunakan untuk memprediksi kualitas wine diambil dari platform Kaggle dan dapat diakses melalui tautan berikut: [Wine Quality Dataset oleh Yasser H](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset). Dataset ini dipublikasikan oleh Yasser H pada tanggal 13 Oktober 2022. 

Kumpulan data ini menyajikan informasi kimiawi dan sensorik dari berbagai sampel wine, yang mencakup 1.143 entri. Data ini sangat berguna untuk analisis kualitas wine, klasifikasi berdasarkan fitur-fitur kimia, serta pelatihan model machine learning prediktif. 

Variabel target dalam dataset ini adalah `quality`, yang merepresentasikan skor kualitas wine berdasarkan penilaian sensorik (biasanya dalam rentang 0–10).

Informasi struktur dataset dapat dilihat pada gambar berikut:

<img src="https://github.com/user-attachments/assets/1ccb3dfe-494e-494e-99f6-8f9b6adc7798" alt="Informasi Dataset" width="400">
Dari gambar yang di tampilkan terdapat 11 variabel bertipe float64 dan 2 variabel bertipe int64

### Deskripsi Variabel
Dataset ini memiliki 13 variabel dengan keterangan sebagai berikut:
| Variabel              | Keterangan                                                                 |
|-----------------------|----------------------------------------------------------------------------|
| `fixed acidity`       | Kadar keasaman tetap dalam wine, berpengaruh terhadap rasa 🍇              |
| `volatile acidity`    | Kadar keasaman yang mudah menguap, bisa mempengaruhi bau tidak sedap 🍶     |
| `citric acid`         | Asam sitrat yang memberikan rasa segar pada wine 🍋                         |
| `residual sugar`      | Jumlah gula yang tersisa setelah fermentasi, mempengaruhi rasa manis 🍬     |
| `chlorides`           | Jumlah garam (ion klorida) dalam wine, mempengaruhi rasa 🍸                 |
| `free sulfur dioxide` | SO₂ bebas yang berfungsi sebagai pengawet dalam wine 🧪                     |
| `total sulfur dioxide`| Total jumlah SO₂ (bebas dan terikat) dalam wine 🧂                         |
| `density`             | Kerapatan wine, berkaitan dengan kandungan alkohol dan gula 🔍             |
| `pH`                  | Tingkat keasaman atau kebasaan wine 🔵🔴                                   |
| `sulphates`           | Senyawa sulfat yang digunakan untuk stabilisasi wine ⚗️                   |
| `alcohol`             | Persentase kandungan alkohol dalam wine 🍷                                 |
| `quality`             | Skor kualitas wine berdasarkan penilaian sensorik (0–10) 🏅                |

| `Id`                  | Nomor identifikasi unik untuk setiap sampel data 🆔  


### Menangani Missing Value dan Duplicate Data (Duplikasi Data)
Pada tahap ini kita akan mengecek data yang tidak valid pada dataset. Setelah diperiksa apakah terdapat kolom yang bernilai null, hasilnya adalah tidak ada. Sedangkan data duplikat atau data ganda juga tidak ada. Maka dengan demikian data siapa untuk dianalisis pada tahap selanjutnya.

### Exploratory Data Analysis (EDA)

### 1. pesebaran data kolom numerikal dalam bentuk grafik histogram
Pada tahap ini, kita akan membuat visualisasi data numerikal dalam bentuk grafik dengan menggunakan library python matplotlib
<img src="https://github.com/user-attachments/assets/bddb04d6-e896-4b0f-8fe9-6969c2804a5a" alt="Informasi Dataset" width="600">

Interpretasi:
1. Fixed Acidity
    - Distribusi cenderung **miring ke kanan** (*right-skewed*).
    - Sebagian besar wine memiliki fixed acidity sekitar **7–9**.
    - Terdapat beberapa **outlier** hingga mencapai angka **15**.
2. Volatile Acidity
    - Terdistribusi **cukup normal**, dengan puncak di sekitar **0,5**.
    - Nilai **volatile acidity yang tinggi** biasanya berkorelasi negatif terhadap kualitas wine.
3. Citric Acid
    - Banyak sampel memiliki nilai **citric acid rendah atau nol**.
    - Menunjukkan bahwa **tidak semua wine mengandung asam sitrat** dalam kadar tinggi.
4. Residual Sugar
    - Distribusinya **right-skewed**.
    - Mayoritas wine memiliki residual sugar **1–3 g/L**, tetapi beberapa mencapai hingga **15,5 g/L**.
5. Chlorides
    - Distribusi **sangat miring ke kanan**.
    - Sebagian besar wine memiliki kadar chlorides rendah (**< 0,1**).
6. Free Sulfur Dioxide & Total Sulfur Dioxide
    - Kedua fitur ini memiliki distribusi **right-skewed**.
    - Kandungan sulfur dioxide umumnya berada pada tingkat **rendah hingga sedang**.
7. Density
    - Distribusi **cukup normal**.
    - Nilai density berpusat di sekitar **0,9967**.
8. pH
    - Terdistribusi **normal**, mayoritas berada di antara **3,0–3,5**.
    - Rentang ini sesuai dengan tingkat keasaman wine pada umumnya.
9. Sulphates
    - Distribusinya **right-skewed**.
    - Sebagian besar wine memiliki kadar sulphates di sekitar **0,5–0,7**, dengan beberapa nilai ekstrem.
10. Alcohol
    - Distribusi **miring ke kanan**.
    - Kebanyakan wine memiliki kadar alkohol sekitar **10–11%**, namun ada yang mencapai hampir **15%**.
10. Quality
    - Distribusi kualitas wine **terpusat di skor 5 dan 6**.
    - Sangat sedikit wine yang mendapat nilai **3 atau 8**, menunjukkan mayoritas wine dalam dataset ini memiliki kualitas **sedang**.
11. Id
    - Kolom ini hanya merupakan **identifikasi unik** untuk tiap wine.
    - Tidak memiliki nilai analitis dan **terdistribusi merata**.
  
### 2. Analisis Korelasi Fitur terhadap Kualitas Wine 
Pada tahap ini kita akan Melihat korelasi variabel numerik dengan menggunakan Heatmap
<img src="https://github.com/user-attachments/assets/6392fc73-db28-4fec-b88c-ef57f8d09c09" alt="Informasi Dataset" width="600">

Interpretasi:
1. Fitur yang Berkorelasi Cukup Baik dengan Target `quality`
    - **Alcohol**  
      Menunjukkan **korelasi positif** dengan `quality` sebesar **0.48**.  
      → Artinya, semakin tinggi kadar alkohol, cenderung semakin baik kualitas wine.
    - **Volatile Acidity**  
      Memiliki **korelasi negatif** dengan `quality` sebesar **-0.41**.  
      → Semakin tinggi keasaman volatil, cenderung menurunkan kualitas wine.
2. Fitur yang Berkorelasi Tinggi Satu Sama Lain (*Multikolinearitas*)
    - **Free Sulfur Dioxide** dan **Total Sulfur Dioxide**  
      Memiliki korelasi sebesar **0.66**.  
      → Karena `total sulfur dioxide` mencakup `free sulfur dioxide`, korelasi tinggi ini wajar.  
      → Hal ini bisa menjadi pertimbangan untuk melakukan **reduksi fitur** saat modeling, guna menghindari multikolinearitas.
3. Korelasi Lain terhadap `quality`
    - Korelasi fitur-fitur lain terhadap `quality` tergolong **lemah** (di bawah 0.3).
    - Walaupun pengaruhnya tidak besar secara langsung, fitur-fitur tersebut tetap **layak diuji efektivitasnya** dalam proses pemodelan, seperti regresi atau klasifikasi.

### 3. Analisis hubungan antar semua fitur numerik dalam dataset df_wine dalam bentuk scatter plot, dan juga histogram
<img src="https://github.com/user-attachments/assets/978ef0f4-e699-49f3-ab8b-be48831128a0" alt="Informasi Dataset" width="600">

Interpretasi:
### Tujuan Visualisasi Pairplot
Visualisasi pairplot digunakan untuk:
1. Melihat **pola hubungan antar dua fitur numerik** (scatter plot).
2. Menampilkan **distribusi masing-masing fitur** (histogram pada diagonal).
3. Mengidentifikasi **outlier** dan potensi **pola klaster** dalam data.

### Hasil Temuan
- Beberapa fitur memiliki distribusi yang **tidak normal (right-skewed)**:
  - `residual sugar`
  - `chlorides`  
  → Fitur-fitur ini perlu dipertimbangkan untuk **transformasi** (misalnya dengan **log transform**) sebelum modeling.

- Terlihat **pola linier positif** antara pasangan fitur tertentu:
  - `density` dengan `residual sugar`
  - `citric acid` dengan `fixed acidity`

- **Outlier** cukup jelas terlihat pada beberapa fitur:
  - `sulphates`
  - `chlorides`
  - `residual sugar`

### 4. Analisis distribusi skor kualitas wine dalam bentuk bar chart
<img src="https://github.com/user-attachments/assets/c67bae84-29eb-4064-8b9b-eecaae37c2680" alt="Informasi Dataset" width="600">

Interpretasi:
1. Distribusi Tidak Merata
Mayoritas wine dalam dataset memiliki skor **5 dan 6**,  
menunjukkan bahwa sebagian besar wine berada dalam **kategori kualitas sedang**.
2. Sedikit Wine Berkualitas Sangat Rendah atau Sangat Tinggi
Wine dengan skor **3 dan 8** relatif sangat sedikit dibanding skor lainnya.  
Hal ini menunjukkan bahwa wine dengan kualitas **sangat buruk atau sangat baik** jarang ditemukan dalam sampel.
3. Pola Distribusi
Distribusi `quality` secara umum **membentuk pola agak simetris**,  
dengan **puncak pada skor 5 dan 6**, kemudian menurun di skor yang lebih rendah atau lebih tinggi.
4. Implikasi
Fokus analisis atau peningkatan kualitas wine sebaiknya diarahkan pada **wine dengan skor 5 dan 6**,  
karena kelompok ini **mendominasi populasi** dan berpotensi untuk **ditingkatkan kualitasnya** melalui perbaikan proses atau bahan.

### 5. Analisis hubungan antara kadar alkohol dan kualitas wine
<img src="https://github.com/user-attachments/assets/db8b7214-8c73-447f-a69b-06b3ee423770" alt="Informasi Dataset" width="600">

Interpretasi:
1. Grafik menunjukkan distribusi jumlah wine berdasarkan kadar alkohol untuk berbagai skor kualitas.
2. Wine dengan kadar alkohol sekitar 9 hingga 10% memiliki jumlah paling banyak, terutama untuk wine dengan skor kualitas lebih rendah.
3. Wine dengan skor kualitas lebih tinggi cenderung memiliki kadar alkohol yang lebih tinggi dibandingkan wine dengan kualitas rendah.
4.cTerlihat bahwa semakin tinggi kadar alkohol, jumlah wine semakin sedikit, namun wine berkualitas bagus lebih sering ditemukan di kategori alkohol tinggi.
5. Secara umum, ada kecenderungan positif antara kadar alkohol dan skor kualitas: wine dengan alkohol lebih tinggi cenderung memiliki skor kualitas lebih baik.

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
