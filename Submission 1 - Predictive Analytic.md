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

<img src="https://github.com/user-attachments/assets/978ef0f4-e699-49f3-ab8b-be48831128a0" alt="Informasi Dataset" width="800">

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
Pada tahap ini kita akan melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Ada beberapa tahap persiapan data perlu dilakukan, yaitu:
1. Memisah fitur dan taarget
2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.
3. Scaling fitur numerik untuk StandardScaler
4. Melakukan Balancing data karena data tidak seimbang

### 1. Memisah fitur dan taarget
Pada tahap ini, dataset dipisahkan menjadi dua bagian: fitur (X) dan target (y).
- X berisi seluruh kolom kecuali kolom quality, karena kolom-kolom ini akan digunakan sebagai input untuk memprediksi kualitas wine.
- y adalah kolom quality yang berfungsi sebagai label atau target prediksi dalam model machine learning yang akan dibuat.
Langkah ini penting karena model supervised learning membutuhkan pemisahan yang jelas antara fitur (sebagai variabel independen) dan target (sebagai variabel dependen).

### 2. Pembagian dataset dengan fungsi train_test_split dari library sklearn.

<img src="https://github.com/user-attachments/assets/87fe4dbc-0ad6-4cff-8f94-4d42ee32a77c" alt="Informasi Dataset" width="600">

Dataset dibagi menjadi dua bagian yaitu 80% dan 20%: data latih yang memiliki data 914 dan data uji memiliki data 229 menggunakan  menggunakan train_test_split dari library sklearn

### 3. Scaling fitur numerik untuk StandardScaler
Pada tahap ini dilakukan standarisasi terhadap fitur-fitur numerik menggunakan StandardScaler dari sklearn.preprocessing.

Tujuan:
- Untuk menyamakan skala antar fitur, karena sebagian besar algoritma machine learning sensitif terhadap perbedaan skala antar variabel.
- StandardScaler mengubah data sehingga memiliki mean = 0 dan standar deviasi = 1.

Langkah:
- fit_transform() hanya dilakukan pada data training agar tidak terjadi data leakage (kebocoran informasi dari data uji).
- transform() kemudian digunakan pada data uji menggunakan parameter hasil training.
Fitur yang distandarisasi:
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
Dengan standarisasi ini, model dapat dilatih secara lebih stabil dan akurat.

### 4. Melakukan Balancing data karena data tidak seimbang
Pada tahap ini melakukan penanganan data tidak seimbang (imbalanced data) pada label target quality.

<img src="https://github.com/user-attachments/assets/f4a1bcfb-76b7-424d-b63a-f41c9acaa655" alt="Informasi Dataset" width="600">

**Masalah:** 
Distribusi label quality pada data training sangat tidak seimbang, di mana beberapa kelas (misalnya quality = 3, 4, dan 8) hanya memiliki sedikit sampel. Ketidakseimbangan ini dapat menyebabkan model bias terhadap kelas mayoritas.

**Solusi:** 
Digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk melakukan oversampling pada kelas minoritas secara sintetis. SMOTE bekerja dengan membuat sampel baru berdasarkan interpolasi antar tetangga terdekat di kelas minoritas.

**Langkah yang dilakukan**
- Tentukan parameter k_neighbors pada SMOTE dengan memperhitungkan jumlah sampel terkecil di kelas minoritas untuk menghindari error saat proses oversampling. Pastikan nilai k_neighbors tidak lebih besar dari jumlah sampel terkecil di kelas minoritas.
- Terapkan SMOTE hanya pada data training yang telah diskalakan (X_train_scaled) agar proses resampling tidak memengaruhi data uji (X_test), yang seharusnya tetap tidak dimanipulasi.
- Dengan langkah ini, semua kelas dalam data training akan memiliki jumlah sampel yang seimbang, memungkinkan model untuk belajar secara lebih adil terhadap setiap kelas tanpa bias terhadap kelas mayoritas.

## Model Development
Pada bagian ini, kita akan membangun 3 model machine learning untuk menguji sebarapa baik akurasi model, sehingga model tersebut yang disarankan untuk memprediksi prestasi siswa.

### 1. Model Development dengan Logistic Regression
salah satu algoritma klasifikasi yang umum digunakan untuk memprediksi variabel kategorikal. Meskipun sederhana, model ini cukup kuat dalam kasus klasifikasi multi-kelas jika datanya bersih dan terstandarisasi.

Logistic Regression bekerja dengan menghitung probabilitas setiap kelas menggunakan fungsi sigmoid (logistic function) dan memilih kelas dengan probabilitas tertinggi. Untuk klasifikasi multi-kelas, digunakan pendekatan multinomial.

Parameter yang digunakan:
- class_weight='balanced'
Digunakan agar model memberi bobot lebih besar pada kelas minoritas, membantu menangani ketidakseimbangan kelas.
- max_iter=5000
Jumlah maksimum iterasi ditingkatkan untuk memastikan model sempat konvergen, terutama karena jumlah kelas cukup banyak.
- solver='saga'
Optimizer saga dipilih karena mendukung penalti l1, l2, dan berskala baik untuk dataset besar atau sparse.
- random_state=42
Diset agar hasil eksperimen konsisten dan dapat direproduksi.

Alasan Pemilihan:
Logistic Regression dipilih sebagai baseline model karena:
- Cepat dan efisien untuk dijalankan.
- Hasilnya mudah diinterpretasikan.
- Memberikan performa awal yang bisa dibandingkan dengan model yang lebih kompleks.

### 2. Model Development dengan Support Vector Machine (SVM)
Merupakan algoritma klasifikasi yang efektif untuk kasus linear maupun non-linear. SVM bekerja dengan mencari hyperplane terbaik yang memisahkan kelas-kelas pada data. Ketika data tidak dapat dipisahkan secara linear, SVM dapat menggunakan teknik kernel untuk mentransformasikan data ke dimensi lebih tinggi.

SVM mencari hyperplane (garis batas) yang memaksimalkan margin antar kelas. Dengan kernel rbf, SVM dapat memetakan data ke dimensi lebih tinggi agar lebih mudah dipisahkan. SVM sangat berguna dalam klasifikasi dengan margin yang tegas.

Parameter yang digunakan:
- kernel='rbf'
Kernel radial digunakan untuk menangani data yang tidak dapat dipisahkan secara linear.
- C=1.0
Nilai default untuk parameter regularisasi; mengontrol kompleksitas model.
- class_weight='balanced'
Otomatis memberi bobot lebih besar pada kelas minoritas untuk mengatasi data yang tidak seimbang.
-probability=True
Diaktifkan agar model bisa memberikan probabilitas prediksi (berguna untuk visualisasi atau thresholding ke depan).

Alasan pemilihan: SVM dipilih karena:
- Mampu menangani klasifikasi non-linear dengan baik.
- Performa kuat pada dataset berskala kecil hingga menengah.
- Memberikan baseline pembanding yang berbeda dari Logistic Regression.

### 3. Model Development dengan Random Forest
sebuah algoritma ensemble learning berbasis pohon keputusan (Decision Tree). Random Forest bekerja dengan membangun banyak pohon keputusan secara acak lalu menggabungkan hasil prediksi dari masing-masing pohon untuk menentukan kelas akhir melalui proses voting.

Pendekatan ini membuat model:
- Lebih tahan terhadap overfitting dibanding pohon tunggal.
- Memiliki performa dan stabilitas lebih baik, terutama untuk dataset yang kompleks atau tidak linear.

Parameter yang digunakan:
- n_estimators=100 Jumlah pohon dalam hutan. Semakin banyak pohon, semakin stabil hasilnya, namun waktu komputasi juga bertambah.
- class_weight='balanced' (jika digunakan) Memberi bobot lebih pada kelas minoritas untuk menangani imbalance antar kelas.
- random_state=42 Agar hasil eksperimen bisa diulang (reproducible).

Alasan Pemilihan:
Random Forest dipilih karena:
- Kemampuan menangani data yang kompleks, baik linear maupun non-linear.
- Robust terhadap outlier dan noise.
- Tidak memerlukan scaling fitur, meskipun scaling tetap dilakukan dalam proyek ini untuk konsistensi dengan model lain.
- Memiliki fitur importance ranking, yang berguna dalam analisis lebih lanjut.

## Evaluasi Model
Pada proyek ini, penilaian model menggunakan confusion matrix, akurasi, dan f1 score sebagai metrik evaluasi untuk masing-masing model.
Penerapan Matriks Confusion, Akurasi, dan Skor f1 sebagai berikut:

#### 1. Model Development dengan  Logistic Regression
Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model * Logistic Regression*

<img src="https://github.com/user-attachments/assets/b0da796a-b8fb-4682-bd01-f91d29b2dbcf" alt="Informasi Dataset" width="600">

Interpretasi: 
- Akurasi model: Model Logistic Regression mencapai 42.36%, yang cukup rendah. akurasi ini menunjukkan bahwa model masih kesulitan memprediksi kualitas wine secara akurat, terutama pada kelas minoritas.
- F1-score tertimbang (weighted avg): F1-score tertimbang adalah 0.48, yang mengindikasikan bahwa secara keseluruhan, model masih menunjukkan keseimbangan precision dan recall yang cukup lemah. Terutama pada kelas dengan jumlah sampel kecil, model kesulitan memberikan prediksi yang baik.

#### 2. Model Model Development dengan SVM
Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *SVM*

<img src="https://github.com/user-attachments/assets/52c69864-24cd-4f84-b792-94f7be70f73f" alt="Informasi Dataset" width="600">

Interpretasi:
- Akurasi model: Model SVM dengan kernel linear mencapai akurasi 17.90%, yang sangat rendah. Akurasi ini menunjukkan bahwa model masih kesulitan memprediksi kelas dengan baik, terutama pada kelas minoritas yang jarang muncul dalam dataset.
- F1-score tertimbang (weighted avg): F1-score tertimbang adalah 0.23, yang mengindikasikan bahwa secara keseluruhan, model menunjukkan keseimbangan precision dan recall yang sangat lemah. Terutama pada kelas dengan jumlah sampel kecil, model kesulitan memberikan prediksi yang baik.

#### 3. Model Development dengan Random Forest
Berikut merupakan matriks confusion, akurasi, dan skor f1 dari model *Random Forest*

<img src="https://github.com/user-attachments/assets/8821b118-814a-41db-bea6-850b60b4850c" alt="Informasi Dataset" width="600">

Interpretasi: 
- Akurasi model sebesar 70%, menunjukkan bahwa model cukup baik dalam mengenali sebagian besar data uji, terutama pada kelas mayoritas (quality 5 dan 6).
- F1-score tertimbang (weighted avg): Weighted average cukup tinggi, karena mempertimbangkan jumlah data per kelas (dominan di kelas 5 dan 6).

### Hasil Evaluasi
Dari seluruh akurasi yang diketahui dari keempat model, dibentuk bar plot untuk melihat perbandingan nilai akurasi model sebagai berikut. 

<img src="https://github.com/user-attachments/assets/99e5e890-1a01-4ced-8b65-d22d121ed4b2" alt="Informasi Dataset" width="600">

Berdasarkan gambar di atas dan evaluasi masing-masing model untuk mengetahui skor akurasi, skor F1, dan jumlah kesalahan klasifikasi pada masing-masing model, didapat model *Random Forest* merupakan model terbaik karena memiliki skor akurasi dan skor F1 tertinggi.

## Kesimpulan
Setelah menguji data menggunakan 3 model *machine learning*, yaitu ***Logistic Regresion**, ***Support Vector Machine* (SVM)**, ***Random Forest*** untuk memprediksi performa kualitas wine, diperoleh:
- ***Random Forest*** menunjukkan performa terbaik dengan akurasi 69.87%, jauh lebih unggul dibandingkan model lainnya.
- ***Logistic Regression*** memiliki akurasi sedang sebesar 42.36%, namun masih jauh tertinggal dari Random Forest.
- ***SVM*** menunjukkan performa paling rendah dengan akurasi hanya 17.90%, yang menunjukkan bahwa model ini kurang cocok untuk dataset yang digunakan.
