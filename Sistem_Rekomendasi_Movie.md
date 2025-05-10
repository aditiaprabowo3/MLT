# Laporan Proyek Machine Learning - Aditia Prabowo

## Domain Proyek
Kemajuan teknologi, terutama dalam bidang perangkat bergerak dan internet, telah menciptakan peluang besar bagi pengembangan sistem rekomendasi. Pengguna kini dapat mengakses berbagai platform streaming seperti Netflix, Disney+, dan Amazon Prime yang menawarkan ribuan film. Namun hal ini juga menimbulkan tantangan baru untuk menemukan film yang sesuai dengan selera mereka di antara banyaknya pilihan yang tersedia `[5]`. Untuk mengatasi masalah ini, sistem rekomendasi film dikembangkan sebagai solusi efektif. istem rekomendasi berfungsi untuk memberikan rekomendasi yang dipersonalisasi berdasarkan interaksi pengguna sebelumnya dan preferensi yang diungkapkan, sehingga memudahkan mereka dalam menentukan pilihan `[6]`. Teknologi ini telah menjadi elemen penting dalam meningkatkan pengalaman pengguna di platform digital. 
Dari latar belakang itulah penulis mengambil topik ini sebagai studi kasus proyek machine learning yang penulis kerjakan dalam membangun sebuah model untuk proyek aplikasi yang sedang penulis kembangkan. Diharapkan model ini nantinya akan berguna untuk memberikan rekomendasi filim bagi pengguna sesuai dengan kebutuhannya.

## Business Understanding
Sistem rekomendasi film dirancang untuk memperkaya pengalaman pengguna dalam menemukan film yang sesuai dengan selera mereka. Dari perspektif bisnis, penting untuk memiliki pemahaman yang mendalam tentang apa yang diinginkan pengguna serta bagaimana sistem bisa memenuhi kebutuhan tersebut. Dengan semakin banyaknya konten film yang tersedia di berbagai platform streaming, pengguna seringkali merasa kesulitan dalam memilih film yang ingin ditonton. Hal ini menciptakan kebutuhan akan sistem rekomendasi yang efisien, yang dapat membantu pengguna menemukan film berdasarkan preferensi pribadi mereka.

### Problem Statement
Berdasarkan latar belakang diatas, proyek ini berfokus pada beberapa masalah utama yang perlu dipecahkan:
* Bagaimana cara melakukan pengolahan data yang baik sehingga dapat digunakan untuk membangun model sistem rekomendasi yang efektif?
* Bagaimana memberikan rekomendasi bagi pengguna yang yang memiliki kesaaman pola dengan film yang disukai?
* Bagaimana cara membangun model machine learning yang mampu merekomendasikan film berdasarkan preferensi pengguna?

### Goal
Tujuan dibuat proyek ini adalah sebagai barikut:
* Melakukan pengolahan data secara efisien agar dapat digunakan dalam pembangunan model sistem rekomendasi.
* Membungun model rekomendasi bagi pengguna yang yang memiliki kesaaman pola dengan film yang disukai.
* Membangun model machine learning yang dapat memberikan rekomendasi film dengan tingkat akurasi tinggi.

### Solution Statement
Dalam proyek ini, untuk mengatasi asalah diatas, digunakan teknik analisis data dan metode machine learning yaitu:
* Menggunakan Teknik Univariate Exploratory Data Analysis (EDA) dan Preparation Data untuk proses pengolahan data yang efektif dan efisien.
* Menggunakan Model Content-Based Filtering untuk merekomendasikan film berdasarkan kemiripan film berdasarkan perilaku pengguna.
* Mengunakan model Model-Based Deep Learning Collaborative Filtering meberikan rekomendasi dengan tingkat akurasi yang tinggi.

Model flowchart sistem yang diususkan untuk kedua model diatas yakni:

1. Flowchart Content-Based Filtering (CBF)

![cbf-CBF drawio (1)](https://github.com/user-attachments/assets/2d9b916c-263a-4c5b-8850-3f926680b3d9)

2. Deep Learning Collaborative Filtering (CF)

![cbf-CF drawio](https://github.com/user-attachments/assets/e70d8e81-1201-4897-b016-4a5f091471a9)


## Data Understanding
Data understanding dalam proyek sistem rekomendasi film melibatkan pengumpulan, analisis, dan pemahaman tentang data yang akan digunakan untuk membangun model rekomendasi. Berikut adalah beberapa aspek penting dari data understanding dalam konteks ini.

### Informasi Dataset
Dataset yang digunakan yaitu The Movies Dataset. Informasi dari dataset film ini dapata dilihat pada tabel berikut:
| Jenis      | Keterangan     |
| -----------------------     | ------------------------------------------------------------------------- |
| Sumber                      | Dataset: [Kaggle](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset)  |
| Dataset Owner               | Manas Garg		                        |
| Kategori                    | Movies & TV Shows                         |
| Jenis dan Ukuran Berkas     | ZIP Version 7 (866 KB)                 |
| Jumlah File Dataset         | 2 File (CSV)                              |

Dari informasi tabel di atas, dapat dilihat bahwa file-file dalam dataset ini berisi metadata untuk seluruh film yang tercantum dalam Movie Recommender Dataset. Dataset ini mencakup film-film yang sduah tayang. Poin data yang tersedia mencakup informasi seperti moveId, taitlel, genre. Dataset ini memberikan gambaran menyeluruh tentang film-film yang ada dalam database, yang dapat digunakan untuk analisis lebih lanjut terkait rekomendasi film dan preferensi pengguna.

Kumpulan data ini juga memiliki file yang berisi 100836 rating dari 610 pengguna untuk seluruh 9742 film. Ratingnya dalam skala 1-5 dan diperoleh dari situs resmi GroupLens. Berikut penjelasan dari penjelasan file-file dalam kumpulan data tersebut.

### Membaca Dataset 
Selanjutnya pada tahap ini dataset diatas akan disimpan pada variabel menggunakan fungsi pandas.read_csv. Hasilnya dapat ditampilkan pada gambar berikut:

**Movies Dataset**

![Screenshot (1642)](https://github.com/user-attachments/assets/7a318ed8-e08d-48f6-92f1-3b5cd9cf43d6)

Dalam gambar di atas kita dapat mengetahui bahwa data movies terdiri dari 3 kolom, Kolom-kolom tersebut antara lain:

- movieId: ID unik untuk setiap film, yang digunakan juga dalam data rating (df_ratings).
- title: Judul film beserta tahun rilis.
- genres: Genre film yang dipisahkan dengan tanda pipe (|), misalnya: Comedy|Romance.

**Ratings Dataset**

![Screenshot (1643)](https://github.com/user-attachments/assets/65b19ebf-6845-47e5-a9e8-f49bdd1a6ab2)

Dari gambar di atas, kita dapat mengetahui bahwa data ratings terdiri dari 4 kolom. Kolom-kolom tersebut antara lain:

- userId: ID pengguna yang memberikan rating.
- movieId: ID film (bisa di-join dengan df_movies).
- rating: Nilai rating yang diberikan (skala 0.5 – 5.0).
- timestamp: Waktu pemberian rating dalam format Unix Epoch (detik sejak 1 Jan 1970).

### Exploratory Data Analysis - EDA

Analisis eksploratif data (EDA) adalah tahap penting dalam analisis data yang bertujuan untuk memahami dan mengeksplorasi karakteristik dataset sebelum melakukan analisis yang lebih mendalam. Dataset yang digunakan dalam proyek ini yaitu dataset fIlm yang dapat dijelaskan sebagai barikut: 

**1. Univariate Analysis**

Pada tahap ini, akan dialkukan analisis statistik yang melibatkan variabel movies dan ratings untuk melihat distribusi, pola, dan karakteristik data tersebut serta menggunakan grafik untuk menggambarkan distribusi genre dan rating film, serta hubungan antara fitur-fitur dalam dataset.

* **Deskripsi Variabel**
  
    Pada tahap ini akan ditampilkan deskripsi variabel yang digunakan yaitu:
  
    *a. Movies*
  
    Deskripsi variabel ratings dapat dilihat pada gambari berikut:
  
    ![b](https://github.com/user-attachments/assets/087cac81-788a-4aa1-ae81-a9920305087d)

    Dataset movies terdiri dari 9.742 baris dan 3 kolom yang dapat dijelaskan sebagai berikut:

    | Variabel                    | Keterangan     |
    | -----------------------     | ------------------------------------------------------------------------- |
    | movieId|ID unik untuk film yang dinilai oleh pengguna untuk mendapatkan informasi lebih detail tentang film tersebut.                                  |
    | title | Judul dari film yang ada dalam dataset |
    | genre                 | Daftar genre film, seperti Action, Comedy, Drama. Biasanya berupa JSON atau daftar string   |
    
    *b. ratings *
  
    Deskripsi variabel movies dapat dilihat pada gambari berikut:

    ![tipe_data_movies ](https://github.com/user-attachments/assets/2c2af7b3-e319-4841-8850-092f4cc242da)
  
    Berdasarkan gambar diatas, dataset movies terdiri dari 100004 baris dan 24 kolom yang dapat dijelaskan sebagai berikut:

    | Variabel                    | Keterangan     |
    | -----------------------     | ------------------------------------------------------------------------- |
    | userId        | ID unik untuk pengguna yang memberikan penilaian (rating). Ini digunakan untuk mengidentifikasi pengguna secara anonim.|
    |movieId|ID unik untuk film yang dinilai oleh pengguna untuk mendapatkan informasi lebih detail tentang film tersebut|
    |rating|Nilai yang diberikan oleh pengguna untuk film tertentu dengan skala 1 hingga 5, di mana angka yang lebih tinggi menunjukkan penilaian yang lebih positif.|
    |timestamp|Waktu ketika penilaian diberikan, direpresentasikan dalam format UNIX timestamp|

* **Melihat Informasi Tipe Data**
  
    Infromasi dari tipe data variabel dapat dilihat pada gambar berikut:

    *1. Variable movies*
  
   ![tipe_data_movies ](https://github.com/user-attachments/assets/1101b319-1b88-46d1-94a4-cbec4c8a0588)
  
    *2. Variabel ratings*

   ![tipe_data_rating](https://github.com/user-attachments/assets/858e07b6-86e0-4fb6-9e98-cd7d0adcb46c)
  
    Dapat dilihat pada informasi dataset **movies** terdiri dari 2 variabel dengan tipe data object dan 1 variabel dengan tipe data int64, sedangkan untuk informasi dataset **ratings** terdiri dari 3 variabel dengan tipe data int64 dan 1 variabel dengan tipe data float64

* **Menghitung Total Dataset**
  
![total datset](https://github.com/user-attachments/assets/a829b1e9-dad3-46cd-92be-d0de40052aef)

Pada tahap ini, jumlah variabel dataset movies sebanyak 9742 dan memiliki 3 kolom sedangkan jumlah variabel dataset ratings sebanyak 100836 dan memiliki 4 kolom.

* **Menghitung Total Data Unik**
  
    Jumlah rincian data unik dapat dilihat pada gambar berikut:

    ![dataset unik](https://github.com/user-attachments/assets/5c92e544-da79-45cd-93d5-c1d4e548f8ae)

    Dari hasil diatas terdapat 9742 film pada dataset movies, 9724 film pada dataset ratings dan 610 user pada dataset ratings

* **Pengecekan *Outliers***
  
    Pada tahap ini, akan diperlihatkan statistik deskriptif dari dataset variabel df_movies dan ratings mengunakan fungsi `describe()`.

    *1. Variabel movies*
  
    ![movie describe](https://github.com/user-attachments/assets/7998eaa1-cac1-4545-9f11-9ed5d36a2725)

    *2. Variabel ratings*

    ![ratings describe](https://github.com/user-attachments/assets/14f0e2ff-95f4-4f2e-8388-619016104a4e)

    Berdasarkan tampilan deskriptif dataset movies dan ratings dapat dilihat tidak mencolok ada pesebaran nilai yang menimbulkan `outlier`.

* **Distribusi Ratings**
  
    Langkah ini bertujuan untuk:
    * Mengidentifikasi nilai rating yang paling umum diberikan oleh pengguna.
    * Menilai apakah data rating cenderung condong ke satu nilai (misalnya, lebih banyak rating tinggi atau rendah).
    * Membantu memahami pola preferensi pengguna.

    Tampilan distribusi rating dapat dilihat pada gambar berikut:

    ![grafik ratings](https://github.com/user-attachments/assets/13b9ea09-3521-4bd2-b6c7-ee6fa1f3921d)
  
    Berdasarkan diagram plot rating diatas, dapat dilihat bahwa nilai ratings paling umum diberikan pengguna adalah rating 4.0 dengan presentasi 286.6%, rating 3.0 dengan presentasi 19.9%, rating 5.0 dengan prestansi 13.1%. rating 3.5 dengan presentasi 13.0%, sedangkan nilai rating yang lain berada di bawah pada presentasi 12.0%

* **Distribusi Gengre**
  
    Distribusi genre film adalah aspek penting dalam sistem rekomendasi, karena membantu memahami preferensi pengguna dan pola konsumsi film. Pada proyek ini menggunakan metode visualisasi Data dalam menampilkan grafik batang yang menggambarkan proporsi masing-masing genre secara visual, sehingga memudahkan pemahaman. Pada tahap ini akan dilakukan membersihkan, memproses, dan menormalkan data dalam kolom genres pada DataFrame df_movies Ada beberapa fungsi yang dipakai yakni:
    * `fillna('')`, berfungsi untuk mengisi nilai null atau NaN dalam kolom genres dengan string kosong ('').
    * `apply(lambda x: x.split('|') if x else [])`, fungsi lambda ini memisahkan string genre berdasarkan tanda | menjadi list. Jika datanya kosong, akan dikembalikan list kosong [].
    
    Selanjutanya, ubah setiap elemen dalam daftar (genre) menjadi baris terpisah dengan fungsi `explode()`, kemudian menghitung jumlah kemunculan setiap genre dengan fungsi `value_counts()` dan terakhir membuat diagram batang untuk menampilkan distribusi genre dengan plot bar `plot(kind='bar')`. Langkah pertama, buat variabel dataframe baru untuk melakukan analisis visualisasi data. Kemudian konversi fitur(variabel) genres ke dalam bentuk list sehingga dapat dianalisi. Berikut adalah gambar distribrusi genres menggunakan grafik bar.

    ![download](https://github.com/user-attachments/assets/3b79aa72-cb7e-483e-b572-8293e3d899a6)

    Dari grafik diatas, dapat dilihat bahwa genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar 4361 dan 3756. Sedangkan genre yang jumlahnya dibawahnya ada Thriller dengan 1894,Action dengan 1828,Romance dengan 1596,Adventure dengan 1263,Crime dengan 1199, selain itu 12 yang lain di bawah 1000

* **Analisis Rating Tertinggi**
  
    Selanjutnya gabungkan dataset df_movies dan ratings dengan fungsi pandas `pd.merge` dan mencari 10 film dengan rating tertinggi. Alisis rating tertinggi dapat dilihat pada gambar berikut:

    ![rating hight](https://github.com/user-attachments/assets/4797880a-d123-4ea9-b9d2-a3f88b8f7ab2)

    Dapat dilihat pada gambar diatas dari 10 rating tertinggi film yang ada, film dengan judul Forrest Gump memiliki rating teratas dengan mean rating 4.164 dan total rating sebanyak 329

* **Membandingkan Peringkat rata-rata vs Jumlah total peringkat**
  
    Pada tahap ini akan dibandingkan rata-rata rangkin dan total rangking menggunakan `joinplot` untuk melihat pesebaran data yang dapat dilihat pada gambari dibawah ini:

    ![cihuy pisan](https://github.com/user-attachments/assets/d8530e60-98b8-43d6-ab23-fd545f4a647c)
  
    Berdasarkan grafik pesebaran data diatas total rating tertinggi berada diatas 250 sebanyak 5 film, sadangkan rata-rata terbanyak pengguna memberi rating terhadap film berada diretang nilai 2 - 4.5 rating.
  
## Data Preparation

Data preparation adalah langkah penting dalam pengembangan sistem rekomendasi film yang efektif. Proses ini mencakup beberapa tahap, mulai dari pengumpulan data hingga pemrosesan akhir sebelum data digunakan dalam model machine learning. 

###  Data Cleaning

Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Setelah data terkumpul ada beberapa langkah yang perlu lakukan dalam tahap ini yaitu:

* **Menangani Nilai Kosong (Missing Value)**
  
    Pada tahap ini, akan dilakukan pengecekan nilai kosong serta menanganinya pada variabel dataset movies dan ratings. Hasilnya dapat dilihat pada gambar berikut:

    ![clean movie](https://github.com/user-attachments/assets/6873aa9a-c63b-4439-b66e-469675996248)

    Dari gambar diatas, tidak ada data yang bernilai null sehingga aman untuk proses selanjutnya
  
    ![ratings clean](https://github.com/user-attachments/assets/854f61dd-283c-4927-b94b-e26782233f17)
  
    Dari hasil diatas, tidak ada data yg bernilai null sehingga aman untuk proses selanjutnya

* **Menangani Duplikat Data (Duplicated Data)**
  
    Pada tahap ini, akan dilakukan pengecekan data ganda serta menanganinya pada variabel dataset movies dan ratings. Setalah dilakukan pengecekan kedua data juga hasil yang ditampilkan yakni 0 maka tidak ada duplikat data pada dataset movies dan ratings
  
### Data Preprocessing

Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Langkah-langkah yang dilakukan dalam proyak ini yaitu:

* **Mengurutkan Pengguna dan Film berdasarkan ID**
  
    Pada tahap ini, akan dilakukan pengurutan data berdasarkan userId pada variabel dataset ratings dan movieId pada variabel dataset df_movies. Hasilnya dapat dilihat pada gambar berikut:

  * Pengurutan data berdasarkan userId pada dataset ratings

  ![encang](https://github.com/user-attachments/assets/2313653e-52c5-443d-aac0-63768f604cc8)

  * Pengurutan data berdasarkan movieId pada dataset df_movies
    
    ![urutkan data](https://github.com/user-attachments/assets/42358e8d-06d4-440c-af0b-77abe2e6075c)

* **Mengubah fitur genre movie ke bentuk list**
* 
    Pada tahap ini, fitur genres pada variabel dataset df_movies masih dalam bentuk format json, maka perlu diubah kedalam bentuk list sehingga dapat dilakukan dalam proses pelatihan model. Hasilnya dapat dilihat pada gambar berikut:

   ![genre movie](https://github.com/user-attachments/assets/0c91891e-5d37-4b1e-84cb-753b741d31d1)

* **Melakukan penggabungan dataset df_movies dan ratings**
   
    Selanjutnya akan dilakukan penggabungan dataset variabel df_movies dan ratings menggunkan fungsi `inner` melalui fitur movieId. Hasilnya dapat dilihat pada gambar berikut:

    ![merge data](https://github.com/user-attachments/assets/9c14432c-3e7c-47d0-aa17-c07a77a78586)
    
* **Menghapus fitur yang tidak diperlukan**
  
    Langkah selanjutnya, melakukan penghapusan fitur-titru yang tidak diperlukan dalam proses pelathan model nanti yakni fitur *timestamp*. Hasilnya dapat dilihat pada gambar berikut:
  
    ![drop fitur](https://github.com/user-attachments/assets/9f6ecb2d-bee5-40be-a1b2-98f006b31f68)

    Hasil gambar diatas menunjukan 10 data gabungan movies dan ratings, dimana fitur genre sudah menjadi bentuk list dan diurutkan berdasarkan *userId*.

* **Mengambil 20000 dataset secara acak**
  
    Selanjutnya diambil 20000 gabugan dataset variabel df_movies dan ratings menggunakan fungsi `shuffle` dari library `sklearn.utils` untuk memperoleh data secara acak dengan tujuan mempermudah pengolahan dan mencegah crash. Hasilnya dapat dilihat pada gambar berikut:

   ![mengambil dataset](https://github.com/user-attachments/assets/43e3810a-0750-464f-9db9-c69fa506d051)
  
    Pada gambar diatas terdapat 20000 baris dan 5 kolom yakni *userId*, *movieId*, *rating*, *title* dan *genres*.

### Content-Based Filtering

Content-Based Filtering adalah metode dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item yang telah disukai atau dinilai oleh pengguna. Teknik yang digunakan yaitu teknik `TF-IDF` (Term Frequency-Inverse Document Frequency) untuk menentukan bobot fitur dan menghitung kesamaan antara item dalam hal ini adalah `genres`.

Berikutnya, lanjutkan ke tahap persiapan dengan membuat variabel preparation yang berisi dataframe df_sample_final yang dapat dilihat pada gambar berikut ini:

![content base](https://github.com/user-attachments/assets/74299d24-faff-41b6-8392-0051ac1899a7)

Kemudian mengurutkan berdasarkan movieId. Hasilnya dpat dilihat pada gambar berikut:

![urut movie](https://github.com/user-attachments/assets/68a0c2f2-ba2c-4dc9-a311-aefe5c3bf13a)

Selanjutnya, lakukan konversi data series menjadi list. Dalam hal ini, menggunakan fungsi `tolist()` dari library `numpy`. Setelah konversi dilakukan diperoleh variabel `movieId`, `movie_name`, `movie_genres` dengan jumlah masing-masing sebanyak 5189.

Tahap terakhir, membuat dictionary untuk menentukan pasangan `key-value` pada data `movie_id`, `movie_name` dan `movie_genres` yang telah siapkan sebelumnya. Hasilnya dapat dilihat pada gambar berikut:

![dictionory](https://github.com/user-attachments/assets/5d1612ce-83ed-40a0-a91c-b4c281397426)

Selanjutnya, gunakan fungsi `TfidfVectorizer` untuk mengkonversi `genres`. Namun sebelum itu genres perlu dikonversi dari list ke string akar dapay diproses.

![list to string](https://github.com/user-attachments/assets/5c38854e-3846-4166-b9d5-66675fd0e283)

Setelah mendapat index seluruh genre film, akan di fit lalu ditransformasikan ke bentuk matriks sehingga diperoleh ukuran (5189, 24) serta mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense(). Hasilnya dapat dilihat pada gambar berikut:

![todense](https://github.com/user-attachments/assets/a911f0b5-bfb9-4624-a251-1d7300a10e6f)

Setelah dibentuk matriks, dibuat tabel berisi judul film beserta genrenya berdasarkan TF-IDF yang telah diinisiasi. Hasilnya dapat dilihat pada gambar berikut:

![judul film](https://github.com/user-attachments/assets/3b4510e1-3907-4b40-af4d-bf356e736455)

### Collaborative Filtering (CF)
Pada tahap ini data prerataion CF, Langkah pertama, cek dataset dengan fungsi `info()`, hasilnya ditampilkan pada gambar berikut:

![colaborative](https://github.com/user-attachments/assets/d609a3d4-b9e0-464f-9644-05c3cde9d103)

Dari hasil diatas, terdapat 20000 baris dan 5 kolom dan memiliki 1 tipe data float64, 2 tipe data int64 dan 2 tipe data object. Langkah kedua Kedua, hapus kolom yang tidak dibutuhkan dalam pelatihan yaitu `genres` dan `title`. Langkah berikutnya, urutkan berdasarkan kolom `userId` untuk  masuk pada tahap encoding `userId` dan `movieId`.

* **Encoding userId dan movieId**
  
    Pada tahap ini, akan dilakukan encoding pada `userId` dan `movieId`. Hasilnya dapat ditampilkan pada gamabr dibwah ini:
    1. Encoding *userId*
       
    ![encode](https://github.com/user-attachments/assets/78df2888-83d0-4673-9a29-4d73996ceb73)

    2. Encoding *movieId*

    ![encode movie](https://github.com/user-attachments/assets/b6bbce41-7e45-4d05-a4fa-0e55d2ee9e99)

    Selanjutnya ambil total_user, total movie dan nilai rating minimum dan maksimum untuk proses pembagian dataset sebelum melakukan pelatihan model. Hasilnya diperoleh yaitu 610 pengguna, 5189 film serta nilai rating minimum sebesar 0.5 dan maksimum sebesar 5.0.

* **Membagi Data untuk Training dan Validasi**
  
    Pada tahap ini, data training dan data validasi dibagi untuk proses pelatihan model. Namun sebelum itu, perlu mengacak dataset sehingga menjadi data yang valid. Hasilnya seperti pada gambar berikut:

    ![Training dan Validasi](https://github.com/user-attachments/assets/e25958d4-4a8a-4574-b2c8-164e725f773f)    

    Selanjutnya, buat variabel x untuk mencocokkan data user dan Movie menjadi satu value, kemudian variabel y untuk membuat rating dari hasil. Terakhir,  bagi menjadi `80%` data train dan `20%`` data validasi.

## Modeling and Result

Pada tahap ini ada dua model yang dipakai untuk dilatih, di evaluasi dan memberikan rekomendasi kepada pengguna film. Kedua model tersebut dapat dijelaskan sebagai berikut:

### Modeling Content-Based Filtering (CBF)

Pada proyek ini, metode yang digunakan adalah `Consine Similarity`,  yang berfungsi mengukur kesamaan antara dua dokumen atau vektor dalam ruang multidimensi. Metode ini digunakan untuk sistem rekomendasi berbasis `Content-Based Filtering` yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item genre film yang telah disukai atau dinilai oleh pengguna. Menurut Firmansyah(2018), `Cosine similarity` digunakan dalam ruang positif, dimana hasilnya dibatasi antara nilai `0` dan `1`. Kalau nilainya `0` maka dokumen tersebut dikatakan mirip jika hasilnya 1 maka nilai tersebut dikatakan tidak mirip Perhatikan bahwa batas ini berlaku untuk sejumlah dimensi.

Langkah pertama hitung `cosine similarity` pada matrix `tf-idf` yang dapat dilihat pada gambar berikut:

![cosine similarity](https://github.com/user-attachments/assets/34858bd1-62c1-42d7-9397-efa6a59b748a)

Langkah kedua, lihat hasil cosine similarity pada matrix tf-idf antar judul film yang mirip berdasarkan genre.

![result cosine similarity](https://github.com/user-attachments/assets/115a17c2-9b89-44ca-90a9-c20d46c8265e)

Selanjutnya, buat fungsi rekomendasi film berdasarkan kemiripan genre dengan menerapkan fungsi Top-N rekokemendasi serta menguji dan mengevaluasi model yang dibuat.

### Pengujian Sistem Rekomendasi

Pada proses pengujian akan diambil satu judul film untuk dilakukan pengujian seperti yang terlihat pada gambar berikut:

![uy](https://github.com/user-attachments/assets/48c0f5de-a01f-464b-8dfe-74bbcf9f72b6)

10 hasil rekomendasi film dapat dilihat pada gambar berikut:

![oko](https://github.com/user-attachments/assets/385c01ec-db4a-437a-b1e8-5f2bb29210f4)

Dapat dilihat genre film uji yang dimasukan adalah `Adventure`, `Children`, `Fantasy`. Hasilnya genre ini tersebar di dalam 10 judul film yang memiliki kesaaman genre.

### Modeling Collaborative Filtering (CF)

Pada tahap ini menggunakan pendekatan Model-Based Deep Learning Collaborative Filtering. Metode `Deep Learning Neural Network (DNN)` yang merupakan subkategori dari machine learning yang menggunakan struktur ANN yang sangat dalam, dikenal sebagai deep neural networks. Deep learning melibatkan jaringan saraf dengan banyak lapisan tersembunyi, yang memungkinkan model untuk belajar dan mengenali pola yang sangat kompleks dan abstrak dari data `[2]`.

Pada tahap ini, model menghitung skor kecocokan antara user dan movie teknik embedding. Pertama, dilakukan proses embedding terhadap data user dan movie. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan movie. Selain itu, ditambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [`0,1`] dengan fungsi aktivasi sigmoid. Di sini, dibuatkan class `RecommenderNe`t dengan `keras Model class`. Kedua, lakukan proses compile terhadap model. Model ini menggunakan `Binary Crossentropy` untuk menghitung `loss function`, `Adam (Adaptive Moment Estimation)` sebagai `optimizer`, serta Mean Absolute Error(MAE) dan Root Mean Squared Error (RMSE) sebagai metrics evaluation.

Langkah berikutnya, mulailah proses training. Pada proses ini menggunakan fungsi `callbacks`, dimana jika kinerja model tidak mengalami keanaikan maka pelatiahan dihentikan. Pada proses training parameter yang digunakan yakni `batch_size=8`, `epoch = 50`, `shuffle = True` dan `verbose=1`

Proses latihan model dapat dilihat pada gambar berikut:

![op lur](https://github.com/user-attachments/assets/4d407fda-6372-4e66-a7f1-df47009406ef)

Dapat dilihat, hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1653 dan root_mean_squared_error: 0.2101

### Penujian Sistem Rekomendasi

Proses pengujian sistem dilakukan berdasarkan Top-10 Rekomendasi film terbaik kepada pengguna yang memiliki kesamaan:

![iya](https://github.com/user-attachments/assets/5eedc572-885d-4394-8788-5ad4e8b423a3)

## Evaluation

### Evaluation Content-Based Filtering (CBF)

Evaluasi CBF merupakan proses untuk menilai atau mengevaluasi kinerja sistem rekomendasi berbasis konten dalam memberikan rekomendasi yang relevan kepada pengguna. Pada proyek ini sistem rekomendasi berbasis konten (CBF) bekerja dengan cara menganalisis atribut rating dari judul film yang disukai oleh pengguna dan kemudian merekomendasikan judul film lain dengan atribut rating yang serupa. Tujuan evaluasi ini adalah untuk mengukur seberapa baik sistem dapat memberikan rekomendasi yang tepat dan memuaskan berdasarkan preferensi atau interaksi pengguna sebelumnya. Fungsi yang digunakan untuk mengukur CBF yakni metriks presisi. Metrik presisi mengungkapkan berapa banyak kelas yang diprediksi diberi label dengan benar (posistif) [7].

Berikut adalah rumus dari metriks presisi[7]:

`Precision = True_Positive / (True_Positive + False_Positive)`

Keterangan:

* Precision = Hasil Presisi
* True_Positive = Prediksi benar
* False_Positive = Prediksi salah

Hasil pengujian menggunakan atribut genre {"Adventure", "Children", "Fantasy"} dengan 10 rekomendasi film yakni:

| | Title                   |	Genres                          | Hasil Presisis  |
|-| ----------              | ----------                      | -----------  |
|0|	Chronicles of Narnia (2004)	      | Adventure, Children, Fantasy           | TRUE   |
|1|	Bridge to Terabithia (2007)	                | Adventure, Children, Fantasy          | TRUE   |
|2|	Escape to Witch Mountain (1975)	|	Adventure, Children, Fantasy | TRUE   |
|3|	Pete's Dragon (2016) |	Adventure, Children, Fantasy          | TRUE   |
|4|	Jumanji (1995)             | Adventure, Children, Fantasy         | TRUE   |
|5|	NeverEnding Story III, The (1994)                 |	Adventure, Children, Fantasy          | TRUE   |
|6|	Harry Potter and the Sorcerer's Stone (1997)              |	Adventure, Children, Fantasy           | TRUE   |
|7|	Return to Oz (1985)        |	Adventure, Children, Fantasy   | TRUE   |
|8|	Water Horse: Legend of the Deep, The (2007) |	Adventure, Children, Fantasy   | TRUE   |
|9| Golden Compass, The (2007)              |	Adventure, Children, Fantasy | TRUE   |

diperoleh nilai _True_Positive_ = 10 , _False_Positive_ = 0. Jika dimasukan dalam rumus metriks maka nilai precision 10/10 = 1 atau 100.00%.

### Evaluation Collaborative Filtering (CF)

Pada tahap ini akan digunakan metrik evaluasi  untuk mengukur kinerja model (formula dan cara metrik tersebut bekerja) serta visualisasi metrik dengan teknik Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). 

1. Mean Absolute Error (MAE)
MAE adalah salah satu metode evaluasi yang umum digunakan dalam data science. MAE menghitung rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual. Dengan kata lain, MAE menghitung berapa rata-rata kesalahan absolut dalam prediksi. Semakin kecil nilai MAE, semakin baik kualitas model tersebut `[3]`.

Rumus MAE:
   
![MAE](https://github.com/user-attachments/assets/0a342837-503b-487a-9dd7-b9f205dc1185)

Dimana:
* n adalah jumlah sampel dalam data
* y_i adalah nilai aktual
* ŷ_i adalah nilai prediksi
    
2. Root Mean Squared Error (RMSE)
RMSE adalah turunan dari MSE. Seperti namanya, RMSE adalah akar kuadrat dari MSE. RMSE menghitung rata-rata dari selisih kuadrat antara nilai prediksi dan nilai aktual kemudian diambil akar kuadratnya. Semakin kecil nilai RMSE, semakin baik kualitas model tersebut `[3]`.

Rumus RMSE:
   
![RMSE](https://github.com/user-attachments/assets/058da06d-8a58-4cca-b01b-468cdcf4e0e4)

Dimana:
* n adalah jumlah sampel dalam data
* y_i adalah nilai aktual
* ŷ_i adalah nilai prediksi
  
Selanjutnya dilakukan visualisasi metrik seperti Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). Kedua metrik ini sangat penting dalam mengevaluasi kinerja model prediksi. Kedua metrik ini memberikan informasi tentang seberapa baik model dapat memprediksi nilai aktual, dan visualisasi dapat membantu dalam memahami perbandingan antara keduanya serta tren kesalahan dari waktu ke waktu `[3]`. 

Hasil dari kedua metiks tersebut dapat ditampilakn pada gambar dibawah ini:

* Gambar Visualisasi Metriks MAE

![mean](https://github.com/user-attachments/assets/b39ac416-efed-4ecf-a1f4-5e775fb0c596)

Berdasarkan hasil `fitting` nilai konvergen metrik MAE berada sedikit di bawah 0.123 untuk training, dan sedikit di bawah 0.165 untuk validasi.

* Gambar Visualisasi Metriks RMSE
  
  ![root](https://github.com/user-attachments/assets/9354f5ba-6821-4e0b-a840-90673b778284)

Berdasarkan hasil fitting Nilai konvergen metrik RMSE berada sedikit di bawah 0.160 untuk training, dan sekitar 0.210 untuk validasi.

## Kesimpulan 

Berdasarkan hasil yang diperoleh setelah melakukan proses pengolahan data sampai proses evaaluasi dapat dismpulkan bahwah:

1. Pengunaan Teknik EDA dapat melihat distribusi data pada data rating dan data genre film dengan jelas. Nilai ratings paling umum diberikan pengguna adalah adalah rating 4.0 dengan presentasi 286.6%, rating 3.0 dengan presentasi 19.9%, rating 5.0 dengan prestansi 13.1%. rating 3.5 dengan presentasi 13.0%, sedangkan nilai rating yang lain berada di bawah pada presentasi 12.0%. Sedangkan genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar `4361` dan `3756`,  Sedangkan genre yang jumlahnya dibawahnya ada Thriller dengan 1894,Action dengan 1828,Romance dengan 1596,Adventure dengan 1263,Crime dengan 1199, selain itu 12 yang lain di bawah 1000. Film dengan judul Forrest Gump memiliki rating teratas dengan mean rating 4.164 dan total rating sebanyak 329.
   
2. Dengan preparation data yang sistematis, seperti menangani nilai hilang (missing values), menghapus atau menangani outlier, dan melakukan encoding pada data kategorikal, proses analisis data menjadi lebih efisien dan akurat. Data yang bersih dan siap digunakan akan mengurangi risiko kesalahan dalam model analitik.
   
3. Dengan mengunakan metode Content-Based Filtering dapat memberikan 10 rekomendaasi film kepada sesama pengguna berdasarkan kesaaman perilaku pengguna dengan nilai presesion matriks sebesar 100.00%.
 
4. Penggunaan Model-Based Deep Learning Collaborative Filtering memberikan hasil rekomendasi yang lebih akurat dan relevan bagi pengguna. Hal ini di buktikan dngan hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1692 dan root_mean_squared_error: 0.2145 dan juga tampilan matriks visualisasi yang menunjukan nilai MAE dan RMSE berada dibawah 0.123 pada epoh ke-19.

## Daftar Pustaka

1. D. A. R. Ariantini, A. S. M. Lumenta and A.Jacobus, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016.
   
2. Neural Network: Cikal Bakal Revolusi Deep Learning. Tersedia: [Tautan](https://www.dicoding.com/blog/neural-network-cikal-bakal-revolusi-deep-learning/). Diakses pada: mei 2025.
   
3. Perbedaan MAE, MSE, RMSE, dan MAPE pada Data Science. Tersedia: [Tautan]([https://pages.github.com/](https://www.trivusi.web.id/2023/03/perbedaan-mae-mse-rmse-dan-mape.html)). Diakses pada: Mei 2025.
   
4. Firmansyah Fataruba, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016. "PENERAPAN METODE COSINE SIMILARITY UNTUK PENGECEKAN KEMIRIPAN JAWABAN UJIAN SISWA", JATI (Jurnal Mahasiswa Teknik Informatika) Vol. 2  No. 2, September 2018.
 
5. Nathania, R.A. 2024. Sistem Rekomendasi Film Dengan Collaborative Deep Learning. (Skripsi, Fakultas Teknologi Informasi dan Sains, Universitas Katolik Parahyangan: Bandung).
   
6. Salim .E, Paragantha. J, Lauro M, "Perancangan Sistem Rekomendasi Film menggunakan metode Contentbased Filtering" (Paper, Jurusan Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara: Jakarta Barat).
   
7. Metrik Evaluasi. Tersedia: [Tautan](https://learn.microsoft.com/id-id/azure/ai-services/language-service/custom-text-classification/concepts/evaluation-metrics). Diakses pada: Mei 2025.
