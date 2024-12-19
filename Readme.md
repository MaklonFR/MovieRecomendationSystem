# Laporan Proyek Machine Learning - Maklon Jacob Frare

## Domain Proyek
Kemajuan teknologi, terutama dalam bidang perangkat bergerak dan internet, telah menciptakan peluang besar bagi pengembangan sistem rekomendasi. Pengguna kini dapat mengakses berbagai platform streaming seperti Netflix, Disney+, dan Amazon Prime yang menawarkan ribuan film. Namun hal ini juga menimbulkan tantangan baru untuk menemukan film yang sesuai dengan selera mereka di antara banyaknya pilihan yang tersedia `[5]`. Untuk mengatasi masalah ini, sistem rekomendasi film dikembangkan sebagai solusi efektif. istem rekomendasi berfungsi untuk memberikan rekomendasi yang dipersonalisasi berdasarkan interaksi pengguna sebelumnya dan preferensi yang diungkapkan, sehingga memudahkan mereka dalam menentukan pilihan `[6]`. Teknologi ini telah menjadi elemen penting dalam meningkatkan pengalaman pengguna di platform digital. 
Dari latar belakang itulah penulis mengambil topik ini sebagai studi kasus proyek machine learning yang penulis kerjakan dalam membangun sebuah model untuk proyek aplikasi yang sedang penulis kembangkan. Diharapkan model ini nantinya akan berguna untuk memberikan rekomendasi filim bagi pengguna sesuai dengan kebutuhannya.

## Business Understanding
Sistem rekomendasi film bertujuan untuk meningkatkan pengalaman pengguna dalam menemukan film yang sesuai dengan preferensi mereka. Dalam konteks bisnis, pemahaman yang mendalam tentang kebutuhan pengguna dan cara sistem dapat memenuhi kebutuhan tersebut sangat penting. Adanya peningkatan jumlah konten film yang tersedia di berbagai platform streaming, sering kali pengguna menghadapi kesulitan dalam memilih film yang ingin ditonton. Hal ini menciptakan kebutuhan untuk sistem rekomendasi yang efektif, yang dapat membantu pengguna menemukan film berdasarkan preferensi pribadi mereka.

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

## Data Understanding
Data understanding dalam proyek sistem rekomendasi film melibatkan pengumpulan, analisis, dan pemahaman tentang data yang akan digunakan untuk membangun model rekomendasi. Berikut adalah beberapa aspek penting dari data understanding dalam konteks ini.

### Informasi Dataset
Dataset yang digunakan yaitu The Movies Dataset. Informasi dari dataset film ini dapata dilihat pada tabel berikut:
| Jenis      | Keterangan     |
| -----------------------     | ------------------------------------------------------------------------- |
| Sumber                      | Dataset: [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  |
| Dataset Owner               | Rounak Banik		                        |
| Lisensi                     | CC0: Public Domain                        |
| Kategori                    | Movies & TV Shows                         |
| Usability                   | 8.24                                      |
| Jenis dan Ukuran Berkas     | ZIP Version 7 (943.76 MB)                 |
| Jumlah File Dataset         | 7 File (CSV)                              |

Dari informasi tabel diatas terlihat bahwa file-filenya berisi metadata untuk seluruh 45.000 film yang tercantum dalam Kumpulan Data Full MovieLens. Kumpulan data ini terdiri dari film-film yang dirilis pada atau sebelum Juli 2017. Poin data mencakup pemeran, kru, kata kunci plot, anggaran, pendapatan, poster, tanggal rilis, bahasa, perusahaan produksi, negara, jumlah suara TMDB, dan rata-rata suara.

Kumpulan data ini juga memiliki file yang berisi 26 juta rating dari 270.000 pengguna untuk seluruh 45.000 film. Ratingnya dalam skala 1-5 dan diperoleh dari situs resmi GroupLens. Berikut penjelasan dari penjelasan file-file dalam kumpulan data tersebut.

*movie_metadata.csv*: File Metadata Film utama. Berisi informasi tentang 45.000 film yang ditampilkan dalam kumpulan data Full MovieLens. Fitur-fiturnya meliputi poster, latar belakang, anggaran, pendapatan, tanggal rilis, bahasa, negara produksi, dan perusahaan.

*keywords.csv*: Berisi kata kunci plot film untuk film MovieLens kami. Tersedia dalam bentuk Objek JSON yang dirangkai.
  
*credits.csv*: Terdiri dari Informasi Pemeran dan Kru untuk semua film kami. Tersedia dalam bentuk Objek JSON yang dirangkai.
  
*links.csv*: File yang berisi ID TMDB dan IMDB dari semua film yang ditampilkan dalam kumpulan data Full MovieLens.
  
*links_small.csv*: Berisi ID TMDB dan IMDB dari sebagian kecil 9.000 film dari Kumpulan Data Lengkap.
  
*rating_small.csv*: Bagian dari 100.000 rating dari 700 pengguna pada 9.000 film.

### Membaca Dataset
Selanjutnya pada tahap ini dataset diatas akan disimpan pada variabel menggunakan fungsi pandas.read_csv. Hasilnya dapat ditampilkan pada gambar berikut:

![Rincian File Dataset](https://github.com/user-attachments/assets/0870ef20-4978-4d30-a5d6-847d2d6aa956)

Hasil dari gambar diatas merupakan jumlah data dalam file-file dataset film.

Pada proyek ini, hanya menggunakan 2 file csv yaitu `ratings_small.csv` (variabel `ratings`) dan `movies_metadata.csv` (variabel `movies`). 

### Exploratory Data Analysis - EDA
Analisis eksploratif data (EDA) adalah tahap penting dalam analisis data yang bertujuan untuk memahami dan mengeksplorasi karakteristik dataset sebelum melakukan analisis yang lebih mendalam. Dataset yang digunakan dalam proyek ini yaitu dataset fIlm yang dapat dijelaskan sebagai barikut: 

#### Univariate Analysis
Pada tahap ini, akan dialkukan analisis statistik yang melibatkan variabel movies dan ratings untuk melihat distribusi, pola, dan karakteristik data tersebut serta menggunakan grafik untuk menggambarkan distribusi genre dan rating film, serta hubungan antara fitur-fitur dalam dataset.

##### Deskripsi Variabel

a. ratings (ratings_small.csv)
Deskripsi variabel ratings dapat dilihat pada gambari berikut:

![dataset-ratings](https://github.com/user-attachments/assets/559f75c8-abca-4f8d-a0fa-b4d514bc605c)

Berdasarkan gambar diatas, variabel ratings terdiri dari 100004 baris dan 4 kolom yang dapat dijelaskan sebagai berikut:

| Variabel                    | Keterangan     |
| -----------------------     | ------------------------------------------------------------------------- |
| userId                      | ID unik untuk pengguna yang memberikan penilaian (rating). Ini digunakan untuk mengidentifikasi pengguna secara anonim.  |
| movieId|ID unik untuk film yang dinilai oleh pengguna untuk mendapatkan informasi lebih detail tentang film tersebut.                                  |
| rating | Nilai yang diberikan oleh pengguna untuk film tertentu dengan skala 1 hingga 5, di mana angka yang lebih tinggi menunjukkan penilaian yang lebih positif. |
| timestamp                   | Waktu ketika penilaian diberikan, direpresentasikan dalam format UNIX timestamp (jumlah detik sejak 1 Januari 1970).   |

b. movies (movies_metadata.csv)

Deskripsi variabel movies dapat dilihat pada gambari berikut:

![dataset-movies](https://github.com/user-attachments/assets/6bebf0ce-2e94-4b35-85e5-e5b858df8f20)

Berdasarkan gambar diatas, dataset movies terdiri dari 100004 baris dan 24 kolom yang dapat dijelaskan sebagai berikut:

| Variabel                    | Keterangan     |
| -----------------------     | ------------------------------------------------------------------------- |
|adult        | Mengindikasikan apakah film tersebut untuk orang dewasa (adult content). Nilainya biasanya True atau False.|
|belongs_to_collection|Informasi tentang koleksi atau seri film tertentu yang mencakup film ini (misalnya, film dalam seri Harry Potter). Biasanya berupa JSON atau string deskriptif.|
|Budget|Anggaran produksi film dalam satuan mata uang (biasanya USD). Nilainya berupa angka.|
|genres|Daftar genre film, seperti Action, Comedy, Drama. Biasanya berupa JSON atau daftar string.|
|homepage|URL dari situs web resmi film tersebut.|
|id|ID unik untuk film, biasanya merujuk pada database film tertentu seperti TMDb.|
|imdb_id|ID unik dari film di IMDb (misalnya, tt1234567).|
|original_language|Bahasa asli film tersebut dalam format kode bahasa ISO 639-1 (misalnya, en untuk bahasa Inggris).|
|original_title|Judul asli film dalam bahasa produksinya.|
|overview|Judul asli film dalam bahasa produksinya.|
|popularity|Skor popularitas film berdasarkan sistem tertentu, sering dihitung menggunakan algoritma dari platform film.|
|poster_path|Path atau tautan menuju gambar poster film. Biasanya berupa path yang dapat digabungkan dengan URL dasar untuk akses.|
|production_companies|Informasi tentang perusahaan produksi film tersebut. Biasanya berupa JSON dengan nama dan ID perusahaan.|
|production_countries|Negara tempat film tersebut diproduksi. Biasanya berupa JSON dengan nama negara dan kode negara.|
|release_date|Tanggal rilis film (format: YYYY-MM-DD).|
|revenue|Pendapatan kotor yang diperoleh film (biasanya dalam USD).|
|runtime|Durasi film dalam menit.|
|spoken_languages|Bahasa yang digunakan dalam dialog film. Biasanya berupa JSON dengan nama dan kode bahasa.|
|status|Status rilis film (misalnya, Released, In Production).|
|tagline|Slogan atau frasa singkat yang biasanya digunakan untuk promosi film.|
|title|Judul utama film yang digunakan untuk promosi.|
|video|Mengindikasikan apakah ada video terkait film. Nilainya biasanya berupa True atau False.|
|vote_average|Nilai rata-rata yang diberikan oleh pengguna (misalnya dari IMDb atau TMDb) berdasarkan skala tertentu (biasanya 1-10).|
|vote_count|Jumlah suara atau ulasan yang diberikan untuk film tersebut.|

##### Melihat Informasi Tipe Data
Infromasi dari tipe data variabel dapat dilihat pada gambar berikut:

 * variabel movies
   ![tipe_data_movies ](https://github.com/user-attachments/assets/5ade6703-b64a-4e3c-8aab-5e5f42965c9f)

 * variabel ratings
   ![tipe_data_rating](https://github.com/user-attachments/assets/1da41dcc-34ea-4543-87ca-d983fad448f1)

Dapat dilihat pada informasi dataset **movies** 20 variable dengan tipe data object dan 4 variabel bertipe float64. Sedangkan pada informasi dataset **ratings** terdapat 1 variabel dengan tipe data float64 dan 3 variable dengan tipe data int64.

##### Menghitung Total Dataset
Pada tahap ini, jumlah variabel dataset movies sebanyak 45466 dan memiliki 5 kolom sedangkan jumlah variabel dataset ratings sebanyak 100004 dan memiliki 4 kolom.

##### Menghitung Total Data Unik
Jumlah rincian data unik dapat dilihat pada gambar berikut:

![total_data_unik](https://github.com/user-attachments/assets/7434e8c2-574f-415c-9317-c00174cb9ade)

Dari hasil diatas terdapat 45436 film pada dataset movies, 9066 film pada dataset ratings dan 671 user pada dataset ratings

##### Melihat Data Deskriptif pada variabel dataset
Data deskriptif pada variabel movies dan ratings dapat dilihat pada gambar berikut:
* variabel movies
![deskriptif_movies](https://github.com/user-attachments/assets/b597510a-8dd8-4566-b68f-0408ca3ed349)

* variabel ratings
![deskriptif_ratings](https://github.com/user-attachments/assets/0f22fcb3-8c3f-4fa3-9d00-c9a72b8f2c2f)

Berdasarkan tampilan deskriptif variabel dataset movies dan ratings dapat dilihat tidak mencolok ada pesebaran nilai yang menimbulkan outlier

##### Distribusi Ratings
Langkah ini bertujuan untuk:
1. Mengidentifikasi nilai rating yang paling umum diberikan oleh pengguna.
2. Menilai apakah data rating cenderung condong ke satu nilai (misalnya, lebih banyak rating tinggi atau rendah).
3. Membantu memahami pola preferensi pengguna.

Tampilan distribusi rating dapat dilihat pada gambar berikut:

![distribusi-rating](https://github.com/user-attachments/assets/2d61d465-c467-4d81-8be9-f91d5297a698)

Berdasarkan diagram plot rating diatas, dapat dilihat bahwa nilai ratings paling umum diberikan pengguna adalah rating 4.0 dengan presentasi 28.7%, rating 3.0 dengan presentasi 20.1%, rating 5.0 dengan prestansi 15.1%. Sedangkan nilai rating yang lain berada dibawah pada presentasi 12.0%

##### Distribusi Gengres
Distribusi genre film adalah aspek penting dalam sistem rekomendasi, karena membantu memahami preferensi pengguna dan pola konsumsi film. Pada proyek ini menggunakan metode visualisasi Data dalam menampilkan grafik batang yang menggambarkan proporsi masing-masing genre secara visual, sehingga memudahkan pemahaman. Pada tahap ini akan dilakukan membersihkan, memproses, dan menormalkan data dalam kolom genres pada DataFrame df_movies Ada beberapa fungsi yang dipakai yakni:
* `fillna('[]')`, berfungsi untuk mengisi nilainull atau NaN dalam kolom genres dengan string kosong dalam format list (`[]`).
* `apply(literal_eval)`, fungsi literal_eval dari pustaka ast untuk mengubah string yang terlihat seperti Python literal menjadi tipe data list.
* `apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []`, fungsi lambda ini memproses setiap nilai dalam kolom genres: Jika nilai adalah sebuah daftar
* `(isinstance(x, list))`, maka ambil nilai dari kunci name untuk setiap elemen. Jika nilai bukan daftar, mengembalikan daftar kosong (`[]`).
    
Selanjutanya, ubah setiap elemen dalam daftar (genre) menjadi baris terpisah dengan fungsi `explode()`, kemudian menghitung jumlah kemunculan setiap genre dengan fungsi `value_counts()` dan terakhir membuat diagram batang untuk menampilkan distribusi genre dengan plot bar `plot(kind='bar')`. Langkah pertama, buat variabel dataframe baru untuk melakukan analisis visualisasi data. Kemudian konversi fitur(variabel) genres ke dalam bentuk list sehingga dapat dianalisi. Berikut adalah gambar distribrusi genres menggunakan grafik bar.

![genres_distribusi](https://github.com/user-attachments/assets/119090a5-1781-4afa-bb8f-041d7c4170b0)

Dari gambar grafik diatas, dapat dilihat bahwa genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar 20265 dan 13182. Sedangkan genre yang lain berada dibawah 10000. Terlihat juga ada 12 genre dengan jumlah 1.

##### Analisis Daftar film dengan skor tertinggi di seluruh rentang film
Untuk membuat daftar film dengan skor tertinggi menggunakan metode Weighted Score. Metode ini merupakan perhitungan skor berbobot untuk menggabungkan nilai-nilai yang berbeda berdasarkan pentingnya masing-masing komponen. Dalam konteks film, perlu menghitung skor berbobot berdasarkan informasi yang tersedia, seperti rata-rata penilaian (`vote_average`), jumlah suara (`vote_count`), dan jumlah suara rata-rata minimum yang diperlukan untuk dipertimbangkan dalam daftar.
Keterangan:
v = jumlah suara untuk film tertentu (vote_count)
m = jumlah suara minimum untuk masuk ke daftar (threshold)
R = rata-rata skor film tersebut (vote_average)
C = rata-rata skor semua film dalam dataset (rata-rata global)

Hasilnya dapat dilihat pada gambar berikut:

![analisis_top_rating_movies](https://github.com/user-attachments/assets/48127eca-ba1a-4fa8-b4b3-4c9645481459)

Gambar diatas menunjukan 5 film dengan skor tertinggi yang diberikan oleh pengguna. Dapat dilihat film dengan judul _Dilwale Dulhania Le Jayenge_	memiliki skor tertiggi yaitu 8.929668.

##### Analisis Rating Tertinggi
Selanjutnya gabungkan dataset df_movies dan ratings dengan fungsi pandas `pd.merge` dan mencari 10 film dengan rating tertinggi. Alisis rating tertinggi dapat dilihat pada gambar berikut:

![10-analisis rating](https://github.com/user-attachments/assets/257800b2-4b5e-4ecc-8598-3c679031faca)

Dapat dilihat pada gambar diatas dari 10 rating tertinggi film yang ada, film dengan judul _Terminator 3: Rise of the Machines_ memiliki rating teratas dengan *mean rating* 4.256 dan total rating sebanyak 324.

##### Membandingkan Peringkat rata-rata vs Jumlah total peringkat
Pada tahap ini akan dibandingkan rata-rata rangkin dan total rangking menggunakan `joinplot` untuk melihat pesebaran data yang dapat dilihat pada gambari dibawah ini:

![mean vs total rating](https://github.com/user-attachments/assets/336ee099-c9a0-4fbb-b4be-00c80b0048a9)

Berdasarkan grafik pesebaran data diatas total rating terting berada diatas 250 sebanyak 5 film, sadangkan rata-rata terbanyak pengguna memberi rating terhadap film berada diretang nilai 2 - 4.5 rating.

## Data Preparation
Data preparation adalah langkah penting dalam pengembangan sistem rekomendasi film yang efektif. Proses ini mencakup beberapa tahap, mulai dari pengumpulan data hingga pemrosesan akhir sebelum data digunakan dalam model machine learning. 

###  Data Clean
Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Setelah data terkumpul ada beberapa langkah yang perlu lakukan dalam tahap ini yaitu:

#### 1. Mengambil Fitur Sesuai Kebutuhan
Pada pronyek ini, dataset *movies_metadata (movies)*  hanya mengambil beberapa fitur atau kolom sesuai kebutuhan analsis pengolahan data yakni `['id', 'genres', 'title', 'vote_average', 'vote_count']`. Fitur-fitur tersebut dapat dilihat pada gambar di bawah ini yang menampilkan 5 data pada setiap fitur

![ambil_fitur_sesuai_kebutuhan](https://github.com/user-attachments/assets/e2f19108-10f3-4c33-828e-ea29e101fafb)

#### 2. Menyesuaikan Tipe Data Primary Key dan Foregein Key
Pada tahap ini, perlu menyesuaikan tipe data `primary key` dan `foregein key`. Jika dilihat pada informasi sebelumnya, dataset movies atribut id (`primary key`) dengan type data `object` berbeda pada dataset ratings atribut `movieId` dengan type data `int64`. Oleh karena itu, perlu menyamakan tipe data tersebut dengan cara menyamakan nama atribut movieId dan tipe data `int64`.

#### 3. Menangani Nilai Kosong (Missing Value)
Pada tahap ini, akan dilakukan pengecekan nilai kosong serta menanganinya pada variabel dataset movies dan ratings. Hasilnya dapat dilihat pada gambar berikut:

![nilai_null_movies](https://github.com/user-attachments/assets/fbf0fcb0-e41e-49a5-8cd7-bf742b4e6190)

Dari gambar diatas, nilai null terdapat pada variabel `title`, `vote_average` dan `vote_count` memiliki nilai null = 3.

![ratings_null](https://github.com/user-attachments/assets/4355e5ff-3740-492c-95a2-08c7db0490d1)

Dari gambar diatas terlihat variabel dataset ratings tidak memiliki nilai null.

Selanjutnya hapus jumlah data dengan nilai null, karena sangat sedikit dan tidak signifikan dibandingkan keseluruhan dataset.

#### 4. Menangani Duplikat Data (Duplicated Data)
Pada tap ini, akan dlakukan pengecekan data ganda serta menanganinya pada variabel dataset movies dan ratings. Setalah dilakukan pengecekan terdapat 28 data ganda pada variabel dataset movies dan tidak ada data ganda pada variabel dataset ratings. Terakhir lakukan penghapusan data ganda pada variabel dataset movies.

#### 5. Menangani Outliers
Pada tahap ini, akan diperlihatkan statistik deskriptif dari dataset variabel df_movies dan ratings mengunakan fungsi `describe()`.

* Variabel movies
   
![des_movies](https://github.com/user-attachments/assets/82b83bfb-2c0c-4c82-882b-bb0feda63a01)

* Variabel ratings
   
![des_rating](https://github.com/user-attachments/assets/3ff035f5-9f5e-4f90-a641-043753e98149)

Berdasarkan tampilan deskriptif dataset movies dan ratings dapat dilihat tidak mencolok ada pesebaran nilai yang menimbulkan `outlier`.

### Data Preprocesing
Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Langkah-langkah yang dilakukan dalam proyak ini yaitu pertama mengurutkan pengguna berdasarkan ID Pengguna, mengubah fitur genres movie ke bentuk list, melakukan penggabungan dataset ratings dan movies dan menghapus fitur yang tidak diperlukan. Hasilnya dapat dilihat pada gambar berikut:

![data_final](https://github.com/user-attachments/assets/6fb9e788-f5ce-43c8-b2ea-ff122c545db1)

Hasil gambar diatas menunjukan 10 data gabungan movies dan ratings, dimana fitur genre sudah menjadi bentuk list dan diurutkan berdasarkan iduser.

Selanjutnya diambil 20000 gabugan dataset movies dan ratings menggunakan fungsi `shuffle` dari library `sklearn.utils` untuk memperoleh data secara acak dengan tujuan mempermudah pengolahan dan mencegah crash. Hasilnya dapat dilihat pada gambar berikut:

![datset-20000](https://github.com/user-attachments/assets/8b8f03c8-e9c8-403b-a36d-658840e7c26c)

Pada gambar diatas terdapat 20000 baris dan 5 kolom yakni `userId`, `movieId`, `rating`, `genres` dan `title`.

### Content Based Filtering
Content-Based Filtering adalah metode dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item yang telah disukai atau dinilai oleh pengguna. Teknik yang digunakan yaitu teknik `TF-IDF` (Term Frequency-Inverse Document Frequency) untuk menentukan bobot fitur dan menghitung kesamaan antara item dalam hal ini adalah `genres`.

Berikutnya, lanjutkan ke tahap persiapan dengan membuat variabel preparation yang berisi dataframe df_sample_final kemudian mengurutkan berdasarkan movieId. Hasilnya dpat dilihat pada gambar berikut:



Selanjutnya, lakukan konversi data series menjadi list. Dalam hal ini, menggunakan fungsi `tolist()` dari library `numpy`. Setelah konversi dilakukan diperoleh variabel `movieId`, `movie_name`, `movie_genres` dan `title` dengan jumlah masing-masing sebanyak 2272.

Tahap terakhir, membuat dictionary untuk menentukan pasangan `key-value` pada data `movie_id`, `movie_name` dan `movie_genres` yang telah siapkan sebelumnya. Hasilnya dapat dilihat pada gambar berikut:

![movie_new](https://github.com/user-attachments/assets/fe874af2-1726-43f3-befe-9a5910665ff6)

Selanjutnya, gunakan fungsi `TfidfVectorizer` untuk mengkonversi `genres`. Namun sebelum itu genres perlu dikonversi dari list ke siting akar dapay diproses.

![TfidfVectorizer_Genre](https://github.com/user-attachments/assets/b2d91377-89ce-4d14-9320-1ca8bb81bd35)

Setelah mendapat index seluruh genre film, akan difit lalu ditransformasikan ke bentuk matriks sehingga diperoleh ukuran (2209, 22) serta mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense(). Hasilnya dapat dilihat pada gambar berikut:

![matrik-tendonse](https://github.com/user-attachments/assets/6054bfe3-9c22-4014-99e6-4ea804fa512c)

Setelah dibentuk matriks, dibuat tabel berisi judul film beserta genrenya berdasarkan TF-IDF yang telah diinisiasi. Hasilnya dapat dilihat pada gambar berikut:

![image](https://github.com/user-attachments/assets/d1fdfa27-ed24-4e89-a8f4-4567e10dcf65)

### Collaborative Filtering (CF)
Pada tahap ini data prerataion CF, Langkah pertama, cek dataset dengan fungsi `info()`, hasilnya ditampilkan pada gambar berikut:

![sampel_final](https://github.com/user-attachments/assets/f3083afc-7726-40d1-92c9-edd5e6ec6c66)

Dari hasil diatas, terdapat 20000 baris dan 7 kolom dan memiliki 3 tipe data `float64`, 2 tipe data `int64` dan 2 tipe data `object`. Langkah kedua Kedua, hapus kolom yang tidak dibutuhkan dalam pelatihan yaitu `genres` dan `title`. Langkah berikutnya, urutkan berdasarkan kolom `userId` untuk  masuk pada tahap encoding `userId` dan `movieId`.

#### Encoding userId dan movieId
Pada tahap ini, akan dilakukan encoding pada `userId` dan `movieId`. Hasilnya dapat ditampilkan pada gamabr dibwah ini:
* Encoding `userId`

![userId-encoding](https://github.com/user-attachments/assets/961d7b86-7b07-4780-b844-e531947aafb9)

* Encoding `movieId`

![movieId-encoding](https://github.com/user-attachments/assets/3a3ba6a0-b9d1-4278-8640-fe9cdd6567df)

Selanjutnya ambil total_user, total movie dan nilai rating minimum dan maksimum untuk proses pembagian dataset sebelum melakukan pelatihan model. Hasilnya diperoleh yaitu 669 pengguna, 2272 film serta nilai rating minimum sebesar 0.5 dan maksimum sebesar 5.0.

#### Membagi Data untuk Training dan Validasi
Pada tahap ini, data training dan data validasi dibagi untuk proses pelatihan model. Namun sebelum itu, perlu mengacak dataset sehingga menjadi data yang valid. Hasilnya seperti pada gambar berikut:

![data-acak-uji](https://github.com/user-attachments/assets/1edac292-34f4-4d77-a7f0-f98632a8e7af)

Selanjutnya, buat variabel x untuk mencocokkan data user dan Movie menjadi satu value, kemudian variabel y untuk membuat rating dari hasil. Terakhir,  bagi menjadi `80%` data train dan `20%`` data validasi.

## Modeling and Result
Pada tahap ini ada dua model yang dipakai untuk dilatih, di evaluasi dan memberikan rekomendasi kepada pengguna film. Kedua model tersebut dapat dijelaskan sebagai berikut:

### Modeling Content Based Filtering (CBF)
Pada proyek ini, metode yang digunakan adalah `Consine Similarity`,  yang berfungsi mengukur kesamaan antara dua dokumen atau vektor dalam ruang multidimensi. Metode ini digunakan untuk sistem rekomendasi berbasis `Content-Based Filtering` yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item genre film yang telah disukai atau dinilai oleh pengguna. Menurut Firmansyah(2018), `Cosine similarity` digunakan dalam ruang positif, dimana hasilnya dibatasi antara nilai `0` dan `1`. Kalau nilainya `0` maka dokumen tersebut dikatakan mirip jika hasilnya 1 maka nilai tersebut dikatakan tidak mirip Perhatikan bahwa batas ini berlaku untuk sejumlah dimensi.

Langkah pertama hitung `cosine similarity` pada matrix `tf-idf` yang dapat dilihat pada gambar berikut:

![consimilarity-1](https://github.com/user-attachments/assets/3369cf7b-12ff-44ed-9f1f-51dd549f2d41)

Langkah kedua, lihat hasil cosine similarity pada matrix tf-idf anta judul film yang mirip berdasarkan genre.

![consimilarity-2](https://github.com/user-attachments/assets/a8597a78-d66c-4afc-aefd-2b1ea59018c2)

Selanjutnya, buat fungsi rekomendasi film berdasarkan kemiripan genre dengan menerapkan fungsi Top-N rekokemendasi serta menguji dan mengevaluasi model yang dibuat.

#### Pengujian Sistem Rekomendasi
Pada proses pengujian akan diambil satu judul film untuk dilakukan pengujian seperti yan terlihat pada gambar berikut:

![cfb-1](https://github.com/user-attachments/assets/054a8b62-f530-48ef-bad9-5fb85fe3ae32)

10 hasil rekomendasi film dapat dilihat pada gambar berikut:

![top-10-cbf](https://github.com/user-attachments/assets/be229df6-c9c5-479c-94a6-ab14b9d0be02)

Dapat dilihat genre film uji yang dimasukan adalah `Crime`, `Drama`, `Romance`. Hasilnya genre ini tersebar di dalam 10 judul film yang memiliki kesaaman genre.

### Modeling Collaborative Filtering (CF)
Pada tahap ini menggunakan pendekatan Model-Based Deep Learning Collaborative Filtering. Metode `Deep Learning Neural Network (DNN)` yang merupakan subkategori dari machine learning yang menggunakan struktur ANN yang sangat dalam, dikenal sebagai deep neural networks. Deep learning melibatkan jaringan saraf dengan banyak lapisan tersembunyi, yang memungkinkan model untuk belajar dan mengenali pola yang sangat kompleks dan abstrak dari data `[2]`.

Pada tahap ini, model menghitung skor kecocokan antara user dan movie teknik embedding. Pertama, dilakukan proses embedding terhadap data user dan movie. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan movie. Selain itu, ditambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [`0,1`] dengan fungsi aktivasi sigmoid. Di sini, dibuatkan class `RecommenderNe`t dengan `keras Model class`. Kedua, lakukan proses compile terhadap model. Model ini menggunakan `Binary Crossentropy` untuk menghitung `loss function`, `Adam (Adaptive Moment Estimation)` sebagai `optimizer`, serta Mean Absolute Error(MAE) dan Root Mean Squared Error (RMSE) sebagai metrics evaluation.

Langkah berikutnya, mulailah proses training. Pada proses ini menggunakan fungsi `callbacks`, dimana jika kinerja model tidak mengalami keanaikan maka pelatiahan dihentikan. Pada proses training parameter yang digunakan yakni `batch_size=8`, `epoch = 50`, `shuffle = True` dan `verbose=1`

Proses latihan model dapat dilihat pada gambar berikut:

![pelatihan-cf-deeplearning](https://github.com/user-attachments/assets/cf428bd1-4f7c-42c7-a226-671a649e786b)

Dapat dilihat, hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1382 dan root_mean_squared_error: 0.1749

#### Penujian Sistem Rekomendasi
Proses pengujian sistem dilakukan berdasarkan Top-10 Rekomendasi film terbaik kepada pengguna yang memiliki kesamaan:

![image](https://github.com/user-attachments/assets/b480d86d-1d9a-44a7-ade7-6d41b12e43e7)

## Evaluation

### Evaluation Content Based Filtering (CBF)
Evaluasi CBF merupakan proses untuk menilai atau mengevaluasi kinerja sistem rekomendasi berbasis konten dalam memberikan rekomendasi yang relevan kepada pengguna. Pada proyek ini sistem rekomendasi berbasis konten (CBF) bekerja dengan cara menganalisis atribut rating dari judul film yang disukai oleh pengguna dan kemudian merekomendasikan judul film lain dengan atribut rating yang serupa. Tujuan evaluasi ini adalah untuk mengukur seberapa baik sistem dapat memberikan rekomendasi yang tepat dan memuaskan berdasarkan preferensi atau interaksi pengguna sebelumnya. Fungsi yang digunakan untuk mengukur CBF yakni metriks presisi. Metrik presisi mengungkapkan berapa banyak kelas yang diprediksi diberi label dengan benar (posistif) [7].

Berikut adalah rumus dari metriks presisi[7]:

`Precision = True_Positive / (True_Positive + False_Positive)`

Keterangan:
* Precision = Hasil Presisi
* True_Positive = Prediksi benar
* False_Positive = Prediksi salah

Hasil pengujian menggunakan atribut genre {"Crime", "Drama", dan "Romance"} dengan 10 rekomendasi film yakni:

| | title                   |	genres      | hasil presisis  |
|-| ----------              | ----------  | -----------  |
|0|	Schizo	                | Crime, Drama, Romance           | TRUE   |
|1|	The Little Thief	      | Romance, Drama, Crime          | TRUE   |
|2|	Made in Hong Kong       |	Crime, Drama, Romance | TRUE   |
|3|	The Thomas Crown Affair |	Drama, Crime, Romance          | TRUE   |
|4|	The Thomas Crown Affair |	Romance, Crime, Thriller, Drama          | TRUE   |
|5|	3-Iron                  |	Drama, Romance, Crime  | TRUE   |
|6|	Angel Face              |	Crime, Drama, Romance  | TRUE   |
|7|	Prizzi's Honor          |	Romance, Comedy, Crime, Drama | TRUE   |
|8|	Tie Me Up! Tie Me Down! |	Comedy, Crime, Drama, Romance | TRUE   |
|9|	Young Adam              |	Drama, Thriller, Crime, Romance| TRUE   |

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
    
![mae](https://github.com/user-attachments/assets/0a4b5a01-0914-48aa-874a-7e2e055b6669)

Berdasarkan hasil `fitting` nilai konvergen metrik MAE berada sedikit dibawah 0.1373 untuk training dan sedikit diatas 0.1500 untuk validasi.

* Gambar Visualisasi Metriks RMSE

![rmse](https://github.com/user-attachments/assets/cc352b2f-bade-4efa-ace5-2ebc42abce6b)

Berdasarkan hasil fitting nilai konvergen metrik RMSE berada sedikit diatas 0.1762 untuk training dan sedikit dibawah 0.180 untuk validasi.

## Kesimpulan
Berdasarkan hasil yang diperoleh setelah melakukan proses pengolahan data sampai proses evaaluasi dapat dismpulkan bahwah:
1. Pengunaan Teknik EDA dapat melihat distribusi data pada data rating dan data genre film dengan jelas. Nilai ratings paling umum diberikan pengguna adalah rating `4.0` dengan presentasi `28.7%`, rating `3.0` dengan presentasi `20.1%`, rating `5.0` dengan prestansi `15.1%`. Sedangkan nilai rating yang lain berada dibawah pada presentasi `12.0%`. Sedangkan genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar `20265` dan `13182`, sementara genre yang lain berada dibawah `10000`. Film dengan judul Terminator `3: Rise of the Machines` memiliki rating teratas dengan mean rating `4.256` dan total rating sebanyak `324`. Total rating terting berada diatas `250` sebanyak `5 film`, sedangkan rata-rata terbanyak pengguna memberi rating terhadap film berada diretang nilai `2 - 4.5` rating.
2. Dengan preparation data yang sistematis, seperti menangani nilai hilang (missing values), menghapus atau menangani outlier, dan melakukan encoding pada data kategorikal, proses analisis data menjadi lebih efisien dan akurat. Data yang bersih dan siap digunakan akan mengurangi risiko kesalahan dalam model analitik.
3. Dengan mengunakan metode Content-Based Filtering dapat memberikan 10 rekomendaasi film kepada sesama pengguna berdasarkan kesaaman perilaku pengguna dengan nilai presesion matriks sebesar 100.00%.
4. Penggunaan Model-Based Deep Learning Collaborative Filtering memberikan hasil rekomendasi yang lebih akurat dan relevan bagi pengguna. Hal ini di buktikan dngan hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1373 dan root_mean_squared_error: 0.1762 dan juga tampilan matriks visualisasi yang menunjukan nilai MAE dan RMSE berada dibawah 0.180 pada epoh ke-17.

## Daftar Pustaka
1. D. A. R. Ariantini, A. S. M. Lumenta and A.Jacobus, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016.
2. Neural Network: Cikal Bakal Revolusi Deep Learning. Tersedia: [Tautan](https://www.dicoding.com/blog/neural-network-cikal-bakal-revolusi-deep-learning/). Diakses pada: Desember 2024.
3. Perbedaan MAE, MSE, RMSE, dan MAPE pada Data Science. Tersedia: [Tautan]([https://pages.github.com/](https://www.trivusi.web.id/2023/03/perbedaan-mae-mse-rmse-dan-mape.html)). Diakses pada: Desember 2024.
4. Firmansyah Fataruba, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016. "PENERAPAN METODE COSINE SIMILARITY UNTUK PENGECEKAN KEMIRIPAN JAWABAN UJIAN SISWA", JATI (Jurnal Mahasiswa Teknik Informatika) Vol. 2  No. 2, September 2018.
5. Nathania, R.A. 2024. Sistem Rekomendasi Film Dengan Collaborative Deep Learning. (Skripsi, Fakultas Teknologi Informasi dan Sains, Universitas Katolik Parahyangan: Bandung).
6. Salim .E, Paragantha. J, Lauro M, "Perancangan Sistem Rekomendasi Film menggunakan metode Contentbased Filtering" (Paper, Jurusan Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara: Jakarta Barat).
7. Metrik evaluasi. Tersedia: [Tautan](https://learn.microsoft.com/id-id/azure/ai-services/language-service/custom-text-classification/concepts/evaluation-metrics). Diakses pada: Desember 2024.
