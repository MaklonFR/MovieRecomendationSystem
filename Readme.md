# Laporan Proyek Machine Learning - Maklon Jacob Frare

## Domain Proyek
Kemajuan teknologi, terutama dalam bidang perangkat bergerak dan internet, telah menciptakan peluang besar bagi pengembangan sistem rekomendasi. Pengguna kini dapat mengakses berbagai platform streaming seperti Netflix, Disney+, dan Amazon Prime yang menawarkan ribuan film. Namun hal ini juga menimbulkan tantangan baru untuk menemukan film yang sesuai dengan selera mereka di antara banyaknya pilihan yang tersedia [1][2]. Untuk mengatasi masalah ini, sistem rekomendasi film dikembangkan sebagai solusi efektif. istem rekomendasi berfungsi untuk memberikan rekomendasi yang dipersonalisasi berdasarkan interaksi pengguna sebelumnya dan preferensi yang diungkapkan, sehingga memudahkan mereka dalam menentukan pilihan [3][5]. Teknologi ini telah menjadi elemen penting dalam meningkatkan pengalaman pengguna di platform digital. 
Dari latar belakang itulah penulis mengambil topik ini sebagai studi kasus proyek machine learning yang penulis kerjakan dalam membangun sebuah model untuk proyek aplikasi yang sedang penulis kembangkan. Diharapkan model ini nantinya akan berguna untuk memberikan rekomendasi filim bagi pengguna sesuai dengan kebutuhannya.

## Business Understanding
Sistem rekomendasi film bertujuan untuk meningkatkan pengalaman pengguna dalam menemukan film yang sesuai dengan preferensi mereka. Dalam konteks bisnis, pemahaman yang mendalam tentang kebutuhan pengguna dan cara sistem dapat memenuhi kebutuhan tersebut sangat penting. Adanya peningkatan jumlah konten film yang tersedia di berbagai platform streaming, sering kali pengguna menghadapi kesulitan dalam memilih film yang ingin ditonton. Hal ini menciptakan kebutuhan untuk sistem rekomendasi yang efektif, yang dapat membantu pengguna menemukan film berdasarkan preferensi pribadi mereka.

### Problem Statement
Berdasarkan latar belakang diatas, proyek ini berfokus pada beberapa masalah utama yang perlu dipecahkan:
* Bagaimana cara melakukan pengolahan data yang baik sehingga dapat digunakan untuk membangun model sistem rekomendasi yang efektif?
* Bagaimana cara membangun model machine learning yang mampu merekomendasikan film berdasarkan preferensi pengguna?
* Bagaimana memastikan bahwa rekomendasi film yang diberikan benar-benar sesuai dengan minat pengguna?

### Goal
Tujuan dibuat proyek ini adalah sebagai barikut:
* Melakukan pengolahan data secara efisien agar dapat digunakan dalam pembangunan model sistem rekomendasi.
* Membangun model machine learning yang dapat memberikan rekomendasi film dengan tingkat akurasi tinggi.
* Meningkatkan user experience dengan memberikan rekomendasi film yang relevan dan menarik bagi pengguna[1][4].

### Solution Statement
Dalam proyek ini, untuk mengatasi asalah diatas, digunakan teknik analisis data dan metode machine learning yaitu:
* Teknik Univariate Exploratory Data Analysis (EDA) dan Preparation Data untuk proses pengolahan data yang efektif dan efisien. 
* Content-Based Filtering untuk merekomendasikan film berdasarkan kemiripan atribut seperti genre, aktor, dan sinopsis. Ini memastikan bahwa pengguna mendapatkan rekomendasi yang relevan berdasarkan film yang telah mereka tonton sebelumnya[1][2].
* Collaborative Filtering untuk memanfaatkan data dari interaksi pengguna lain untuk memberikan rekomendasi. Metode ini bergantung pada kesamaan preferensi antara pengguna untuk menemukan film baru yang mungkin disukai[2][3].

## Data Understanding
Data understanding dalam proyek sistem rekomendasi film melibatkan pengumpulan, analisis, dan pemahaman tentang data yang akan digunakan untuk membangun model rekomendasi. Data merupakan komponen kunci dalam sistem ini, karena kualitas dan relevansi data akan mempengaruhi akurasi rekomendasi yang dihasilkan. Berikut adalah beberapa aspek penting dari data understanding dalam konteks ini.

### Informasi Dataset

  | Jenis      | Keterangan     |
  | -----------------------     | ------------------------------------------------------------------------- |
  | Sumber                      | Dataset: [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  |
  | Dataset Owner               | Rounak Banik		                        |
  | Lisensi                     | CC0: Public Domain                        |
  | Kategori                    | Movies & TV Shows                         |
  | Usability                   | 8.24                                      |
  | Jenis dan Ukuran Berkas     | ZIP Version 7 (943.76 MB)                 |
  | Jumlah File Dataset         | 7 File (CSV)                              |

File-file CSV yakni:
* credits.csv
* keywords.csv
* links.csv
* links_small.csv
* movies_metadata.csv
* ratings.csv
* ratings_small.csv

Pada proyek ini hanya menggunakan 2 file csv yaitu ratings.csv (ratings) dan movies_metadata.csv (movies) yang dapat dijelaskan sebagai berikut:
**1. ratings_small.csv**
Terdiri dari dari 100.000 rating dari 700 pengguna pada 9.000 film. Total baris sebanyak 100004 dan kolom sebayak 4 dengan kolom sebagai berikut:
* userId
ID unik untuk pengguna yang memberikan penilaian (rating). Ini digunakan untuk mengidentifikasi pengguna secara anonim.
* movieId
ID unik untuk film yang dinilai oleh pengguna. ID ini biasanya terhubung dengan dataset film utama untuk mendapatkan informasi lebih detail tentang film tersebut.
* rating
Nilai yang diberikan oleh pengguna untuk film tertentu. Biasanya berupa skala, seperti dari 1 hingga 5 atau 1 hingga 10, di mana angka yang lebih tinggi menunjukkan penilaian yang lebih positif.
* timestamp
Waktu ketika penilaian diberikan, direpresentasikan dalam format UNIX timestamp (jumlah detik sejak 1 Januari 1970).

**2. movies_metadata.csv**
Memiliki sebanyak 45466 baris dan 24 Kolom dengan rincian kolom sebagai berikut
* adult
Mengindikasikan apakah film tersebut untuk orang dewasa (adult content). Nilainya biasanya True atau False.
belongs_to_collection
* Informasi tentang koleksi atau seri film tertentu yang mencakup film ini (misalnya, film dalam seri Harry Potter). Biasanya berupa JSON atau string deskriptif.
* Budget
Anggaran produksi film dalam satuan mata uang (biasanya USD). Nilainya berupa angka.
* genres
Daftar genre film, seperti Action, Comedy, Drama. Biasanya berupa JSON atau daftar string.
* homepage
URL dari situs web resmi film tersebut.
* id
ID unik untuk film, biasanya merujuk pada database film tertentu seperti TMDb.
* imdb_id
ID unik dari film di IMDb (misalnya, tt1234567).
* original_language
Bahasa asli film tersebut dalam format kode bahasa ISO 639-1 (misalnya, en untuk bahasa Inggris).
* original_title
Judul asli film dalam bahasa produksinya.
* overview
Ringkasan singkat tentang alur atau cerita dari film.
* popularity
Skor popularitas film berdasarkan sistem tertentu, sering dihitung menggunakan algoritma dari platform film.
* poster_path
Path atau tautan menuju gambar poster film. Biasanya berupa path yang dapat digabungkan dengan URL dasar untuk akses.
* production_companies
Informasi tentang perusahaan produksi film tersebut. Biasanya berupa JSON dengan nama dan ID perusahaan.
* production_countries
Negara tempat film tersebut diproduksi. Biasanya berupa JSON dengan nama negara dan kode negara.
* release_date
Tanggal rilis film (format: YYYY-MM-DD).
* revenue
Pendapatan kotor yang diperoleh film (biasanya dalam USD).
* runtime
Durasi film dalam menit.
* spoken_languages
Bahasa yang digunakan dalam dialog film. Biasanya berupa JSON dengan nama dan kode bahasa.
* status
Status rilis film (misalnya, Released, In Production).
* tagline
Slogan atau frasa singkat yang biasanya digunakan untuk promosi film.
* title
Judul utama film yang digunakan untuk promosi.
* video
Mengindikasikan apakah ada video terkait film. Nilainya biasanya berupa True atau False.
* vote_average
Nilai rata-rata yang diberikan oleh pengguna (misalnya dari IMDb atau TMDb) berdasarkan skala tertentu (biasanya 1-10).
* vote_count
Jumlah suara atau ulasan yang diberikan untuk film tersebut.

### Mengambil Fitur Sesuai Kebutuhan
Pada pronyek ini, dataset *movies_metadata (movies)* kita hanya mengambil beberapa fitur atau kolom sesuai kebutuhan analsis pengolahan data yakni `['id', 'genres', 'title', 'vote_average', 'vote_count']`. Fitur-fitur tersebut dapat dilihat pada gambar di bawah ini yang menampilkan 5 data pada setiap fitur

Gambar_1.

### Melihat Tipe Data Pada Dataset
Tahap ini kita akan melihat Informsi tipe data pada dataset ratings dan movies yang akan kita gunakan. Selengkapnya dapat dilihat pada gambar berikut.

Gambar_2.

Dapat dilihat pada informasi dataset **movies**, 3 variable dengan tipe data object dan 2 variabel bertipe float64. Selain itu ada perbedaan pada variabel `title`, `vote_average` dan `vote_count` yang memiliki data 45460 dengan variabel id dan genres dengan data 45466. Sedangkan pada informasi dataset **ratings** terdapat 1 variabel dengan tipe data float64 dan 3 variable dengan tipe data int64.

### Menangani Missing Value
Mengatasi missing values (nilai yang hilang) merupakan langkah penting dalam pengolahan data untuk sistem rekomendasi film. Keberadaan missing values dapat mempengaruhi kualitas model dan akurasi rekomendasi yang dihasilkan. Berikut langkah-langkah yang perlu diambil dalam menganangi missing value pada dataset movies dan ratings.
	Pertama kita lakukan untuk dataset movies
* Menggunakan fungsi `isnull().sum()` untuk menampilkan jumlah nilai yg hilang pada setiap fitur(variabel)

Gambar-3

Dari hasil diatas, nilai null terdapat pada variabel title, vote_average dan vote_count memiliki nilai null = 6 pada.
* Menampilkan isi dataset yang memuat nilai yang hilang dengan fungsi `isnull().any(axis = 1)`

Gambar-4

* Menghapus missing value pada dataset menggunakan fungsi `dropna(subset=[‘nama_fitur’]`

Gambar-5

Kedua, kita lakukan pada dataset ratings, setelah dicek tidak memiliki missing value seperti yang terlihat pada gambar dibah ini:

Gambar-6

### Manangani Duplikat Data
Menangani duplikat data adalah langkah penting dalam proses pembersihan data untuk memastikan integritas dan kualitas dataset yang digunakan dalam sistem rekomendasi film. Duplikat data dapat menyebabkan bias dalam analisis dan menghasilkan rekomendasi yang tidak akurat. Berikut adalah langkah-langkah untuk menangani duplikat data.
Pertama, kita lakukan untuk dataset **movies**,
* Indentfikasi dataset dengan menggunakan fungsi ` duplicated().sum()`. Setelah dilakukan pengecekan ternyata memiliki 28 data ganda. 
* Menampilkan isi dataset yang memiliki data ganda pada setiap fitur (kolom), seperti yang terlihat pada gambar berikut:

Gambar-7

* Menghapus duplikat data pada dataset dengan menggunakan fungsi `drop_duplicates(inplace = True)`

Kedua, lakukan pada dataset **ratings**, setelah dicek tidak memiliki data duplikat.

### Distribusi Ratings
Distribusi ratings merupakan aspek penting dalam sistem rekomendasi film, karena dapat mempengaruhi cara model merekomendasikan film kepada pengguna. Pada langkah ini kita akan mendistribusi ratings dengan tujuan:
* Mengidentifikasi nilai rating yang paling umum diberikan oleh pengguna.
* Menilai apakah data rating cenderung condong ke satu nilai.
* Membantu memahami pola preferensi pengguna.
Proses distribusi rating mengunakan library `matplotlib` seperting yang ditampilkan pada gambar berikut:

Gambar-8

Berdasarkan diagram plot rating diatas, dapat dilihat bahwa nilai ratings paling umum diberikan pengguna adalah rating 4.0 dengan presentasi 28.7%, rating 3.0 dengan presentasi 20.1%, rating 5.0 dengan prestansi 15.1%. Sedangkan nilai rating yang lain berada dibawah pada presentasi 12.0%.

### Distribusi Gengres
Distribusi genre film adalah aspek penting dalam sistem rekomendasi, karena membantu memahami preferensi pengguna dan pola konsumsi film. Pada proyek ini menggunakan metode Visualisasi Data dalam menampilkan grafik batang yang menggambarkan proporsi masing-masing genre secara visual, sehingga memudahkan pemahaman. 
Pada tahap ini kita akan membersihkan, memproses, dan menormalkan data dalam kolom genres pada DataFrame df_movies. Ada beberapa fungsi yang kita pakai yakni:
* `fillna('[]')`, berfungsi untuk mengisi nilainull atau NaN dalam kolom genres dengan string kosong dalam format list (`[]`).
•	`apply(literal_eval)`, fungsi literal_eval dari pustaka ast untuk mengubah string yang terlihat seperti Python literal menjadi tipe data list.
•	`apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []`, fungsi lambda ini memproses setiap nilai dalam kolom genres: Jika nilai adalah sebuah daftar `(isinstance(x, list))`, maka ambil nilai dari kunci name untuk setiap elemen. Jika nilai bukan daftar, mengembalikan daftar kosong (`[]`).

Selanjutanya kita ubah setiap elemen dalam daftar (genre) menjadi baris terpisah dengan fungsi `explode()`, kemudian menghitung jumlah kemunculan setiap genre dengan fungsi `value_counts()` dan terakhir kita membuat diagram batang untuk menampilkan distribusi genre dengan plot bar `plot(kind='bar')`. Berikut adalah gambar distribrusi genres menggunakan grafik bar.

Gambar-9

Dari grafik diatas, dapat dilihat bahwa genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar 20243 dan 13137. Sedangkan genre yang lain berada dibawah 10000.

### Daftar film dengan skor tertinggi di seluruh rentang film
Untuk membuat daftar film dengan skor tertinggi menggunakan metode Weighted Score. Metode ini merupakan perhitungan skor berbobot untuk menggabungkan nilai-nilai yang berbeda berdasarkan pentingnya masing-masing komponen. Dalam konteks film, kita perlu menghitung skor berbobot berdasarkan informasi yang tersedia, seperti rata-rata penilaian (vote_average), jumlah suara (vote_count), dan jumlah suara rata-rata minimum yang diperlukan untuk dipertimbangkan dalam daftar. 
Keterangan:
v = jumlah suara untuk film tertentu (vote_count)
m = jumlah suara minimum untuk masuk ke daftar (threshold)
R = rata-rata skor film tersebut (vote_average)
C = rata-rata skor semua film dalam dataset (rata-rata global)
Hasilnya 


















