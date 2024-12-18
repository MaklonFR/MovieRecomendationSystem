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
Selanjutnya pada tahap ini, kita akan baca data-data diatas menggunakan fungsi pandas.read_csv. Hasilnya dapat ditampilkan pada gambar berikut:

![Rincian File Dataset](https://github.com/user-attachments/assets/0870ef20-4978-4d30-a5d6-847d2d6aa956)

Hasil dari gambar diatas merupakan jumlah data dalam file-file dataset film.

Pada proyek ini kita hanya menggunakan 2 file csv yaitu `ratings_small.csv` (variabel `ratings`) dan `movies_metadata.csv` (variabel `movies`). Dari kedua file ini kita akan melihat informasi apa saja yang ada di dalammya.

### Exploratory Data Analysis - EDA
Analisis eksploratif data (EDA) adalah tahap penting dalam analisis data yang bertujuan untuk memahami dan mengeksplorasi karakteristik dataset sebelum melakukan analisis yang lebih mendalam. Dataset yang digunakan dalam proyek ini yaitu dataset fIlm yang dapat dijelaskan sebagai barikut: 

#### Univariate Analysis
Berdasarkan varibel-variabel dataset di ataas, kita cukup mengambil variabel sesuai kebutuhan analisis dan pelatihan model pada proyek ini yakni movies dan rating.

##### Deskripsi Variabel

a. ratings (ratings_small.csv)
Pada langkah ini kita akan menampilkan informasi variabel ratings dan movies dengan fungsi `info`. Pertama kita mulai cek variabel ratings yang hasilnya seperti gambari berikut:

![dataset-ratings](https://github.com/user-attachments/assets/559f75c8-abca-4f8d-a0fa-b4d514bc605c)

Berdasarkan gambar diatas, variabel ratings terdiri dari 100004 baris dan 4 kolom yang dapat dijelaskan sebagai berikut:

| Variabel                    | Keterangan     |
| -----------------------     | ------------------------------------------------------------------------- |
| userId                      | ID unik untuk pengguna yang memberikan penilaian (rating). Ini digunakan untuk mengidentifikasi pengguna secara anonim.  |
| movieId|ID unik untuk film yang dinilai oleh pengguna untuk mendapatkan informasi lebih detail tentang film tersebut.                                  |
| rating | Nilai yang diberikan oleh pengguna untuk film tertentu dengan skala 1 hingga 5, di mana angka yang lebih tinggi menunjukkan penilaian yang lebih positif. |
| timestamp                   | Waktu ketika penilaian diberikan, direpresentasikan dalam format UNIX timestamp (jumlah detik sejak 1 Januari 1970).   |

b. movies (movies_metadata.csv)

Kedua kita cek variabel movies. Hasilnya sperti gambar berikut:

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
Tahap ini kita akan melihat Informsi tipe data pada dataset ratings dan movies yang akan kita gunakan. Selengkapnya dapat dilihat pada gambar berikut.

 * variabel movies
   ![tipe_data_movies ](https://github.com/user-attachments/assets/5ade6703-b64a-4e3c-8aab-5e5f42965c9f)

 * variabel ratings
   ![tipe_data_rating](https://github.com/user-attachments/assets/1da41dcc-34ea-4543-87ca-d983fad448f1)

Dapat dilihat pada informasi dataset **movies** 20 variable dengan tipe data object dan 4 variabel bertipe float64. Sedangkan pada informasi dataset **ratings** terdapat 1 variabel dengan tipe data float64 dan 3 variable dengan tipe data int64.

##### Menghitung Total Dataset
Pada tahap ini, jumlah variabel dataset movies sebanyak 45466 dan memiliki 5 kolom sedangkan jumlah variabel dataset ratings sebanyak 100004 dan memiliki 4 kolom.

##### Menghitung Total Data Unik
Jumlah rincian data unik kita dapat dilihat pada gambar berikut:

![data_unik](https://github.com/user-attachments/assets/87801017-e92f-42eb-9c1b-2a504604aa3b)

Dapat dilihat bahwa jumlah id dalam film sebanyak 45436 pada movies dan jumlah movieId pada ratings sebanyak 9066.  Sedangkan jumlah pengguna unik pada ratings sebanyak 671. 

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
Distribusi genre film adalah aspek penting dalam sistem rekomendasi, karena membantu memahami preferensi pengguna dan pola konsumsi film. Pada proyek ini menggunakan metode visualisasi Data dalam menampilkan grafik batang yang menggambarkan proporsi masing-masing genre secara visual, sehingga memudahkan pemahaman. Pada tahap ini kita akan membersihkan, memproses, dan menormalkan data dalam kolom genres pada DataFrame df_movies Ada beberapa fungsi yang kita pakai yakni:
* `fillna('[]')`, berfungsi untuk mengisi nilainull atau NaN dalam kolom genres dengan string kosong dalam format list (`[]`).
* `apply(literal_eval)`, fungsi literal_eval dari pustaka ast untuk mengubah string yang terlihat seperti Python literal menjadi tipe data list.
* `apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []`, fungsi lambda ini memproses setiap nilai dalam kolom genres: Jika nilai adalah sebuah daftar
* `(isinstance(x, list))`, maka ambil nilai dari kunci name untuk setiap elemen. Jika nilai bukan daftar, mengembalikan daftar kosong (`[]`).
    
Selanjutanya kita ubah setiap elemen dalam daftar (genre) menjadi baris terpisah dengan fungsi `explode()`, kemudian menghitung jumlah kemunculan setiap genre dengan fungsi `value_counts()` dan terakhir kita membuat diagram batang untuk menampilkan distribusi genre dengan plot bar `plot(kind='bar')`. Langkah pertama kita buat variabel dataframe baru untuk melakukan analisis visualisasi data. Kemudian kita konversi fitur(variabel) genres ke dalam bentuk list sehingga dapat dianalisi. Berikut adalah gambar distribrusi genres menggunakan grafik bar.

![genres_distribusi](https://github.com/user-attachments/assets/119090a5-1781-4afa-bb8f-041d7c4170b0)

Dari gambar grafik diatas, dapat dilihat bahwa genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar 20265 dan 13182. Sedangkan genre yang lain berada dibawah 10000. Terlihat juga ada 12 genre dengan jumlah 1.

##### Analisis Daftar film dengan skor tertinggi di seluruh rentang film
Untuk membuat daftar film dengan skor tertinggi menggunakan metode Weighted Score. Metode ini merupakan perhitungan skor berbobot untuk menggabungkan nilai-nilai yang berbeda berdasarkan pentingnya masing-masing komponen. Dalam konteks film, kita perlu menghitung skor berbobot berdasarkan informasi yang tersedia, seperti rata-rata penilaian (`vote_average`), jumlah suara (`vote_count`), dan jumlah suara rata-rata minimum yang diperlukan untuk dipertimbangkan dalam daftar.
Keterangan:
v = jumlah suara untuk film tertentu (vote_count)
m = jumlah suara minimum untuk masuk ke daftar (threshold)
R = rata-rata skor film tersebut (vote_average)
C = rata-rata skor semua film dalam dataset (rata-rata global)

Hasilnya dapat dilihat pada gambar berikut:

![analisis_top_rating_movies](https://github.com/user-attachments/assets/48127eca-ba1a-4fa8-b4b3-4c9645481459)

Gambar diatas menunjukan 5 film dengan skor tertinggi yang diberikan oleh pengguna. Dapat dilihat film dengan judul _Dilwale Dulhania Le Jayenge_	memiliki skor tertiggi yaitu 8.929668.

##### Analisis Rating Tertinggi
Selanjutnya kita gabungkan dataset df_movies dan ratings dengan fungsi pandas pd.merge dan mencari 10 film dengan rating tertinggi. Alisis rating tertinggi dapat dilihat pada gambar berikut:

![10-analisis rating](https://github.com/user-attachments/assets/257800b2-4b5e-4ecc-8598-3c679031faca)

Dapat dilihat pada gambar diatas dari 10 rating tertinggi film yang ada, film dengan judul _Terminator 3: Rise of the Machines_ memiliki rating teratas dengan *mean rating* 4.256 dan total rating sebanyak 324.

##### Membandingkan Peringkat rata-rata vs Jumlah total peringkat
Pada tahap ini kita akan membandingkan rata-rata rangkin dan total rangking menggunakan joinplot untuk melihat pesebaran data yang dapat dilihat pada gambari dibawah ini:

![mean vs total rating](https://github.com/user-attachments/assets/336ee099-c9a0-4fbb-b4be-00c80b0048a9)

Berdasarkan grafik pesebaran data diatas total rating terting berada diatas 250 sebanyak 5 film, sadangkan rata-rata terbanyak pengguna memberi rating terhadap film berada diretang nilai 2 - 4.5 rating.

## Data Preparation
Data preparation adalah langkah penting dalam pengembangan sistem rekomendasi film yang efektif. Proses ini mencakup beberapa tahap, mulai dari pengumpulan data hingga pemrosesan akhir sebelum data digunakan dalam model machine learning. 

###  Data Clean
Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Setelah data terkumpul ada beberapa langkah yang perlu kita lakukan dalam tahap ini yaitu:

#### 1. Mengambil Fitur Sesuai Kebutuhan
Pada pronyek ini, dataset *movies_metadata (movies)* kita hanya mengambil beberapa fitur atau kolom sesuai kebutuhan analsis pengolahan data yakni `['id', 'genres', 'title', 'vote_average', 'vote_count']`. Fitur-fitur tersebut dapat dilihat pada gambar di bawah ini yang menampilkan 5 data pada setiap fitur

![gambar-1](https://github.com/user-attachments/assets/87240afd-52bc-4419-a8e3-c414848c605a) 

#### 2. Menyesuaikan Tipe Data Primary Key dan Foregein Key
Pada tahap ini, kita perlu menyesuaikan tipe data `primary key` dan `foregein key`. Jika dilihat pada informasi sebelumnya, dataset movies atribut id (`primary key`) dengan type data `object` berbeda pada dataset ratings atribut `movieId` dengan type data `int64`. Oleh karena itu, kita perlu menyamakan tipe data tersebut dengan cara menyamakan nama atribut movieId dan tipe data `int64`.

#### 3. Menangani Nilai Kosong (Missing Value)
Pada tap ini kita akan lakukan pengecekan nilai kosong serta menanganinya pada variabel dataset movies dan ratings. Hasilnya dapat dilihat pada gambar berikut:

![nilai_null_movies](https://github.com/user-attachments/assets/fbf0fcb0-e41e-49a5-8cd7-bf742b4e6190)

Dari gambar diatas, nilai null terdapat pada variabel `title`, `vote_average` dan `vote_count` memiliki nilai null = 6.

![nilai_null_ratings](https://github.com/user-attachments/assets/5391c467-523e-433b-9fe6-97935d2ac2d5)

Dari gambar diatas terlihat variabel dataset ratings tidak memiliki nilai null.

Selanjutnya kita hapus jumlah data dengan nilai null, karena sangat sedikit dan tidak signifikan dibandingkan keseluruhan dataset.

#### 4. Menangani Duplikat Data (Duplicated Data)
Pada tap ini kita akan lakukan pengecekan data ganda serta menanganinya pada variabel dataset movies dan ratings. Setalah dilakukan pengecekan terdapat 28 data ganda pada variabel dataset movies dan tidak ada data ganda pada variabel dataset ratings. Terakhir kita lakukan penghapusan data ganda pada variabel dataset movies.

#### 5. Menangani Outliers
Pada tahap ini, sebelum kita tangani outlier, kita lihat statistik deskriptif dari dataset variabel df_movies dan ratings kita mengunakan fungsi `describe()`.

* Variabel movies
   
![des_movies](https://github.com/user-attachments/assets/82b83bfb-2c0c-4c82-882b-bb0feda63a01)

* Variabel ratings
   
![des_rating](https://github.com/user-attachments/assets/3ff035f5-9f5e-4f90-a641-043753e98149)

Berdasarkan tampilan deskriptif dataset movies dan ratings dapat dilihat tidak mencolok ada pesebaran nilai yang menimbulkan `outlier`.

### Data Preprocesing
Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Langkah-langkah yang dilakukan dalam proyak ini adalah sebagai berikut:
* Mengurutkan pengguna berdasarkan ID Pengguna
* Mengubah fitur genres movie ke bentuk list
* Melakukan penggabungan dataset ratings dan movies
* Mengambil Dataset sesuai kebutuhan
  Pada proyek ini, data yang diambil sebanyak **20000 data** dari gabungan dataset `movies` dan `ratings` dengan teknik sampling menggunakan fungsi `shuffle` dari library
  `sklearn.utils` untuk mengambil data secara acak dataset besar untuk mempermudah pengolahan dan mencegah `crash`. Setelah semua langkah diatas dieksekusi maka dapat dilihar hasilnya
  pada gambar berikut:

  ![data_final](https://github.com/user-attachments/assets/6fb9e788-f5ce-43c8-b2ea-ff122c545db1)

  Dari gambar diatas dapat dilihat dataset kita sudah tergabung yang terdiri atas jumlah 20000 baris dan 7 kolom yakni kolom `userId`, `movieId`, `rating', `genres`, `title`,
  `vote_average` dan `vote_count`.

### Content Based Filtering
Content-Based Filtering adalah metode dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item yang telah disukai atau dinilai oleh pengguna. Teknik yang digunakan yaitu teknik `TF-IDF` (Term Frequency-Inverse Document Frequency) untuk menentukan bobot fitur dan menghitung kesamaan antara item dalam hal ini adalah `genres`.

Berikutnya, kita bisa melanjutkan ke tahap persiapan dengan membuat variabel preparation yang berisi dataframe df_sample_final kemudian mengurutkan berdasarkan movieId. Hasilnya dpat dilihat pada gambar berikut:

![preparation_cbf](https://github.com/user-attachments/assets/d872735a-8eb3-4dee-b6b3-914a7ce2fc9e)

Selanjutnya, kita perlu melakukan konversi data series menjadi list. Dalam hal ini, kita menggunakan fungsi `tolist()` dari library `numpy`. Setelah konversi dilakukan diperoleh variabel `movieId`, `movie_name` dan `movie_genres` dengan jumlah masing-masing sebanyak 2209.

Tahap terakhir, kita akan membuat dictionary untuk menentukan pasangan `key-value` pada data `movie_id`, `movie_name` dan `movie_genres` yang telah kita siapkan sebelumnya. Hasilnya dapat dilihat pada gambar berikut:

![movie_new](https://github.com/user-attachments/assets/9afd1e83-3722-4c90-aa02-d3d79ba3f915)

Selanjutnya langkah berikut kita gunakan fungsi `TfidfVectorizer` untuk mengkonversi `genres`. Namun sebelum itu genres perlu kita konversi dari list ke siting akar dapay diproses.

![TfidfVectorizer_Genres](https://github.com/user-attachments/assets/0d44dabc-7156-438c-9d5f-51266a1a0a7e)

Setelah mendapat index seluruh genre film, akan difit lalu ditransformasikan ke bentuk matriks sehingga diperoleh ukuran (2209, 22) serta mengubah vektor tf-idf dalam bentuk matriks dengan fungsi todense(). Hasilnya dapat dilihat pada gambar berikut:

![matriks_todense](https://github.com/user-attachments/assets/662356e9-6909-4e0d-89d5-58814c7530fd)

Setelah dibentuk matriks, dibuat tabel berisi judul film beserta genrenya berdasarkan TF-IDF yang telah diinisiasi. Hasilnya dapat dilihat pada gambar berikut:

![tabel_judul_film_genres](https://github.com/user-attachments/assets/8676ab8c-5a8b-4c33-9adc-f4d89ab0ac30)

### Collaborative Filtering (CF)
Pada tahap ini data prerataion CF, Langkah pertama, kita cek dataset kita dengan fungsi `info()`, hasilnya ditampilkan pada gambar berikut:

![sampel_final](https://github.com/user-attachments/assets/f3083afc-7726-40d1-92c9-edd5e6ec6c66)

Dari hasil diatas, terdapat 20000 baris dan 7 kolom dan memiliki 3 tipe data `float64`, 2 tipe data `int64` dan 2 tipe data `object`. Langkah kedua Kedua, kita hapus kolom yang tidak dibutuhkan dalam pelatihan yaitu `genres` dan `title`. Langkah berikutnya kita urutkan berdasarkan kolom `userId` untuk kita masuk pada tahap encoding `userId` dan `movieId`.

#### Encoding userId dan movieId
Pada tahap ini kita akan lakukan encoding pada `userId` dan `movieId`. Hasilnya dapat ditampilkan pada gamabr dibwah ini:
* Encoding `userId`

![encoding-userId](https://github.com/user-attachments/assets/40500668-535e-4ee3-9380-d77f959e49b6)

* Encoding `movieId`

![encoding_movie_id](https://github.com/user-attachments/assets/879755e6-b238-4d15-839f-d2d8c31dbf41)

Selanjutnya kita ambil total_user, total movie dan nilai rating minimum dan maksimum untuk proses pembagian dataset sebelum melakukan pelatihan model. Hasilnya diperoleh yaitu 669 pengguna, 2256 film serta nilai rating minimum sebesar 0.5 dan maksimum sebesar 5.0.

#### Membagi Data untuk Training dan Validasi
Pada tahap ini kita membagi data training dan data validasi untuk proses pelatihan model. Namun sebelum itu kita perlu mengacak dataset kita sehingga menjadi data yang valid. Hasilnya seperti pada gambar berikut:

![dataset_acak](https://github.com/user-attachments/assets/fcbdae71-655c-43a4-b227-17bc8b13359f)

Selanjutnya kita buat variabel x untuk mencocokkan data user dan Movie menjadi satu value, kemudian variabel y untuk membuat rating dari hasil. Terakhir kita Membagi menjadi `80%` data train dan `20%`` data validasi.


## Modeling and Result
Pada tahap ini ada dua model yang dipakai untuk dilatih, di evaluasi dan memberikan rekomendasi kepada pengguna film. Kedua model tersebut dapat dijelaskan sebagai berikut:

### Modeling Content Based Filtering (CBF)
Pada tahp ini kita gunakan metode `Consine Similarity`,  yang berfungsi mengukur kesamaan antara dua dokumen atau vektor dalam ruang multidimensi. Pada proyek ini, kita akan gunakan untuk sistem rekomendasi berbasis `Content-Based Filtering` yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item genre film yang telah disukai atau dinilai oleh pengguna. Menurut Firmansyah(2018), `Cosine similarity` digunakan dalam ruang positif, dimana hasilnya dibatasi antara nilai `0` dan `1`. Kalau nilainya `0` maka dokumen tersebut dikatakan mirip jika hasilnya 1 maka nilai tersebut dikatakan tidak mirip Perhatikan bahwa batas ini berlaku untuk sejumlah dimensi.

Langkah pertama kita menghitung cosine similarity pada matrix tf-idf yang dapat dilihat pada gambar berikut:

![consimlirity_mat](https://github.com/user-attachments/assets/eaac13f1-722e-4bda-9cf9-533d3eca9c7b)

Langkah kedua kita lihat hasil cosine similarity pada matrix tf-idf anta judul film yang mirip berdasarkan genre.

![tampil_similirity_film](https://github.com/user-attachments/assets/d8fbc1b2-47c3-46db-b31a-562ae0122d28)

Selanjutnya kita buat fungsi rekomendasi film berdasarkan kemiripan genre dengan menerapkan fungsi Top-N rekokemendasi serta menguji dan mengevaluasi model yang dibuat.

#### Pengujian Sistem Rekomendasi
Tahap ini kita ambil satu judul film untuk dilakukan pengujian seperti yan terlihat pada gambar berikut:

![film_uji](https://github.com/user-attachments/assets/272fca27-4807-4b62-81cc-0fc62957413d)

5 hasil rekomendasi film dapat dilihat pada gambar berikut:

![Top-N CBF](https://github.com/user-attachments/assets/ff758677-3d68-4aa1-97e5-72f9caf44e96)

Dapat dilihat genre film uji yang kita masukan adalah `Crime`, `Drama`, `Romance`. Pada hasilnya genre ini tersebar di dalam 5 fil yang dihasilkan. Oleh karena itu kita perlu mengukur hasil evalusi modelnya dengan fungsi `metrix precision`.

#### Evaluation
Padata tah ini kita mengunakan fungsi metrix precision. Presisi: Mengukur seberapa presisi/akurat model.  Rasionya antara positif yang diidentifikasi dengan benar (positif benar) dan semua positif yang diidentifikasi. Metrik presisi mengungkapkan berapa banyak kelas yang diprediksi diberi label dengan benar[7].
`Precision = #True_Positive / (#True_Positive + #False_Positive)`

Hasil pengujian diperoleh nilai precision sebesar 90.00%.

### Modeling Collaborative Filtering (CF)
Pada tahap ini menggunakan pendekatan Model-Based Deep Learning Collaborative Filtering. Metode `Deep Learning Neural Network (DNN)` yang merupakan subkategori dari machine learning yang menggunakan struktur ANN yang sangat dalam, dikenal sebagai deep neural networks. Deep learning melibatkan jaringan saraf dengan banyak lapisan tersembunyi, yang memungkinkan model untuk belajar dan mengenali pola yang sangat kompleks dan abstrak dari data `[2]`.

Pada tahap ini, model menghitung skor kecocokan antara user dan movie teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan movie. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan movie. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [`0,1`] dengan fungsi aktivasi sigmoid. Di sini, kita membuat class `RecommenderNe`t dengan `keras Model class`. Kedua kita lakukan proses compile terhadap model. Model ini menggunakan `Binary Crossentropy` untuk menghitung `loss function`, `Adam (Adaptive Moment Estimation)` sebagai `optimizer`, serta Mean Absolute Error(MAE) dan Root Mean Squared Error (RMSE) sebagai metrics evaluation.

Langkah berikutnya, mulailah proses training. Pada proses ini kita gunakan fungsi callbacks, dimana jika kinerja model tidak mengalami keanaikan maka pelatiahan dihentikan. Pada proses training parameter yang digunakan yakni `batch_size=8`, `epoch = 50`, `shuffle = True` dan `verbose=1`

Proses latihan model dapat dilihat pada gambar berikut:

![pelatihan-model](https://github.com/user-attachments/assets/02d28cfe-8958-45b3-ad21-75c51c83abe5)

Dapat dilihat, hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1382 dan root_mean_squared_error: 0.1749

#### Penujian Sistem Rekomendasi
Pada tahap ini kita akan lakukan pengujian terhadap model yang telah dibuat. Sebelumnya, pengguna telah memberi rating pada beberapa film yang telah mereka nonton. Kita menggunakan rating ini untuk membuat rekomendasi film yang mungkin cocok untuk pengguna.
Berikut adalah Top-10 Rekomendasi film terbaik kepada pengguna yang memiliki kesamaan:

![Top-N CF](https://github.com/user-attachments/assets/8277685f-3708-41b5-b1e3-cd91278c3a90)

#### Evaluation
Pada tahap ini, kita menggunakan metrik evaluasi  untuk mengukur kinerja model (formula dan cara metrik tersebut bekerja). Pada tahap ini kita akan lakukan visualisasi metrik dengan teknik Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). 

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
  
Selanjutnya kita lakukan visualisasi metrik seperti Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). Kedua metrik ini sangat penting dalam mengevaluasi kinerja model prediksi. Kedua metrik ini memberikan informasi tentang seberapa baik model dapat memprediksi nilai aktual, dan visualisasi dapat membantu dalam memahami perbandingan antara keduanya serta tren kesalahan dari waktu ke waktu `[3]`. 

Hasil dari kedua metiks tersebut dapat ditampilakn pada gambar dibawah ini:

* Gambar Visualisasi Metriks MAE
    
![MAE-MAtriks](https://github.com/user-attachments/assets/646ab35b-5fdc-448f-8ac7-0e9572ef5977)

Berdasarkan hasil `fitting` nilai konvergen metrik MAE berada sedikit dibawah 0.1370 untuk training dan sedikit diatas 0.1657 untuk validasi.

* Gambar Visualisasi Metriks RMSE

![RMSE_Matriks](https://github.com/user-attachments/assets/f38d7003-d4b1-4952-b8cf-93dc0d9c51c7)

Berdasarkan hasil fitting nilai konvergen metrik RMSE berada sedikit diatas 0.1755 untuk training dan sedikit dibawah 0.221 untuk validasi.

## Kesimpulan
Berdasarkan hasil yang diperoleh setelah melakukan proses pengolahan data sampai proses evaaluasi dapat dismpulkan bahwah:
1. Pengunaan Teknik EDA kita dapat melihat distribusi data pada data rating dan data genre film dengan jelas. Nilai ratings paling umum diberikan pengguna adalah rating `4.0` dengan presentasi `28.7%`, rating `3.0` dengan presentasi `20.1%`, rating `5.0` dengan prestansi `15.1%`. Sedangkan nilai rating yang lain berada dibawah pada presentasi `12.0%`. Sedangkan genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar `20243` dan `13137`, sementara genre yang lain berada dibawah `10000`. Film dengan judul Terminator `3: Rise of the Machines` memiliki rating teratas dengan mean rating `4.256` dan total rating sebanya `324`. Total rating terting berada diatas `250` sebanyak `5 film`, sedangkan rata-rata terbanyak pengguna memberi rating terhadap film berada diretang nilai `2 - 4.5` rating.
2. Dengan preparation data yang sistematis, seperti menangani nilai hilang (missing values), menghapus atau menangani outlier, dan melakukan encoding pada data kategorikal, proses analisis data menjadi lebih efisien dan akurat. Data yang bersih dan siap digunakan akan mengurangi risiko kesalahan dalam model analitik.
3. Dengan mengunakan metode Content-Based Filtering dapat memberikan 10 rekomendaasi film kepada sesama pengguna berdasarkan kesaaman perilaku pengguna dengan nilai presesion matriks sebesar 90.00%.
4. Penggunaan Model-Based Deep Learning Collaborative Filtering memberikan hasil rekomendasi yang lebih akurat dan relevan bagi pengguna. Hal ini di buktikan dngan hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1364 dan root_mean_squared_error: 0.1755 dan juga tampilan matriks visualisasi yang menunjukan nilai MAE dan RMSE berada dibawah 0.2 pada epoh ke-17.

## Daftar Pustaka
1. D. A. R. Ariantini, A. S. M. Lumenta and A.Jacobus, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016.
2. Neural Network: Cikal Bakal Revolusi Deep Learning. Tersedia: [Tautan](https://www.dicoding.com/blog/neural-network-cikal-bakal-revolusi-deep-learning/). Diakses pada: Desember 2024.
3. Perbedaan MAE, MSE, RMSE, dan MAPE pada Data Science. Tersedia: [Tautan]([https://pages.github.com/](https://www.trivusi.web.id/2023/03/perbedaan-mae-mse-rmse-dan-mape.html)). Diakses pada: Desember 2024.
4. Firmansyah Fataruba, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016. "PENERAPAN METODE COSINE SIMILARITY UNTUK PENGECEKAN KEMIRIPAN JAWABAN UJIAN SISWA", JATI (Jurnal Mahasiswa Teknik Informatika) Vol. 2  No. 2, September 2018.
5. Nathania, R.A. 2024. Sistem Rekomendasi Film Dengan Collaborative Deep Learning. (Skripsi, Fakultas Teknologi Informasi dan Sains, Universitas Katolik Parahyangan: Bandung).
6. Salim .E, Paragantha. J, Lauro M, "Perancangan Sistem Rekomendasi Film menggunakan metode Contentbased Filtering" (Paper, Jurusan Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara: Jakarta Barat).
7. Metrik evaluasi. Tersedia: [Tautan](https://learn.microsoft.com/id-id/azure/ai-services/language-service/custom-text-classification/concepts/evaluation-metrics). Diakses pada: Desember 2024.
