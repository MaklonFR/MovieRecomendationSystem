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

### Exploratory Data Analysis - EDA
Analisis eksploratif data (EDA) adalah tahap penting dalam analisis data yang bertujuan untuk memahami dan mengeksplorasi karakteristik dataset sebelum melakukan analisis yang lebih mendalam. Dataset yang digunakan dalam proyek ini yaitu dataset fIlm yang dapat dijelaskan sebagai barikut: 

#### Informasi Dataset
Informasi dari dataset film ini dapata dilihat pada tabel berikut:
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

  *movie_metadata.csv*: File Metadata Film utama. Berisi informasi tentang 45.000 film yang ditampilkan dalam kumpulan data Full MovieLens. Fitur-fiturnya meliputi poster, latar 
  belakang, anggaran, pendapatan, tanggal rilis, bahasa, negara produksi, dan perusahaan.
  
  *keywords.csv*: Berisi kata kunci plot film untuk film MovieLens kami. Tersedia dalam bentuk Objek JSON yang dirangkai.
  
  *credits.csv*: Terdiri dari Informasi Pemeran dan Kru untuk semua film kami. Tersedia dalam bentuk Objek JSON yang dirangkai.
  
  *links.csv*: File yang berisi ID TMDB dan IMDB dari semua film yang ditampilkan dalam kumpulan data Full MovieLens.
  
  *links_small.csv*: Berisi ID TMDB dan IMDB dari sebagian kecil 9.000 film dari Kumpulan Data Lengkap.
  
  *rating_small.csv*: Bagian dari 100.000 rating dari 700 pengguna pada 9.000 film.


#### Membaca Dataset
Selanjutnya pada tahap ini, kita akan baca data-data diatas menggunakan fungsi pandas.read_csv. Hasilnya dapat ditampilkan pada gambar berikut:

![Rincian File Dataset](https://github.com/user-attachments/assets/0870ef20-4978-4d30-a5d6-847d2d6aa956)

Hasil dari gambar diatas merupakan jumlah data dalam file-file dataset film.

Selanjutnya, pada proyek ini, kita hanya menggunakan 2 file csv yaitu ratings_small.csv (ratings) dan movies_metadata.csv (movies). Dari kedua file ini kita akan melihat informasi apa saja yang ada di dalammya.

**1. ratings (ratings_small.csv)**

Berikut merupakan informasi yang ada dalam dataset ratings:

![dataset-ratings](https://github.com/user-attachments/assets/559f75c8-abca-4f8d-a0fa-b4d514bc605c)

Berdasarkan gambar diatas, dataset ratings terdiri dari 100004 baris dan 4 kolom yang dapat dijelaskan sebagai berikut:
| Variabel                    | Keterangan     |
| -----------------------     | ------------------------------------------------------------------------- |
| userId                      | ID unik untuk pengguna yang memberikan penilaian (rating). Ini digunakan untuk mengidentifikasi pengguna secara anonim.  |
| movieId|ID unik untuk film yang dinilai oleh pengguna untuk mendapatkan informasi lebih detail tentang film tersebut.                                  |
| rating | Nilai yang diberikan oleh pengguna untuk film tertentu dengan skala 1 hingga 5, di mana angka yang lebih tinggi menunjukkan penilaian yang lebih positif. |
| timestamp                   | Waktu ketika penilaian diberikan, direpresentasikan dalam format UNIX timestamp (jumlah detik sejak 1 Januari 1970).   |

**2. movies (movies_metadata.csv)**

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

#### Mengambil Fitur Sesuai Kebutuhan
Pada pronyek ini, dataset *movies_metadata (movies)* kita hanya mengambil beberapa fitur atau kolom sesuai kebutuhan analsis pengolahan data yakni `['id', 'genres', 'title', 'vote_average', 'vote_count']`. Fitur-fitur tersebut dapat dilihat pada gambar di bawah ini yang menampilkan 5 data pada setiap fitur

![gambar-1](https://github.com/user-attachments/assets/87240afd-52bc-4419-a8e3-c414848c605a)

#### Melihat Tipe Data Pada Dataset
Tahap ini kita akan melihat Informsi tipe data pada dataset ratings dan movies yang akan kita gunakan. Selengkapnya dapat dilihat pada gambar berikut.

![gambar-2](https://github.com/user-attachments/assets/cf300843-0864-47e9-b27e-b056a5ef5cb6)

Dapat dilihat pada informasi dataset **movies**, 3 variable dengan tipe data object dan 2 variabel bertipe float64. Selain itu ada perbedaan pada variabel `title`, `vote_average` dan `vote_count` yang memiliki data 45460 dengan variabel id dan genres dengan data 45466. Sedangkan pada informasi dataset **ratings** terdapat 1 variabel dengan tipe data float64 dan 3 variable dengan tipe data int64.

#### Univariate Analysis
Univariate analysis adalah teknik yang digunakan untuk menganalisis satu variabel pada suatu waktu. Tujuannya adalah untuk menggambarkan dan memahami sifat dasar dari variabel tersebut, termasuk distribusi nilai, tendensi pusat, dan penyebaran data. Langkah awal kita menyesuaikan tipe data `primary key` dan `foregein key`. Jika dilihat pada informasi sebelumnya, dataset movies atribut id (`primary key`) dengan type data `object` berbeda pada dataset ratings atribut `movieId` dengan type data `int64`. Oleh karena itu, kita perlu menyamakan tipe data tersebut dengan cara mnyamakan nama atribut movieId dan tipe data `int64`. Selanjutnya kita lakukan analisis statistik deskriptif dan visualisasi data.

##### Statistik Deskriptif
Statistik deskriptif memberikan ringkasan numerik dari variabel yang dianalisis. Tampilan statistik deskriptif dapat dilihat pada gambar berikut:

![statistik_deskriptif](https://github.com/user-attachments/assets/3f4b50b2-331b-45aa-8297-9ca95ab8cb2d)

#### Visualisasi Data
Visualisasi adalah alat penting dalam univariate analysis yang memungkinkan analis untuk melihat pola dan distribusi data secara intuitif. Jenis grafik yang dipakai yakni `bar chart`  untuk menampilkan frekuensi atau proporsi kategori dalam variabel kategorikal.
Pada tahap ini kita akan menggunakan grafik untuk menggambarkan distribusi genre dan rating film, serta hubungan antara fitur-fitur dalam dataset.
* Distribusi Ratings
  Pada langkah ini kita akan mendistribusi ratings dengan tujuan, mengidentifikasi nilai rating yang paling umum diberikan oleh pengguna, menilai apakah data rating cenderung condong ke
  satu nilai (misalnya, lebih banyak rating tinggi atau rendah) dan membantu memahami pola preferensi pengguna. Proses distribusi rating mengunakan library `matplotlib` seperting yang
  ditampilkan pada gambar berikut:

  ![gambar-8](https://github.com/user-attachments/assets/26495257-b5cb-46a9-8fe5-429d8aa37f51)

  Berdasarkan diagram plot rating diatas, dapat dilihat bahwa nilai ratings paling umum diberikan pengguna adalah rating 4.0 dengan presentasi 28.7%, rating 3.0 dengan presentasi 20.1%,
  rating 5.0 dengan prestansi 15.1%. Sedangkan nilai rating yang lain berada dibawah pada presentasi 12.0%.

  * Distribusi Gengres
    Distribusi genre film adalah aspek penting dalam sistem rekomendasi, karena membantu memahami preferensi pengguna dan pola konsumsi film. Pada proyek ini menggunakan metode
    Visualisasi Data dalam menampilkan grafik batang yang menggambarkan proporsi masing-masing genre secara visual, sehingga memudahkan pemahaman.
    Pada tahap ini kita akan membersihkan, memproses, dan menormalkan data dalam kolom genres pada DataFrame df_movies Ada beberapa fungsi yang kita pakai yakni:
    * `fillna('[]')`, berfungsi untuk mengisi nilainull atau NaN dalam kolom genres dengan string kosong dalam format list (`[]`).
    * `apply(literal_eval)`, fungsi literal_eval dari pustaka ast untuk mengubah string yang terlihat seperti Python literal menjadi tipe data list.
    * `apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else []`, fungsi lambda ini memproses setiap nilai dalam kolom genres: Jika nilai adalah sebuah daftar
    * `(isinstance(x, list))`, maka ambil nilai dari kunci name untuk setiap elemen. Jika nilai bukan daftar, mengembalikan daftar kosong (`[]`).
    
    Selanjutanya kita ubah setiap elemen dalam daftar (genre) menjadi baris terpisah dengan fungsi `explode()`, kemudian menghitung jumlah kemunculan setiap genre dengan fungsi
    `value_counts()` dan terakhir kita membuat diagram batang untuk menampilkan distribusi genre dengan plot bar `plot(kind='bar')`.
    Langkah pertama kita buat variabel dataframe baru untuk melakukan analisis visualisasi data. Kemudian kita konversi fitur(variabel) genres ke dalam bentuk list sehingga dapat
    dianalisi. Berikut adalah gambar distribrusi genres menggunakan grafik bar.

    ![gambar-9](https://github.com/user-attachments/assets/8fc91fa0-b9bc-404f-8325-951ac2e33440)

    Dari grafik diatas, dapat dilihat bahwa genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar 20243 dan 13137. Sedangkan genre yang lain
    berada dibawah 10000.

#### Analisis Daftar film dengan skor tertinggi di seluruh rentang film
Untuk membuat daftar film dengan skor tertinggi menggunakan metode Weighted Score. Metode ini merupakan perhitungan skor berbobot untuk menggabungkan nilai-nilai yang berbeda
berdasarkan pentingnya masing-masing komponen. Dalam konteks film, kita perlu menghitung skor berbobot berdasarkan informasi yang tersedia, seperti rata-rata penilaian
(vote_average), jumlah suara (vote_count), dan jumlah suara rata-rata minimum yang diperlukan untuk dipertimbangkan dalam daftar.
Keterangan:
      v = jumlah suara untuk film tertentu (vote_count)
      m = jumlah suara minimum untuk masuk ke daftar (threshold)
      R = rata-rata skor film tersebut (vote_average)
      C = rata-rata skor semua film dalam dataset (rata-rata global)
Hasilnya dapat dilihat pada gambar berikut:

![gambar-10](https://github.com/user-attachments/assets/9cac3d6e-5615-4f76-b913-4c45d84e36fa)

Gambar diatas menunjukan 5 filim dengan skor tertinggi yang diberikan oleh pengguna.

#### Analisis Rating Tertinggi
Selanjutnya kita gabungkan dataset df_movies dan ratings dengan fungsi pandas pd.merge dan mencari 10 film dengan rating tertinggi. Alisis rating tertinggi dapat dilihat pada gambar berikut:

![10-analisis rating](https://github.com/user-attachments/assets/257800b2-4b5e-4ecc-8598-3c679031faca)

Dapat dilihat pada gambar diatas dari 10 rating tertinggi film yang ada, film dengan judul **Terminator 3: Rise of the Machines*** memiliki rating teratas dengan *mean rating* 4.256 dan total rating sebanya 324.

#### Membandingkan Peringkat rata-rata vs Jumlah total peringkat
Pada tahap ini kita akan membandingkan rata-rata rangkin dan total rangking menggunakan joinplot untuk melihat pesebaran data yang dapat dilihat pada gambari dibawah ini:

![mean vs total rating](https://github.com/user-attachments/assets/336ee099-c9a0-4fbb-b4be-00c80b0048a9)

Berdasarkan grafik pesebaran data diatas total rating terting berada diatas 250 sebanyak 5 film, sadangkan rata-rata terbanyak pengguna memberi rating terhadap film berada diretang nilai 2 - 4.5 rating.

## Data Preparation
Data preparation adalah langkah penting dalam pengembangan sistem rekomendasi film yang efektif. Proses ini mencakup beberapa tahap, mulai dari pengumpulan data hingga pemrosesan akhir sebelum data digunakan dalam model machine learning.

### Data Cleaning
Setelah data terkumpul, langkah berikutnya adalah membersihkan data. Ini mencakup:
Mengatasi Missing Values: Mengisi nilai yang hilang dengan metode seperti imputasi menggunakan rata-rata atau median rating.
Menghapus Duplikat: Memastikan tidak ada entri duplikat dalam dataset untuk menghindari bias.

#### Menangani Missing Value
Mengatasi missing values (nilai yang hilang) merupakan langkah penting dalam pengolahan data untuk sistem rekomendasi film. Keberadaan missing values dapat mempengaruhi kualitas model dan akurasi rekomendasi yang dihasilkan. Berikut langkah-langkah yang perlu diambil dalam menganangi missing value pada dataset movies dan ratings.
Pertama kita lakukan untuk dataset movies
* Menggunakan fungsi `isnull().sum()` untuk menampilkan jumlah nilai yg hilang pada setiap fitur(variabel)

![gambar-3](https://github.com/user-attachments/assets/67787837-b065-4abd-97ac-42cd7defbe81)

Dari hasil diatas, nilai null terdapat pada variabel title, vote_average dan vote_count memiliki nilai null = 6 pada.
* Menampilkan isi dataset yang memuat nilai yang hilang dengan fungsi `isnull().any(axis = 1)`

![gambar-4](https://github.com/user-attachments/assets/d0932703-e38e-428f-9026-a247b3efef7a)

* Menghapus missing value pada dataset menggunakan fungsi `dropna(subset=[‘nama_fitur’]`

![gambar-5](https://github.com/user-attachments/assets/271bd039-8749-4fae-be95-f2800ea869be)

Kedua, kita lakukan pada dataset ratings, setelah dicek tidak memiliki missing value seperti yang terlihat pada gambar dibah ini:

![gambar-6](https://github.com/user-attachments/assets/eaa9c9d9-13aa-47ce-9482-fcc7e2785bae)

#### Manangani Duplikat Data
Menangani duplikat data adalah langkah penting dalam proses pembersihan data untuk memastikan integritas dan kualitas dataset yang digunakan dalam sistem rekomendasi film. Duplikat data dapat menyebabkan bias dalam analisis dan menghasilkan rekomendasi yang tidak akurat. Berikut adalah langkah-langkah untuk menangani duplikat data.
Pertama, kita lakukan untuk dataset **movies**,
* Indentfikasi dataset dengan menggunakan fungsi ` duplicated().sum()`. Setelah dilakukan pengecekan ternyata memiliki 28 data ganda. 
* Menampilkan isi dataset yang memiliki data ganda pada setiap fitur (kolom), seperti yang terlihat pada gambar berikut:

![gambar-7](https://github.com/user-attachments/assets/a7171168-7ca6-4083-ba09-580b30b9366c)

* Menghapus duplikat data pada dataset dengan menggunakan fungsi `drop_duplicates(inplace = True)`

Kedua, lakukan pada dataset **ratings**, setelah dicek tidak memiliki data duplikat.

### Data Preprocessing
Proses ini bertujuan untuk menyiapkan data mentah agar dapat digunakan secara efektif dalam model machine learning. Langkah-langkah yang dilakukan dalam proyak ini adalah sebagai berikut:
* Mengurutkan pengguna berdasarkan ID Pengguna
* Mengurutkan pengguna berdasarkan ID Pengguna
* Melakukan penggabungan dataset ratings dan movies
* Mengambil Dataset sesuai kebutuhan
Pada proyek ini, data yang diambil sebanyak **20000 data** dari gabungan dataset `movies` dan `ratings` dengan teknik sampling menggunakan fungsi `shuffle` dari library `sklearn.utils` untuk mengambil data secara acak dataset besar untuk mempermudah pengolahan dan mencegah `crash`.
Setelah semua langkah diatas dieksekusi maka dapat dilihar hasilnya pada gambar berikut:

![gambar-11](https://github.com/user-attachments/assets/1aa3a1c4-7778-47d2-9646-47fdac0a5cdc)

Dari gambar diatas dapat dilihat dataset kita sudah tergabung yang terdiri atas jumlah 20000 baris dan 7 kolom yakni kolom `userId`, `movieId`, `rating', `genres`, `title`, `vote_average` dan `vote_count`.

Berikutnya, kita bisa melanjutkan ke tahap persiapan dengan membuat variabel preparation yang berisi dataframe df_sample_final kemudian mengurutkan berdasarkan movieId. Hasilnya dpat dilihat pada gambar berikut:

![gambar-12](https://github.com/user-attachments/assets/bea4db5c-40e8-4c30-9e4e-2a934140e691)

Selanjutnya, kita perlu melakukan konversi data series menjadi list. Dalam hal ini, kita menggunakan fungsi `tolist()` dari library `numpy`. Setelah konversi dilakukan diperoleh variabel `movieId`, `movie_name` dan `movie_genres` dengan jumlah masing-masing sebanyak 2248.

Tahap terakhir, kita akan membuat dictionary untuk menentukan pasangan `key-value` pada data `movie_id`, `movie_name` dan `movie_genres` yang telah kita siapkan sebelumnya. Hasilnya dapat dilihat pada gambar berikut:

![gambar-13](https://github.com/user-attachments/assets/bb2d5e10-cad2-4fe0-b337-b3c12f77ef03)

## Modeling and Result
Pada tahap ini ada dua model yang dipakai untuk dilatih, di evaluasi dan memberikan rekomendasi kepada pengguna film. Kedua model tersebut dapat dijelaskan sebagai berikut:

### Content Based Filtering
Content-Based Filtering adalah metode dalam sistem rekomendasi yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item yang telah disukai atau dinilai oleh pengguna. Teknik yang digunakan yaitu teknik `TF-IDF` (Term Frequency-Inverse Document Frequency) untuk menentukan bobot fitur dan menghitung kesamaan antara item dalam hal ini adalah `genres`.
Pada tahp ini kita gunakan metode `Consine Similarity`, yang berfungsi mengukur kesamaan antara dua dokumen atau vektor dalam ruang multidimensi. Pada proyek ini, kita akan gunakan untuk sistem rekomendasi berbasis Content-Based Filtering yang memberikan rekomendasi berdasarkan karakteristik atau konten dari item genre film yang telah disukai atau dinilai oleh pengguna. Menurut Firmansyah(2018), Cosine similarity digunakan dalam ruang positif, dimana hasilnya dibatasi antara nilai `0` dan `1`. Kalau nilainya `0` maka dokumen tersebut dikatakan mirip jika hasilnya `1` maka nilai tersebut dikatakan tidak mirip Perhatikan bahwa batas ini berlaku untuk sejumlah dimensi.

![Consine similarity](https://github.com/user-attachments/assets/1133e6d7-8d16-4610-9790-7da9d9c3783c)

**Interpretasi Hasil**
Nilai cosine similarity berkisar antara -1 hingga 1.
* Nilai `1` menunjukkan bahwa kedua vektor identik (sudut 0 derajat).
* Nilai `0` menunjukkan bahwa kedua vektor tidak memiliki kemiripan (sudut 90 derajat).
* Nilai `-1` menunjukkan bahwa kedua vektor berlawanan arah (sudut 180 derajat).

Hasil dari penerapan model ini dapat dilihat pada gambar berikut:

[gambar--- CBF]


### Penerapan Model Collaborative Filtering
Pada tahap ini menggunakan pendekatan Model-Based Deep Learning Collaborative Filtering. Metode `Deep Learning Neural Network (DNN)` yang merupakan subkategori dari machine learning yang menggunakan struktur ANN yang sangat dalam, dikenal sebagai deep neural networks. Deep learning melibatkan jaringan saraf dengan banyak lapisan tersembunyi, yang memungkinkan model untuk belajar dan mengenali pola yang sangat kompleks dan abstrak dari data `[2]`.

Pada tahap ini, model menghitung skor kecocokan antara user dan movie teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan movie. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan movie. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [`0,1`] dengan fungsi aktivasi sigmoid. Di sini, kita membuat class `RecommenderNe`t dengan `keras Model class`. Kedua kita lakukan proses compile terhadap model. Model ini menggunakan `Binary Crossentropy` untuk menghitung `loss function`, `Adam (Adaptive Moment Estimation)` sebagai `optimizer`, serta Mean Absolute Error(MAE) dan Root Mean Squared Error (RMSE) sebagai metrics evaluation.

Langkah berikutnya, mulailah proses training. Pada proses ini kita gunakan fungsi callbacks, dimana jika kinerja model tidak mengalami keanaikan maka pelatiahan dihentikan. Pada proses training parameter yang digunakan yakni `batch_size=8`, `epoch = 50`, `shuffle = True` dan `verbose=1`

Proses latihan model dapat dilihat pada gambar berikut:

![pelatihan-model](https://github.com/user-attachments/assets/02d28cfe-8958-45b3-ad21-75c51c83abe5)

Dapat dilihat, hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1382 dan root_mean_squared_error: 0.1749

## Evaluation
Pada tahap ini, kita menggunakan metrik evaluasi  untuk mengukur kinerja model (formula dan cara metrik tersebut bekerja). Pada tahap ini kita akan lakukan visualisasi metrik dengan teknik Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). Kedua metrik ini sangat penting dalam mengevaluasi kinerja model prediksi. Kedua metrik ini memberikan informasi tentang seberapa baik model dapat memprediksi nilai aktual, dan visualisasi dapat membantu dalam memahami perbandingan antara keduanya serta tren kesalahan dari waktu ke waktu. Berikut ada rumus dari teknik tersebut.
1. Mean Absolute Error (MAE)
   MAE adalah salah satu metode evaluasi yang umum digunakan dalam data science. MAE menghitung rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual.
   Dengan kata lain, MAE menghitung berapa rata-rata kesalahan absolut dalam prediksi. Semakin kecil nilai MAE, semakin baik kualitas model tersebut.
   Rumus MAE:
   
   ![MAE](https://github.com/user-attachments/assets/0a342837-503b-487a-9dd7-b9f205dc1185)

   Dimana:
   * n adalah jumlah sampel dalam data
   * y_i adalah nilai aktual
   * ŷ_i adalah nilai prediksi
     
2. Root Mean Squared Error (RMSE)
   RMSE adalah turunan dari MSE. Seperti namanya, RMSE adalah akar kuadrat dari MSE.
   RMSE menghitung rata-rata dari selisih kuadrat antara nilai prediksi dan nilai aktual kemudian diambil akar kuadratnya. Semakin kecil nilai RMSE, semakin baik kualitas model tersebut.

   Rumus RMSE:
   
   ![RMSE](https://github.com/user-attachments/assets/058da06d-8a58-4cca-b01b-468cdcf4e0e4)

   Dimana:
   * n adalah jumlah sampel dalam data
   * y_i adalah nilai aktual
   * ŷ_i adalah nilai prediksi
  
  ### Evaluasi menggunakan Visualisasi Metriks
  Pada tahap ini kita akana lakukan visualisasi metrik seperti Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE). Kedua metrik ini sangat penting dalam mengevaluasi kinerja
  model prediksi. Kedua metrik ini memberikan informasi tentang seberapa baik model dapat memprediksi nilai aktual, dan visualisasi dapat membantu dalam memahami perbandingan antara
  keduanya serta tren kesalahan dari waktu ke waktu `[3]`. Hasil dari kedua metiks tersebut dapat ditampilakn pada gambar dibawah ini:

  * Gambar Visualisasi Metriks MAE
    
   ![metriks-MAE](https://github.com/user-attachments/assets/04a2a29b-b44b-4fde-8396-c1b1652d40ea)

  Berdasarkan hasil `fitting` nilai konvergen metrik MAE berada sedikit dibawah 0.1383 untuk training dan sedikit diatas 0.1550 untuk validasi.

  * Gambar Visualisasi Metriks RMSE

    ![metriks-RMSE](https://github.com/user-attachments/assets/bd08a600-c4d1-42e9-a681-a61b7ae06262)

  Berdasarkan hasil fitting nilai konvergen metrik RMSE berada sedikit diatas 0.1749 untuk training dan sedikit dibawah 0.185 untuk validasi.

## Kesimpulan
Berdasarkan hasil yang diperoleh setelah melakukan proses pengolahan data sampai proses evaaluasi dapat dismpulkan bahwah:
1. Pengunaan Teknik EDA kita dapat melihat distribusi data pada data rating dan data genre film dengan jelas. Nilai ratings paling umum diberikan pengguna adalah rating `4.0` dengan presentasi `28.7%`, rating `3.0` dengan presentasi `20.1%`, rating `5.0` dengan prestansi `15.1%`. Sedangkan nilai rating yang lain berada dibawah pada presentasi `12.0%. Sedangkan genre Drama dan Comedy paling banyak tersebar pada setiap film dalam dataset dengan jumah sebesar `20243` dan `13137`, sementara genre yang lain berada dibawah `10000`.
2. Dengan preparation data yang sistematis, seperti menangani nilai hilang (missing values), menghapus atau menangani outlier, dan melakukan encoding pada data kategorikal, proses analisis data menjadi lebih efisien dan akurat. Data yang bersih dan siap digunakan akan mengurangi risiko kesalahan dalam model analitik.
3. Dengan mengunakan metode Content-Based Filtering dapat memberikan rekomendaasi film kepada sesama pengguna berdasarkan kesaaman perilaku pengguna, namun masih memiliki kelemahan pada *Cold-Start Problem* (item baru) yang belum pernah direkomendasikan sebelumnya akan sulit untuk diintegrasikan ke dalam sistem.
4. Penggunaan Model-Based Deep Learning Collaborative Filtering memberikan hasil rekomendasi yang lebih akurat dan relevan bagi pengguna. Hal ini di buktikan dngan hasil pelatiahn memperoleh nilai mean_absolute_error: 0.1382 dan root_mean_squared_error: 0.1749 dan juga tampilan matriks visualisasi yang menunjukan nilai MAE dan RMSE berada dibawah 0.2 pada epoh ke-17.

## Daftar Pustaka
1. D. A. R. Ariantini, A. S. M. Lumenta and A.Jacobus, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016.
2. Neural Network: Cikal Bakal Revolusi Deep Learning. Tersedia: tautan. Diakses pada: Desember 2024.
3. Perbedaan MAE, MSE, RMSE, dan MAPE pada Data Science. Tersedia: tautan. Diakses pada: Desember 2024.
4. Firmansyah Fataruba, "PENGUKURAN KEMIRIPAN DOKUMEN TEKS BAHASA INDONESIA MENGGUNAKAN METODE COSINE SIMILARITY," E-Journal Teknik Informatika Volume 9, No 1 (2016), ISSN : 2301-8364, vol. IX, pp. 1-8, 2016. "PENERAPAN METODE COSINE SIMILARITY UNTUK PENGECEKAN KEMIRIPAN JAWABAN UJIAN SISWA", JATI (Jurnal Mahasiswa Teknik Informatika) Vol. 2  No. 2, September 2018.
5. Nathania, R.A. 2024. Sistem Rekomendasi Film Dengan Collaborative Deep Learning. (Skripsi, Fakultas Teknologi Informasi dan Sains, Universitas Katolik Parahyangan: Bandung).
6. Salim .E, Paragantha. J, Lauro M, "Perancangan Sistem Rekomendasi Film menggunakan metode Contentbased Filtering" (Paper, Jurusan Teknik Informatika, Fakultas Teknologi Informasi, Universitas Tarumanagara: Jakarta Barat).
