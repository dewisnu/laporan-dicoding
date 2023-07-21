# Laporan Proyek Machine Learning - I Gede Ari Wisnu Sanjaya



##  Proyek Overview

Buku merupakan sumber informasi yang memenuhi berbagai kebutuhan, mulai dari ilmu pengetahuan dan teknologi, seni budaya, ekonomi, politik, sosial, hingga pertahanan dan keamanan, dan lain sebagainya. Membaca buku bukan hanya mengembangkan wawasan intelektual, tetapi juga memiliki kekuatan untuk mengubah masa depan dan memperkaya pikiran, pemikiran, dan iman. 

Namun, berdasarkan data statistic UNESCO 2012 minat membaca di indonesia hanya 0.001, yang berarti setiap 1.000 orang hanya 1 orang  pelajar yang memiliki minat membaca. Hal ini sangat disayangkan mengingat Indonesia sebagai negara yang besar memiliki potensi besar untuk menjadi negara unggul. Rendahnya minat baca di kalangan masyarakat menjadi persoalan penting dalam dunia pendidikan. Oleh karena itu, diperlukan sebuah sistem yang dapat membantu merekomendasikan para pembaca agar lebih mudah mendapatkan informasi buku-buku yang akan dibaca selanjutnya[1].

Sistem rekomendasi sendiri telah digunakan secara luas dalam hampir semua area bisnis di mana seorang konsumen memerlukan informasi untuk membuat keputusan. Terdapat dua pendekatan umum yang digunakan dalam pembuatan sistem rekomendasi, yaitu _content-based filtering_ dan _collaborative filtering_. _Content-based filtering_ adalah metode yang bekerja dengan mencari kesamaan antara item yang akan direkomendasikan dengan item yang telah dilihat oleh pengguna sebelumnya berdasarkan kesamaan konten. Namun, sistem rekomendasi berbasis konten ini masih memiliki kelemahan, yaitu karena semua rekomendasi didasarkan pada konten yang serupa, pengguna tidak mendapatkan rekomendasi untuk konten yang berbeda. Selain itu, sistem rekomendasi ini kurang efektif untuk pengguna pemula, karena mereka tidak mendapatkan masukan dari pengguna sebelumnya[2].

## Business Uderstanding

### Problem Statements

- Bagaimana cara merekomendasikan buku yang disukai pengguna lain dapat diminati dan dijadikan rekomendasi untuk pengguna lainnya?

### Goals

* Membuat Sistem Rekomendasi berdasarkan user yang meberikan rating terhadap suatu buku 

### Solution Approach

Solusi yang diajukan yaitu dengan menggunakan 2 algoritma machine learning untuk sistem rekomendasi yaitu:

- _Content Based Filtering_  adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit. Algoritma ini memberikan rekomendasi berdasarkan aktivitas pada masa lalu[3].
- _Collaborative Filtering_  adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Algoritma ini memberikan rekomendasi berdasarkan nilai rating atau nilai lain, disini saya menggunakan target sebagai dasar penilaian[3].

## Data Understanding

Data atau datasets yang digunakan pada proyek <em> Machine Learning </em> ini adalah data _Book Recommendation Dataset_ yang bisa di akses di [link berikut ini (kaggle)](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Books.csv).

Variabel-variabel yang terdapat pada dataset adalah sebagai berikut:

#### Book.csv 

```
ISBN:ISBN Buku
Book-Title: Judul Buku
Book-Author: Penulis Buku
Year-Of-Publication: Tahun Publikasi
Publisher: Publisher Buku
Image-URL-S:small Gambar sampul Buku ukuran kecil , link amazon
Image-URL-M:Gambar sampul Buku ukuran sedang , link amazon
Image-URL-L:large Gambar sampul Buku ukuran Besar , link amazon
```

#### Rating.csv

```
User-ID: User id unik 
ISBN: ISBN Buku
Book-Rating: rating Buku
```

#### Users.csv

```
User-ID: User id unik 
Location: lokasi user
Age: Umur User
```

#### recsys_taxonomy2.PNG (Tidak Digunakan)



### EDA

Pada Gambar 1 meperlihatkan kebayanyakan User membaca buku dari Publisher Harlequin

![column Publisher graph chart](https://github.com/dewisnu/laporan-dicoding/assets/63925882/cfe3977b-3557-402c-9dc5-68f686401b68)



Gambar 1. Kontribusi Variabel untuk column _Publisher_



Pada Gambar 2 meperlihatkan bahwa kebanyakan User lebih banyak membaca buku _Little Women_ bisa kita liat juga beberapa value cukup balance

![Column Book-Title Graph Chart](https://github.com/dewisnu/laporan-dicoding/assets/63925882/46435481-97c3-448f-b775-93ce430f4138)

Gambar 2. Kontribusi Variabel untuk column _Book-Title_



Pada Gambar 3 meperlihatkan bahwa kebanyakan User lebih banyak membaca buku dari _Agatha Christie_

![columns Book-Author graph chart](https://github.com/dewisnu/laporan-dicoding/assets/63925882/6fdd4298-7c53-492b-beca-af8c8a1790b9)

Gambar 3.Kontribusi Variabel untuk column _Book-Author_



Pada Gambar 4  meperlihatkan bahwa kebanyakan User yang mebace buku ada di rentang usia 33 tahun

![column Age graph chart](https://github.com/dewisnu/laporan-dicoding/assets/63925882/a4e8b772-c41f-437c-aa09-355bbf8fc977)

Gambar 4. Kontribusi Variabel untuk column _Age_



## Data Preparation

### Merge Datasets for Gather Information

Untuk tahap awal saya menggabungkan beberapa data untuk tahap awal analisis. Berikut datanya, ini merupakan gabugan dari Books.csv, Ratings.csv dan Users.csv



Tabel 1. *Merger Data* 

|   NO |       ISBN |          Book-Title |          Book-Author | Year-Of-Publication |               Publisher |                                       Image-URL-S |                                       Image-URL-M |                                       Image-URL-L | User-ID | Book-Rating |                  Location |  Age |
| ---: | ---------: | ------------------: | -------------------: | ------------------: | ----------------------: | ------------------------------------------------: | ------------------------------------------------: | ------------------------------------------------: | ------: | ----------: | ------------------------: | ---: |
|    0 | 0195153448 | Classical Mythology |   Mark P. O. Morford |                2002 | Oxford University Press | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... | http://images.amazon.com/images/P/0195153448.0... |     2.0 |         0.0 | stockton, california, usa | 18.0 |
|    1 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... |     8.0 |         5.0 |  timmins, ontario, canada |  NaN |
|    2 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | 11400.0 |         0.0 |   ottawa, ontario, canada | 49.0 |
|    3 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | 11676.0 |         8.0 |             n/a, n/a, n/a |  NaN |
|    4 | 0002005018 |        Clara Callan | Richard Bruce Wright |                2001 |   HarperFlamingo Canada | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | http://images.amazon.com/images/P/0002005018.0... | 41385.0 |         0.0 |  sudbury, ontario, canada |  NaN |

Bisa dipahami banyak sekali fitur yang bisa digunakan, tetapi nantinya akan kita *drop* di langkah berikutnya, kita tidak akan menggunakan semua fitur nya.

### Handling Null Values

Handling null values yang saya gunakan yaitu dengan drop.

Tabel 2. *Summary of Null Values Every Each Column*

|       Column        | Data Type | Unique Values | Null Values | % null Values |
| :-----------------: | :-------: | :-----------: | :---------: | :-----------: |
|         Age         |  float64  |      141      |   279044    |   0.270301    |
|       User-ID       |  float64  |     92106     |    1209     |   0.001171    |
|     Book-Rating     |  float64  |      11       |    1209     |   0.001171    |
|      Location       |  object   |     22480     |    1209     |   0.001171    |
|     Image-URL-L     |  object   |    271041     |      4      |   0.000004    |
|      Publisher      |  object   |     16807     |      2      |   0.000002    |
|     Book-Author     |  object   |    102023     |      1      |   0.000001    |
|        ISBN         |  object   |    271360     |      0      |   0.000000    |
|     Book-Title      |  object   |    242135     |      0      |   0.000000    |
| Year-Of-Publication |  object   |      202      |      0      |   0.000000    |
|     Image-URL-S     |  object   |    271044     |      0      |   0.000000    |
|     Image-URL-M     |  object   |    271044     |      0      |   0.000000    |

Dapat dilihat dari data diatas sangat banyak sekali null values, maka dari itu saya akan drop data data yang null ? kenapa drop?, data ada sekitar 300 ribuan data null .

Setelah di drop total data ada sekitar 753 ribu  data.

### Filter Column

Kita akan memilih column mana saja yang menuru hasil analisis EDA dan menurut pandangan kita sebagai seorang engineer yang dapat mempengaruhi kinerja model

berikut kolom yang saya pilih

*['ISBN','Book-Title','Book-Author','Publisher','Book-Rating','User-ID','Age',]*

### Bad Value

Saya menyadari banyak sekali value kategorikal yang berbanding jauh dengan yang lainnya misalnya Publisher, ada Publisher yang hanya memiliki 1 value sedangkan Publisher lainnya diatas 100. Maka dari itu saya akan drop value yang jumlah nya kurang dari 100

### Correlation

![Corelation Matrix](https://github.com/dewisnu/laporan-dicoding/assets/63925882/7de7e544-d19e-422c-a5ce-ccf60ea29d31)

Gambar 6. <em>Correlation Matrix</em>

Correlation ini merupakan sebuah teknik atau metode untuk melihat keterkaitan, korelasi, hubungan antara <em>feature</em> dengan <em>feature</em> lainnya.

Disini kita akan menganalisis keterkaiatan <em>feature</em> terhadap kolom target, ada beberapa <em>feature</em> yang sangat kecil korelasi nya terhadap target, tetapi jika feature tersebut di drop akan sangat berpengaruh terhadap feature lainnya.

###  ###  Label Encoder

<em>Label Encoder</em> merupakan step di <em>Machine Learning </em> yang sangat krusial / wajib dilakukan, label encoder ini berfungsi untuk mengubah data kategorikal menjadi bentuk angka, seperti yang kita pelajari dulu bahwa inputan <em>Machine Learning </em> itu harus angka / numeric, maka dari itu kita perlu mengkonversi nya terlebih dahulu

#### Encoding

Sebenarnya ini step digunakan pada saat membuat <em>Collaborative Filtering</em>, tapi step nya hampir sama dengan <em>label encoder</em>, hanya saja di <em>encoding</em> ini ada beberapa step yang memang berguna nanti untuk testing / memprediksi data baru.

### Train Test Split

Saya disini menggunakan rasio 8:2, yaitu 80% untuk train dan 20% untuk validation.

Kita Perlu membagi datasets ke dalam train dan validasi, data train sendiri berfungsi untuk melatih model, data validasi berfungsi untuk memvalidasi model diluar data train, data validasi ini berfungsi untuk memberitahu model bahwa model yang sedang dibuat masih belum cukup baik dalam memprediksi data baru. Data validasi ini biasa digunakan dalam <em> callback </em> untuk mempercepat waktu <em> training </em> karena kita akan memonitor <em> val_loss </em>

## Modeling

* Metode Content Based Filtering

  Content based filtering menggunakan informasi tentang beberapa item/data untuk merekomendasikan kepada pengguna sebagai referensi mengenai informasi yang digunakan sebelumnya. Tujuan dari content based filtering adalah untuk memprediksi persamaan sejumlah informasi yang didapat dari pengguna. Sebagai contoh, seorang pendengar musik sedang mendengar musik bergenre reggae. Platform musik online secara sistem akan merekomendasikan si pengguna untuk mendengarkan musik lain yang berhubungan dengan reggae. Dalam pembuatannya, content based filtering menggunakan konsep perhitungan Cosine Similarity yang intinya mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama.

  Cosine similarity mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity.

  Berikut rumus dari Cosine Similarity

  cosine similarity = $\sum_{i=1}^n * A_i B_i \over \sum_{i=1}^n * A_i^2 * \sum_{i=1}^n * B_i^2$

          Dik :
      
          A : Product Item 1
          B : Product Item 2

- Data yang digunakan pada metode ini adalah data yang disukai oleh pengguna pada masa lalu. Rekomendasi yang dihasilkan merupakan rekomendasi yang berdasarkan data pengguna tersebut di masa lalu.

  - Hasil top N Recommendation terhadap Publisher 370 <code>recommend_books_to_you(df, '370', '40')</code> dan hasil Top N Recommendation nya

    Tabel 3. <em>Top N Recommendation</em>

    |   NO | ISBN_le | Similar with Publisher id 370 |                                Book-Author_le |  Age |                                     Book-Title_le |
    | ---: | ------: | ----------------------------: | --------------------------------------------: | ---: | ------------------------------------------------: |
    |    1 |   18248 |                           370 |                        ART OF MANAGING PEOPLE | 40.0 |                                    Jazz Anecdotes |
    |    2 |    7907 |                           231 | A Man and His Mother: An Adopted Son's Search | 46.0 |             Meditations for Women Who Do Too Much |
    |    3 |   76592 |                            50 |                              Franny and Zooey | 33.0 | Spider Woman's Granddaughters: Traditional Tal... |
    |    4 |    2782 |                           231 |                    A wild old man on the road | 40.0 |                               The Late Night Muse |
    |    5 |   28945 |                           373 |                            Grab Hands and Run | 46.0 |                       De Fun Dont Done Les Norton |
    |    6 |  156920 |                           280 |          Escape Me Never (Harlequin Presents) | 54.0 |                The Episode of the Wandering Knife |

* Metode Colaborative Filtering

  Metode Colaborative filtering merupakan metode yang melakukan proses penyaringan item yang berdasarkan pengguna lain, dengan cara memberikan informasi kepada pengguna berdasarkan kemiripan karakteristik. Dalam pembuatanya saya menggunakan RecommenderNet, pada tahap ini model menghitung skor kecocokan antara pengguna dan buku dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan buku. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan buku. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan buku. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Metode ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. [3]

  - Data yang digunakan pada metode ini adalah data yang berupa nilai, biasanya rating. Disini saya meenggunakan kolom _Book-Rating_

  - Top N Recommendation yang dihasilkan sebagai berikut.

    ```
    Showing recommendations for users: 252848.0
    ===========================
    book with high ratings from user
    --------------------------------
    971246                     The Rolling Stone Book of Comedy
    101744                                    WLD ACCORDNG GARP
    620005    Writer's Market 2000: 8,000 Editors Who Buy Wh...
    303744                                          Beach Music
    551332     The Grass Is Always Greener Over the Septic Tank
    Name: Book-Title, dtype: object
    --------------------------------
    Top 10 book recommendation
    --------------------------------
    Wizard and Glass (The Dark Tower, Book 4)
    Wizard and Glass (The Dark Tower, Book 4)
    Slow Hand: Women Writing Erotica
    The Seven Dials Mystery (St. Martin's Minotaur Mysteries)
    Wizard and Glass (The Dark Tower, Book 4)
    Wizard and Glass (The Dark Tower, Book 4)
    The Devil in Bellminster: An Unlikely Mystery (Unlikely Heroes)
    Vintage Affair (Harlequin Romance, No 3158)
    The Seven Dials Mystery (St. Martin's Minotaur Mysteries)
    Wizard and Glass (The Dark Tower, Book 4)
    ```



## Evaluation

Untuk Content Base Filtering saya saya akan menghitung precision nya dengan rumus berikut.

recommender system precision = p $ \text {of recommendations that are relevants} \over \text{of items we recommended} $

Untuk cara menghitung nya disini saya meminta rekomendasi Buku untuk Publisher 370

Bisa dilihat di tabel 3 ada 1 dari 6 rekomendasi diberikan yang sesuai artinya kita hitung precision nya dengan cara

p = $1 \over 6$

p = 16,6%

Selanjutnya adalah untuk Model Colaborative Filtering karena model yang digunakan adalah model regressi, maka saya akan menggunakan metric untuk evaluasi, berikut adalah metric nya:

### Root Mean Squared Error (RMSE)

<em>Root Mean Squared Error (RMSE)</em> merupakan salah satu cara untuk mengevaluasi model regresi dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Kelebihan dari RMSE yaitu memiliki tingkat sensitivitas yang cukup tinggi. Sedangkan kekurangannya RMSE tidak menggambarkan kesalahan rata-rata saja namun memiliki implikasi lain yang lebih sulit untuk diurai dan dipahami.

rmse = $\sqrt{\sum\nolimits_{n=1}^n \left((y_i - ŷ_i) ^ 2 \over n \right) }$

Diketahui:

- n = Jumlah Data
- yi = Actual Value / Nilai Sebenarnya
- ŷi = Predicted Value / Nilai Prediksi

Hasil Graph Evaluasi

![Loss Train and Test Model Metrics ](https://github.com/dewisnu/laporan-dicoding/assets/63925882/91021a41-b661-4e5c-9f89-e36dd5f5a571)

Gambar 7. <em>Loss Train and Test Model Metrics </em>

Dapat disimpulkan model ini sedikit overfit, seperti yang kita lihat jika dengan data <em> train rmse </em> , tetapi jika dengan data validasi sebaliknya. Ini disebabkan karena sedikit nya improvisasi <em> datasets </em>, dikarenakan keterbatasan komputasi saya hanya bisa memasukan max 100.000 <em>row</em> dalam <em> datasets </em> saja, jika lebih maka akan <em>out of memory</em>.

## Kesimpulan

Setelah melalui beberapa tahap diputuskan bahwa kedua model yang penulis gunakan dapat memprediksi  sesuai dengan apa yang diharapkan meskipun dengan data yang dikikis sebagian. Keterbatasan komputasi merupakan tantangan utama penulis dalam mengerjakan sistem rekomendasi ini, penulis terpaksa memangkas beberapa data supaya dapat dikerjakan oleh sistem.

###  Daftar Refrensi

[1] O. D. Maharani, “Minat Baca Anak-Anak di Kampoeng Baca Kabupaten jember,” *Jurnal Review Pendidikan Dasar : Jurnal Kajian Pendidikan dan Hasil Penelitian*, vol. 3, no. 1, p. 320, 2017. doi:10.26740/jrpd.v3n1.p320-328 

[2] Q. Li and B. M. Kim, “An approach for combining content-based and collaborative filters,” *Proceedings of the sixth international workshop on Information retrieval with Asian languages -*, 2003. doi:10.3115/1118935.1118938 

[3] Dicoding. "Kelas Machine Learning Terapan." https://www.dicoding.com/academies/319 [accessed Jul. 18 2022]

