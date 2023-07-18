# Laporan Proyek Machine Learning - I Gede Ari Wisnu Sanjaya



##  Proyek Overview

Buku merupakan sumber informasi yang memenuhi berbagai kebutuhan, mulai dari ilmu pengetahuan dan teknologi, seni budaya, ekonomi, politik, sosial, hingga pertahanan dan keamanan, dan lain sebagainya. Membaca buku bukan hanya mengembangkan wawasan intelektual, tetapi juga memiliki kekuatan untuk mengubah masa depan dan memperkaya pikiran, pemikiran, dan iman. Dengan membaca buku, pengetahuan kita bertambah, dan pribadi kita menjadi lebih kaya, yang semuanya secara jelas akan mengurangi dampak negatif seperti kenakalan pada anak-anak.

Namun, berdasarkan data statistic UNESCO 2012 minat membaca di indonesia hanya 0.001, yang berarti setiap 1.000 orang hanya 1 orang  pelajar yang memiliki minat membaca. Hal ini sangat disayangkan mengingat Indonesia sebagai negara yang besar memiliki potensi besar untuk menjadi negara unggul. Rendahnya minat baca di kalangan masyarakat menjadi persoalan penting dalam dunia pendidikan. Oleh karena itu, diperlukan sebuah sistem yang dapat membantu merekomendasikan para pembaca agar lebih mudah mendapatkan informasi buku-buku yang akan dibaca selanjutnya[1].

Sistem rekomendasi sendiri telah digunakan secara luas dalam hampir semua area bisnis di mana seorang konsumen memerlukan informasi untuk membuat keputusan. Terdapat dua pendekatan umum yang digunakan dalam pembuatan sistem rekomendasi, yaitu content-based filtering dan collaborative filtering. Content-based filtering adalah metode yang bekerja dengan mencari kesamaan antara item yang akan direkomendasikan dengan item yang telah dilihat oleh pengguna sebelumnya berdasarkan kesamaan konten. Namun, sistem rekomendasi berbasis konten ini masih memiliki kelemahan, yaitu karena semua rekomendasi didasarkan pada konten yang serupa, pengguna tidak mendapatkan rekomendasi untuk konten yang berbeda. Selain itu, sistem rekomendasi ini kurang efektif untuk pengguna pemula, karena mereka tidak mendapatkan masukan dari pengguna sebelumnya[2].

## Business Uderstanding

### Problem Statements

- Bagaimana cara merekomendasikan buku yang disukai pengguna lain dapat diminati dan dijadikan rekomendasi untuk pengguna lainnya?

### Goals

* Membuat Sistem Rekomendasi berdasarkan user yang meberikan rating terhadap suatu buku 

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution Approach

Solusi yang saya ajukan yaitu dengan menggunakan 2 algoritma machine learning untuk sistem rekomendasi yaitu:

- _Content Based Filtering_  adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan tindakan mereka sebelumnya atau umpan balik eksplisit. Algoritma ini memberikan rekomendasi berdasarkan aktivitas pada masa lalu[3].
- _Collaborative Filtering_  adalah algoritma yang bergantung pada pendapat komunitas pengguna. Dia tidak memerlukan atribut untuk setiap itemnya. Algoritma ini memberikan rekomendasi berdasarkan nilai rating atau nilai lain, disini saya menggunakan target sebagai dasar penilaian.

## Data Understanding

Data atau datasets yang digunakan pada proyek <em> Machine Learning </em> ini adalah data _Book Recommendation Dataset_ yang bisa di akses di [link berikut ini (kaggle)](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset?select=Books.csv).





## Daftar Refrensi

[1] O. D. Maharani, “Minat Baca Anak-Anak di Kampoeng Baca Kabupaten jember,” *Jurnal Review Pendidikan Dasar : Jurnal Kajian Pendidikan dan Hasil Penelitian*, vol. 3, no. 1, p. 320, 2017. doi:10.26740/jrpd.v3n1.p320-328 

[2] Q. Li and B. M. Kim, “An approach for combining content-based and collaborative filters,” *Proceedings of the sixth international workshop on Information retrieval with Asian languages -*, 2003. doi:10.3115/1118935.1118938 

[3] Dicoding. "Kelas Machine Learning Terapan." https://www.dicoding.com/academies/319 [accessed Jul. 18 2022]

