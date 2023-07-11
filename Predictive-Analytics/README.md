# Laporan Proyek _Machine Learning_ - I Gede Ari Wisnu Sanjaya
---
## Domain Proyek
ingkat pengangguran adalah indikator kunci dalam analisis ekonomi dan penilaian kesehatan pasar tenaga kerja. Tingkat pengangguran yang tinggi dapat mengindikasikan ketidakseimbangan ekonomi dan dampak sosial yang serius, sedangkan tingkat pengangguran yang rendah mencerminkan pertumbuhan ekonomi yang sehat dan peluang kerja yang lebih baik.

Maka dari permasalahan tersebut penulis ingin membuat sebuah sistem prediksi standart jumlah pengganguran di suatu negara. Agar memudahkan  dalam memahami dan mengelola tingkat pengangguran di suatu negara, serta mendukung perencanaan ekonomi yang lebih baik, kebijakan yang efektif, dan pengambilan keputusan yang tepat di berbagai sektor.

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
* bagaimana cara _preprocessing_ pada data US Unemployment 1948-2021 yang akan digunakan untuk membuat model yang baik ?
* Bagaimana cara memilih/membuat model yang terbaik untuk memprediksi standart jumlah penganguran ?

### Goals
Menjelaskan tujuan dari pernyataan masalah latar belakang:
- Melakukan _preprocessing_ data sehingga data tersebut siap untuk di latih oleh model _Machine Learning_
- Menggunakan library python yaitu _Lazy Predict_ yang dapat langsung membandingkan 20 algoritma  _Machine Learning_, selanjutnya adalah menghitung menggunakan beberapa _metric_ seperti _mse, rmse, r2_ yang akan menjadi tolak ukur model terbaik
- Untuk _preprocessing_ data dapat dilakukan beberapa teknik, diantaranya :
    - Melakukan drop kolom pada kolom yang tidak penting / yang tidak berpengaruh pada prediksi jumlah _Unemployment_ .
    - _Handling null value_, bisa dengan _mean, median_. Untuk projek saya ini kebetulan data nya sudah clean.
    - Melakukan Encoding terhadap kolom yang bertipe object / categorical dengan menggunakan OHE Label Encoder (Jika bertipe data Ranking). Kebetulan di projek ini tidak ada
    - Melakukan pembagian dataset menjadi dua bagian dengan rasio 8:2 / 80% untuk train dan 20% untuk test.
    - Melakukan _MinMax Scaler_.
- Untuk Pemilihan model terbaik data dapat dilakukan beberapa teknik, diantaranya :
    - Menggunakan LazyPredict untuk membandingkan 20+ Algoritma _Machine Learning_.
    - Dari 20 Algoritma akan diambil 4 algoritma terbaik untuk dilanjutkan ke tahap evaluasi
    - Menghitung metric yang akan menjadikan patokan kita untuk memilih model terbaik _(mse, rmse, r2)_
    - Berikut adalah Rumus untuk menghitung _MSE_
      
       mse = $1 \over n$ $\sum_{n=0}^n $ $(y_i - ŷ_p) ^ 2 $
    - Berikut adalah rumus untuk menghitung RMSE
    
      rmse = $\sqrt{\sum\nolimits_{n=1}^n \left((y_i - ŷ_p) ^ 2 \over n \right) }$ 
