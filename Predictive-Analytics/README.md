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
      
    - Berikut adalah rumus untuk menghitung R2
 
      $r^2$ = 1 - $SS_R \over SS_T$ =  1 - $ \sum_{i} (y_i - ŷ_p) ^ 2 \over \sum_{i} (y_i - ȳ) ^ 2$
      
    - Rumus rumus diatas dapat dihitung langsung menggunakan library python yaitu sklearn metrics
      
Setelah goals dicapai, selanjutnya adalah tahap implementasi. Pada tahap ini pemerintah perlu bekerja sama dengan para pelaku usaha, corporasi, perguruan tinggi dan juga lembaga pelatihan agar memudahkan  penyerapan sumber daya manusia yang ada dan juga untuk membantu mengurangi jumlah penggaguran agar tidak melibihi standart jumlah penganguran yang telah ditentukan 

## Data Understanding
Data yang digunakan adalah data yang berasal dari kaggle, data ini berisikan jumlah  US _Unemployment_  dari tahun 1948-2021  berikut adalah datanya (https://www.kaggle.com/datasets/axeltorbenson/unemployment-data-19482021/code)


### Variabel-variabel pada US _Unemployment_ 1948-2021 Datasets adalah sebagai berikut:
- unrate: Total unemployment rate
- unrate_men: Unemployment rate for men
- unrate_women: Unemployment rate for women
- unrate_16_to_17: Unemployment rate for people aged 16 to 17 years old
- unrate_18_to_19: Unemployment rate for people aged 18 to 19 years old
- unrate_20_to_24: Unemployment rate for people aged 20 to 24 years old
- unrate_25_to_34: Unemployment rate for people aged 25 to 34 years old
- unrate_35_to_44: Unemployment rate for people aged 35 to 44 years old
- unrate_45_to_54: Unemployment rate for people aged 45 to 54 years old
- unrate_55_over: Unemployment rate for people aged 55 to over 55 years old


Overview Data:

```
   - Datasets Name :  US Unemployment Data (1948-2021)
    - Overall Columns:
        - Valid : 887
        - MissMatched : 0
        - Missing : 0
    - Source : FED of St. Louis, US Bureau of Labor Statistics
    - Link : 
    - License : U.S. Government Works
    - Inspiration : What make unemployment increase?

```

### Analisis Deskriptif
Tabel 1. _Generative Describe Statistics_
|Parameters|unrate|unrate\_men|unrate\_women|unrate\_16\_to\_17|unrate\_18\_to\_19|unrate\_20\_to\_24|unrate\_25\_to\_34|unrate\_35\_to\_44|unrate\_45\_to\_54|unrate\_55\_over|
|---|---|---|---|---|---|---|---|---|---|---|
|count|887\.0|887\.0|887\.0|887\.0|887\.0|887\.0|887\.0|887\.0|887\.0|887\.0|
|mean|5\.763134160090193|5\.633709131905299|6\.028748590755355|17\.943517474633595|14\.82480270574972|9\.345659526493797|5\.532581736189402|4\.242953776775648|3\.867192784667418|3\.838782412626832|
|std|1\.7401008270503457|1\.9546385465298106|1\.6082521782283488|5\.018894348382397|4\.04786741786198|2\.8009875695474973|1\.9235995427074815|1\.443625929794409|1\.3522472350455321|1\.2415792594790067|
|min|2\.4|1\.9|2\.6|5\.7|5\.2|3\.3|2\.0|1\.6|1\.5|1\.5|
|25%|4\.5|4\.3|4\.9|14\.7|12\.3|7\.7|4\.2|3\.2|2\.95|3\.0|
|50%|5\.5|5\.3|5\.8|17\.8|14\.6|9\.1|5\.2|4\.0|3\.6|3\.6|
|75%|6\.8|6\.7|7\.0|20\.9|17\.0|10\.8|6\.7|5\.0|4\.5|4\.4|
|max|14\.4|13\.3|15\.7|35\.8|33\.3|25\.0|14\.3|11\.3|12\.1|13\.4|

- Hasil Analisis
  - Rata - Rata standart jumlah tingkat _Unemployment_ adalah 5.7 _percent_
  - Jumlah Tingkat  _Unemployment_  terendah adalah 1.5 _percent_ dan tertinggi adalah 35.8 _percent_
    
### Visualization
Berikut adalah jumlah tingkat _Unemployment_ dari waktu ke waktu , tingkat pengganguran meningkat pada tahun 2020 - 2021 sempat menurun pada 2021 akhir dan untuk sekarang tingkat pengganguran mulai meningkat lagi.



![image](https://github.com/dewisnu/laporan-dicoding/assets/63925882/ae92ae52-bd41-4625-b1fb-229e637d8bec)

Gambar 1. Grafik jumlah tingkat _Unemployment_ dari Tahun ke Tahun

Berikut adalah kolerasi antar fitur yang terdapat pada datasets, bisa disimpulkan semua fitur unrate_men - unrate_55_over sangat berpengaruh terhadap column unrate / column yang akan kita prediksi jadi kita akan menggunakan semua column
![image](https://github.com/dewisnu/laporan-dicoding/assets/63925882/c5ec3a3f-1cc0-4b3d-899d-80d4e13fc25f)

Gambar 2.  Korelasi antar Kolom


## Data Preparation

Teknik Data Preparation yang Dilakukan adalah sebagai berikut:

- Karena data tidak memiliki column categorical / object jadi kita skip langkah OHE / Label Encoder ini
- _MinMaxScaller_() : ini merupakan proses scalling yang fungsinya data numeric akan tahan terhadap pencilan data / outliers. _MinMaxScaller_ ini mentransformasi / mengubah data numeric menjadi data numeric yang memiliki rentang 0 - 1
- _TrainTestSplit_() : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru.

Berikut adalah final data setelah digunakan fungsi _MinMaxScaller_() 

Result:
Tabel 2. _Generative Describe Statistics_ setelah di terapkan _MinMaxScaller_
|index|unrate|unrate\_men|unrate\_women|unrate\_16\_to\_17|unrate\_18\_to\_19|unrate\_20\_to\_24|unrate\_25\_to\_34|unrate\_35\_to\_44|unrate\_45\_to\_54|unrate\_55\_over|
|---|---|---|---|---|---|---|---|---|---|---|
|count|834\.0|834\.0|834\.0|834\.0|834\.0|834\.0|834\.0|834\.0|834\.0|834\.0|
|mean|0\.417154276578737|0\.4168761376441017|0\.43296562749800155|0\.4789332468451469|0\.48213621616076124|0\.4681800777865103|0\.3976721062174651|0\.4116164695362354|0\.40683679471517126|0\.42999999999999994|
|std|0\.19573715520207247|0\.19556493389700175|0\.18963166600479284|0\.18327714161848765|0\.1866145945664553|0\.1970961882357763|0\.19428410335392224|0\.1961303756076531|0\.19820918797499196|0\.18907559290382453|
|min|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|0\.0|
|25%|0\.27999999999999997|0\.2771084337349398|0\.29333333333333333|0\.360655737704918|0\.3636363636363636|0\.3529411764705882|0\.25609756097560976|0\.2711864406779661|0\.26415094339622636|0\.2799999999999999|
|50%|0\.4000000000000001|0\.39759036144578325|0\.41333333333333333|0\.4795081967213114|0\.48128342245989314|0\.46218487394957986|0\.3780487804878049|0\.3898305084745762|0\.37735849056603776|0\.4|
|75%|0\.5466666666666666|0\.5421686746987953|0\.5466666666666666|0\.6024590163934426|0\.6029411764705883|0\.5966386554621849|0\.524390243902439|0\.5423728813559321|0\.5283018867924528|0\.54|
|max|1\.0|1\.0|1\.0|1\.0|1\.0|0\.9999999999999999|1\.0|1\.0|1\.0|1\.0|
