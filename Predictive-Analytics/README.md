# Laporan Proyek _Machine Learning_ - I Gede Ari Wisnu Sanjaya
---
## Domain Proyek
Tingkat pengangguran adalah indikator kunci dalam analisis ekonomi dan penilaian kesehatan pasar tenaga kerja. Tingkat pengangguran yang tinggi dapat mengindikasikan ketidakseimbangan ekonomi, dampak sosial yang serius Meningkatkan kemiskinan. Memicu tindakan kriminalitas atau kejahatan. Munculnya ketidaksetaraan politik dan sosial, sedangkan tingkat pengangguran yang rendah mencerminkan pertumbuhan ekonomi yang sehat dan peluang kerja yang lebih baik[1].

Maka dari permasalahan tersebut penulis ingin membuat sebuah sistem prediksi standart jumlah pengganguran di suatu negara. Agar memudahkan  dalam memahami dan mengelola tingkat pengangguran di suatu negara, serta mendukung perencanaan ekonomi yang lebih baik, kebijakan yang efektif, dan pengambilan keputusan yang tepat di berbagai sektor.

## Business Understanding
### Problem Statements
Menjelaskan pernyataan masalah latar belakang:
* bagaimana cara _preprocessing_ pada data US Unemployment 1948-2021 yang akan digunakan untuk membuat model yang baik ?
* Bagaimana cara memilih/membuat model yang terbaik untuk memprediksi standart jumlah penganguran ?

### Goals
Menjelaskan tujuan dari pernyataan masalah latar belakang:

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



![Grafik jumlah tingkat _Unemployment_ dari Tahun ke Tahun](https://github.com/dewisnu/laporan-dicoding/assets/63925882/23581127-ff57-4c87-a538-a9d1c6b1faf1)


Gambar 1. Grafik jumlah tingkat _Unemployment_ dari Tahun ke Tahun

Berikut adalah kolerasi antar fitur yang terdapat pada datasets, bisa disimpulkan semua fitur unrate_men - unrate_55_over sangat berpengaruh terhadap column unrate / column yang akan kita prediksi jadi kita akan menggunakan semua column

![Korelasi antar Kolom](https://github.com/dewisnu/laporan-dicoding/assets/63925882/f2262b2b-3f11-430a-a848-be29404a1924)


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


Untuk Train Test Split kita akan melakukan pembagian dataset menjadi dua bagian dengan rasio 8:2 / 80% untuk train dan 20% untuk test.Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru


## Modeling
Pada tahap ini, akan dikembangkan model _Machine Learning_ dengan melakukan perbandingan 20+ Algoritma Kemudian akan kita pilih 4 Algoritma teratas, selanjutnya dari 4 Algoritma tersebut akan kita evaluasi performa metric nya di tahap Evaluation untuk menentukan model terbaik.Dan berikut hasilnya:

Tabel 3. Hasil Perbandingan model menggunakan _Lazy Predict_
|Model|Adjusted R-Squared|R-Squared|RMSE|Time Taken|
|---|---|---|---|---|
|LassoCV|0\.9981607765360839|0\.9982604934708745|0\.007590360960058252|0\.07328128814697266|
|NuSVR|0\.9981599055572423|0\.9982596697137773|0\.007592157985116283|0\.16698598861694336|
|ElasticNetCV|0\.9981577296002652|0\.9982576117303713|0\.007596645615683601|0\.0480809211730957|
|LassoLarsCV|0\.9981382881722987|0\.9982392243557282|0\.007636624014040912|0\.017354726791381836|
|LassoLarsIC|0\.9981382881722987|0\.9982392243557282|0\.007636624014040912|0\.01003408432006836|
|LinearRegression|0\.9981247257772197|0\.9982263972712259|0\.007664389581266866|0\.006811618804931641|
|TransformedTargetRegressor|0\.9981247257772197|0\.9982263972712259|0\.007664389581266866|0\.007271766662597656|
|RANSACRegressor|0\.9981247257772197|0\.9982263972712259|0\.007664389581266872|0\.01858663558959961|
|BayesianRidge|0\.9981245863250502|0\.9982262653797161|0\.007664674551825534|0\.007128715515136719|
|RidgeCV|0\.9981237991057582|0\.9982255208409881|0\.007666283036051322|0\.007297515869140625|
|OrthogonalMatchingPursuitCV|0\.9980922861783601|0\.9981957164457984|0\.007730396911107291|0\.010756969451904297|
|HuberRegressor|0\.9980855719393483|0\.9981893662317933|0\.007743988609990139|0\.020911455154418945|
|Ridge|0\.998083983538929|0\.998187863949469|0\.007747200536472195|0\.01153564453125|
|LarsCV|0\.9980426360530439|0\.9981487581947464|0\.007830346352535437|0\.03624558448791504|
|LinearSVR|0\.9980193455372339|0\.9981267304177454|0\.007876794919193533|0\.03078913688659668|
|Lars|0\.9978920713986894|0\.9980063566843026|0\.008125930960826878|0\.010445117950439453|
|HistGradientBoostingRegressor|0\.9963356313319192|0\.9965343019223573|0\.010713830232567117|0\.20264387130737305|
|LGBMRegressor|0\.9962718188322794|0\.9964739491365534|0\.010806714648308574|0\.044396400451660156|
|ExtraTreesRegressor|0\.9960755455031486|0\.996288317132496|0\.011087530533591976|0\.15281224250793457|
|GradientBoostingRegressor|0\.9953515113887046|0\.9956035378796785|0\.01206704870182546|0\.17106056213378906|
|RandomForestRegressor|0\.9947734566960141|0\.9950568235016519|0\.012795358439871517|0\.30699849128723145|
|XGBRegressor|0\.9947248215218909|0\.9950108251743185|0\.012854753669704368|0\.05054354667663574|
|BaggingRegressor|0\.9934003657475632|0\.9937581772431773|0\.014378219648695538|0\.037648677825927734|
|SGDRegressor|0\.9925654637747198|0\.9929685410399458|0\.015260617982653416|0\.0072705745697021484|
|KNeighborsRegressor|0\.9860925112179856|0\.9868465316941188|0\.02087226302868927|0\.010062217712402344|
|DecisionTreeRegressor|0\.9838577432033603|0\.9847329258007684|0\.022486781805201857|0\.009985208511352539|
|AdaBoostRegressor|0\.9756053322349153|0\.9769279347041067|0\.02764347483635253|0\.10649943351745605|
|ExtraTreeRegressor|0\.9625159805332764|0\.9645482466489421|0\.034266382409097705|0\.011297941207885742|
|PassiveAggressiveRegressor|0\.960892493433054|0\.9630127799336716|0\.0350005801740497|0\.008588790893554688|
|TweedieRegressor|0\.9589250771138671|0\.9611520307643201|0\.035870180185760124|0\.007681608200073242|
|GaussianProcessRegressor|0\.9466254360823161|0\.9495192377405037|0\.0408895517608501|0\.04036998748779297|
|MLPRegressor|0\.9274392954154942|0\.9313733095194734|0\.047675573445269355|0\.10176396369934082|
|OrthogonalMatchingPursuit|0\.9246398730502075|0\.9287256630655577|0\.04858654200049193|0\.007971525192260742|
|SVR|0\.9117231112354163|0\.9165092076142191|0\.05258582467452991|0\.007441997528076172|
|PoissonRegressor|0\.8710712629899138|0\.8780613752374485|0\.06355065689754948|0\.010561466217041016|
|QuantileRegressor|-0\.06348586143687296|-0\.005826989431259477|0\.18252024377347967|5\.440234422683716|
|DummyRegressor|-0\.07298449276534802|-0\.014810634723853244|0\.18333353081275303|0\.00658416748046875|
|ElasticNet|-0\.07298449276534802|-0\.014810634723853244|0\.18333353081275303|0\.008392333984375|
|LassoLars|-0\.07298449276534802|-0\.014810634723853244|0\.18333353081275303|0\.009058237075805664|
|Lasso|-0\.07298449276534802|-0\.014810634723853244|0\.18333353081275303|0\.02502298355102539|
|KernelRidge|-4\.656095432830404|-4\.349439656351647|0\.42092385978463703|0\.016180038452148438|



Bisa disimpulkan 4 model terbaik yang akan digunakan untuk evaluasi berikut adalah modelnya

Tabel 4. Model Terbaik hasil perbandingan menggunakan _lazy Predict_
|Model|Adjusted R-Squared|R-Squared|RMSE|Time Taken|
|---|---|---|---|---|
|LassoCV|0\.9981607765360839|0\.9982604934708745|0\.007590360960058252|0\.07328128814697266|
|NuSVR|0\.9981599055572423|0\.9982596697137773|0\.007592157985116283|0\.16698598861694336|
|ElasticNetCV|0\.9981577296002652|0\.9982576117303713|0\.007596645615683601|0\.0480809211730957|
|LassoLarsCV|0\.9981382881722987|0\.9982392243557282|0\.007636624014040912|0\.017354726791381836|



### Model yang digunakan

#### Models
- _Lasso CV_ =  Model linier Lasso dengan penyesuaian iteratif sepanjang jalur _regularization_[4]. Regresi Lasso adalah teknik _regularization_. Digunakan untuk metode regresi guna prediksi yang lebih akurat. Model ini menggunakan penyusutan (shrinkage). Penyusutan adalah saat nilai-nilai data disusutkan menuju titik tengah seperti mean. Prosedur Lasso mendorong penggunaan model sederhana dan jarang (yaitu model dengan parameter lebih sedikit). Jenis regresi khusus ini sangat cocok untuk model-model yang menunjukkan tingkat multikolinearitas yang tinggi atau ketika Anda ingin mengotomatisasi bagian-bagian tertentu dari pemilihan model, seperti pemilihan variabel/eliminasi parameter[2].

![_Lasso_ _Algorithm](https://github.com/dewisnu/laporan-dicoding/assets/63925882/09d46b8f-e5ef-472a-b09c-b9d6b04fe7c0)


- _NuSVR_ = Nu Support Vector Regression adalah metode regresi yang mirip dengan NuSVC. Untuk regresi, Nu Support Vector Regression menggunakan parameter nu untuk mengontrol jumlah vektor pendukung. Namun, berbeda dengan NuSVC, di mana nu menggantikan nilai C, dalam Nu Support Vector Regression nu menggantikan parameter epsilon dari epsilon-SVR.
Implementasinya didasarkan pada libsvm.[4].

- _ElasticNetCV_ = Regresi linear mengacu pada model yang mengasumsikan hubungan linear antara variabel input dan variabel target. Dengan satu variabel input, hubungan ini berupa garis lurus, dan dengan dimensi yang lebih tinggi, hubungan ini dapat dianggap sebagai hiperbidang yang menghubungkan variabel input dengan variabel target. Koefisien model ditemukan melalui proses optimisasi yang bertujuan untuk meminimalkan kesalahan kuadrat antara prediksi (yhat) dan nilai target yang diharapkan (y).[3].
  
  loss = sum i=0 to n (y_i – yhat_i)^2

- _LassoLarsCV_  = Model ini menggunakan   penggunaan teknik validasi silang (cross-validation) dan algoritma LARS (Least Angle Regression) untuk melakukan seleksi variabel secara otomatis. Dengan menggunakan metode ini, kita dapat memperoleh model yang lebih optimal dengan mengidentifikasi variabel yang memiliki dampak signifikan terhadap hasil prediksi.[4]. 

## Evaluation
Model yang digunakan adalah model regressi, sesuai penjelasan diatas saya akan menggunakan beberapa metric untuk evaluasi, berikut adalah list nya:
- _Mean Squared Error_
- _Root Mean Squared Error (RMSE)_
- _R2_

### MSE (Mean Squared Error)
_Mean Squared Error (MSE)_ adalah Rata-rata Kesalahan kuadrat diantara nilai aktual dan nilai prediksi. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada prediksi. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil prediksi sesuai dengan data aktual dan bisa dijadikan untuk perhitungan prediksi di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi.

Kelebihan MSE yaitu sederhana dalam perhitungan. Sedangkan kelemahan yang dimiliki MSE adalah akurasi hasil prediksi sangat kecil karena tidak memperhatikan apakah hasil prediksi lebih besar atau lebih kecil dibandingkan kenyataannya

mse = $1 \over n$ $\sum_{n=0}^n $ $(y_i - ŷ_i) ^ 2 $ 
Diketahui:
- n = Jumlah Data
- yi = Actual Value / Nilai Sebenarnya
- ŷi = Predicted Value / Nilai Prediksi


### Root Mean Squared Error (RMSE)
_Root Mean Squared Error (RMSE)_ merupakan salah satu cara untuk mengevaluasi model regresi dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.

Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.

Kelebihan dari RMSE yaitu memiliki tingkat sensitivitas yang cukup tinggi. Sedangkan kekurangannya RMSE tidak menggambarkan kesalahan rata-rata saja namun memiliki implikasi lain yang lebih sulit untuk diurai dan dipahami.

rmse = $\sqrt{\sum\nolimits_{n=1}^n \left((y_i - ŷ_i) ^ 2 \over n \right) }$  

Diketahui:
- n = Jumlah Data
- yi = Actual Value / Nilai Sebenarnya
- ŷi = Predicted Value / Nilai Prediksi

### R2 Score
_R squared_ merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen. Semakin mendekati angka satu, model yang dikeluarkan oleh regresi tersebut akan semakin baik.

Jika kita perhatikan rumus R squared dibawah sangat dipengaruhi oleh nilai Y prediksi atau nilai Y dari hasil rumus dengan nilai Y aktual. Kenyataan yang sering muncul adalah nilai R squared akan semakin membaik (nilainya akan terus mendekati nilai 1) jika kita menambah variabel. Semakin banyak jumlah variabel yang menentukan nilai Y prediksi, maka nilai SSR akan semakin besar yang berakibat pada besarnya nilai R squared.

Sifat R-squared yang akan semakin baik jika menambah variabel inilah yang menjadi kelemahan dari R squared itu sendiri. Semakin banyak variabel independen yang digunakan maka akan semakin banyak “noise” dalam model tersebut dan ini tidak dapat dijelaskan oleh R squared.

 $r^2$ = 1 - $SS_R \over SS_T$ =  1 - $ \sum_{i} (y_i - ŷ_i) ^ 2 \over \sum_{i} (y_i - ȳ) ^ 2$
 
Diketahui:
- SSRes : Kuadrat dari selisih nilai Y prediksi dengan nilai rata-rata Y = ∑ (Ypred – Yrata-rata)²
- SSTotal : Kuadrat dari selisih nilai Y aktual dengan nilai rata-rata Y = ∑ (Yaktual – Yrata-rata)²

### Final Report


Tabel 4. _prediction Result of Model_
|index|y\_true|prediksi\_LassoCV|prediksi\_NuSVR|prediksi\_ElasticNetCV|prediksi\_LassoLarsCV|
|---|---|---|---|---|---|
|434|0\.7599999999999998|0\.8|0\.8|0\.8|0\.8|
|563|0\.35999999999999993|0\.4|0\.4|0\.4|0\.4|
|549|0\.5333333333333334|0\.5|0\.5|0\.5|0\.5|
|671|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|330|0\.8399999999999999|0\.8|0\.8|0\.8|0\.8|
|670|0\.4266666666666666|0\.4|0\.4|0\.4|0\.4|
|692|0\.32|0\.3|0\.3|0\.3|0\.3|
|885|0\.25333333333333324|0\.2|0\.3|0\.2|0\.2|
|660|0\.5466666666666666|0\.6|0\.5|0\.6|0\.6|
|452|0\.6000000000000001|0\.6|0\.6|0\.6|0\.6|
|21|0\.49333333333333323|0\.5|0\.5|0\.5|0\.5|
|611|0\.21333333333333332|0\.2|0\.2|0\.2|0\.2|
|658|0\.4266666666666666|0\.4|0\.4|0\.4|0\.4|
|69|0\.013333333333333308|0\.0|0\.0|0\.0|0\.0|
|197|0\.46666666666666673|0\.5|0\.5|0\.5|0\.5|
|430|0\.7599999999999998|0\.8|0\.8|0\.8|0\.8|
|307|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|160|0\.5599999999999998|0\.6|0\.6|0\.6|0\.6|
|95|0\.2|0\.2|0\.2|0\.2|0\.2|
|790|0\.5599999999999998|0\.6|0\.6|0\.6|0\.6|
|269|0\.4266666666666666|0\.4|0\.4|0\.4|0\.4|
|312|0\.44|0\.4|0\.4|0\.4|0\.4|
|441|0\.6133333333333333|0\.6|0\.6|0\.6|0\.6|
|837|0\.2|0\.2|0\.2|0\.2|0\.2|
|76|0\.44|0\.5|0\.4|0\.5|0\.4|
|485|0\.4133333333333333|0\.4|0\.4|0\.4|0\.4|
|882|0\.44|0\.4|0\.4|0\.4|0\.4|
|96|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|805|0\.4533333333333333|0\.5|0\.5|0\.5|0\.5|
|397|0\.7466666666666666|0\.8|0\.8|0\.8|0\.8|
|623|0\.17333333333333334|0\.2|0\.2|0\.2|0\.2|
|340|0\.5866666666666667|0\.6|0\.6|0\.6|0\.6|
|189|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|264|0\.24000000000000005|0\.2|0\.2|0\.2|0\.2|
|806|0\.4266666666666666|0\.4|0\.4|0\.4|0\.4|
|714|0\.3333333333333333|0\.3|0\.3|0\.3|0\.3|
|857|0\.1866666666666666|0\.2|0\.2|0\.2|0\.2|
|103|0\.17333333333333334|0\.2|0\.2|0\.2|0\.2|
|119|0\.3466666666666666|0\.3|0\.3|0\.3|0\.3|
|665|0\.5466666666666666|0\.6|0\.5|0\.6|0\.6|
|690|0\.37333333333333335|0\.4|0\.4|0\.4|0\.4|
|145|0\.44|0\.4|0\.4|0\.4|0\.4|
|277|0\.5599999999999998|0\.6|0\.6|0\.6|0\.6|
|32|0\.21333333333333332|0\.2|0\.2|0\.2|0\.2|
|112|0\.2|0\.2|0\.2|0\.2|0\.2|
|258|0\.1866666666666666|0\.2|0\.2|0\.2|0\.2|
|661|0\.5333333333333334|0\.5|0\.5|0\.5|0\.5|
|70|0\.10666666666666669|0\.1|0\.1|0\.1|0\.1|
|567|0\.4266666666666666|0\.4|0\.4|0\.4|0\.4|
|81|0\.2933333333333333|0\.3|0\.3|0\.3|0\.3|
|494|0\.37333333333333335|0\.4|0\.4|0\.4|0\.4|
|192|0\.5333333333333334|0\.5|0\.5|0\.5|0\.5|
|570|0\.46666666666666673|0\.5|0\.5|0\.5|0\.5|
|693|0\.2933333333333333|0\.3|0\.3|0\.3|0\.3|
|105|0\.09333333333333332|0\.1|0\.1|0\.1|0\.1|
|12|0\.3466666666666666|0\.3|0\.3|0\.3|0\.3|
|45|0\.05333333333333329|0\.1|0\.1|0\.1|0\.1|
|819|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|199|0\.32|0\.3|0\.3|0\.3|0\.3|
|323|0\.5733333333333333|0\.6|0\.6|0\.6|0\.6|
|774|0\.8266666666666664|0\.8|0\.8|0\.8|0\.8|
|414|0\.9866666666666666|1\.0|1\.0|1\.0|1\.0|
|828|0\.35999999999999993|0\.4|0\.4|0\.4|0\.4|
|621|0\.1866666666666666|0\.2|0\.2|0\.2|0\.2|
|545|0\.6399999999999999|0\.6|0\.6|0\.6|0\.6|
|236|0\.17333333333333334|0\.2|0\.2|0\.2|0\.2|
|131|0\.48000000000000004|0\.5|0\.5|0\.5|0\.5|
|219|0\.15999999999999998|0\.2|0\.2|0\.2|0\.2|
|294|0\.46666666666666673|0\.5|0\.5|0\.5|0\.5|
|821|0\.35999999999999993|0\.4|0\.4|0\.4|0\.4|
|524|0\.5466666666666666|0\.5|0\.5|0\.5|0\.5|
|641|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|240|0\.21333333333333332|0\.2|0\.2|0\.2|0\.2|
|172|0\.35999999999999993|0\.4|0\.4|0\.4|0\.4|
|636|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|278|0\.52|0\.5|0\.5|0\.5|0\.5|
|675|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|473|0\.52|0\.5|0\.5|0\.5|0\.5|
|697|0\.35999999999999993|0\.4|0\.4|0\.4|0\.4|
|74|0\.5333333333333334|0\.5|0\.5|0\.5|0\.5|
|490|0\.37333333333333335|0\.4|0\.4|0\.4|0\.4|
|207|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|232|0\.10666666666666669|0\.1|0\.1|0\.1|0\.1|
|262|0\.11999999999999994|0\.1|0\.1|0\.1|0\.1|
|686|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|210|0\.27999999999999997|0\.3|0\.3|0\.3|0\.3|
|223|0\.15999999999999998|0\.2|0\.2|0\.2|0\.2|
|579|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|377|0\.48000000000000004|0\.5|0\.5|0\.5|0\.5|
|359|0\.48000000000000004|0\.5|0\.5|0\.5|0\.5|
|22|0\.44|0\.4|0\.4|0\.4|0\.4|
|633|0\.15999999999999998|0\.2|0\.2|0\.2|0\.2|
|8|0\.1333333333333333|0\.1|0\.1|0\.1|0\.1|
|662|0\.5066666666666666|0\.5|0\.5|0\.5|0\.5|
|713|0\.3066666666666667|0\.3|0\.3|0\.3|0\.3|
|552|0\.6533333333333333|0\.7|0\.7|0\.7|0\.7|
|244|0\.06666666666666665|0\.1|0\.1|0\.1|0\.1|
|832|0\.22666666666666663|0\.2|0\.2|0\.2|0\.2|
|553|0\.6266666666666667|0\.6|0\.6|0\.6|0\.6|
|79|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|865|0\.1866666666666666|0\.2|0\.2|0\.2|0\.2|
|196|0\.32|0\.3|0\.3|0\.3|0\.3|
|835|0\.27999999999999997|0\.3|0\.3|0\.3|0\.3|
|719|0\.32|0\.3|0\.3|0\.3|0\.3|
|647|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|154|0\.4266666666666666|0\.4|0\.4|0\.4|0\.4|
|309|0\.24000000000000005|0\.2|0\.2|0\.2|0\.2|
|147|0\.37333333333333335|0\.4|0\.4|0\.4|0\.4|
|246|0\.21333333333333332|0\.2|0\.2|0\.2|0\.2|
|118|0\.2933333333333333|0\.3|0\.3|0\.3|0\.3|
|618|0\.27999999999999997|0\.3|0\.3|0\.3|0\.3|
|276|0\.5599999999999998|0\.6|0\.6|0\.6|0\.6|
|285|0\.4000000000000001|0\.4|0\.4|0\.4|0\.4|
|829|0\.3333333333333333|0\.3|0\.3|0\.3|0\.3|
|365|0\.5066666666666666|0\.5|0\.5|0\.5|0\.5|
|313|0\.4533333333333333|0\.5|0\.5|0\.4|0\.5|
|834|0\.2933333333333333|0\.3|0\.3|0\.3|0\.3|
|879|0\.44|0\.4|0\.5|0\.4|0\.4|
|195|0\.38666666666666666|0\.4|0\.4|0\.4|0\.4|
|52|0\.06666666666666665|0\.1|0\.1|0\.1|0\.1|

Dalam tabel yang diberikan, terdapat beberapa model yang digunakan untuk melakukan prediksi. Berikut adalah penjelasan mengenai masing-masing model:

* LassoCV: Model LassoCV merupakan model regresi linear dengan regularisasi L1 (Lasso) yang dikombinasikan dengan teknik validasi silang (cross-validation). Model ini digunakan untuk melakukan prediksi pada kolom "prediksi_LassoCV". Nilai-nilai yang tercantum dalam kolom tersebut adalah hasil prediksi yang diberikan oleh model LassoCV.

* NuSVR: Model NuSVR merupakan model Support Vector Regression (SVR) dengan parameter kernel Nu yang digunakan untuk melakukan prediksi. Model ini digunakan untuk prediksi pada kolom "prediksi_NuSVR". Nilai-nilai dalam kolom tersebut adalah hasil prediksi yang diberikan oleh model NuSVR.

* ElasticNetCV: Model ElasticNetCV adalah model regresi linear yang menggabungkan regularisasi L1 (Lasso) dan regularisasi L2 (Ridge) dengan teknik validasi silang. Model ini digunakan untuk prediksi pada kolom "prediksi_ElasticNetCV". Nilai-nilai dalam kolom tersebut adalah hasil prediksi yang diberikan oleh model ElasticNetCV.

* LassoLarsCV: Model LassoLarsCV merupakan model regresi Lasso dengan teknik validasi silang menggunakan metode Least Angle Regression (LARS). Model ini digunakan untuk prediksi pada kolom "prediksi_LassoLarsCV". Nilai-nilai dalam kolom tersebut adalah hasil prediksi yang diberikan oleh model LassoLarsCV.

Dengan menggunakan model-model di atas, tabel menyajikan hasil prediksi dari masing-masing model untuk setiap sampel atau contoh yang terdaftar dalam tabel. Dengan melihat nilai-nilai prediksi dari setiap model, Anda dapat membandingkan kinerja dan keakuratan model-model tersebut dalam melakukan prediksi terhadap nilai aktual yang tercantum dalam kolom "y_true".

kolom "y_true" mengacu pada nilai aktual atau nilai sebenarnya dari variabel yang sedang diprediksi. Dalam konteks ini, "y_true" mewakili nilai target yang sebenarnya atau nilai yang ingin diprediksi oleh model-model tersebut.

Di mana kita menggunakan model - model tersebut untuk memprediksi standart jumlah tingkat pengangguran maka "y_true" akan berisi jumlah tingkat pengangguran sebenarnya dari dalam dataset. Nilai-nilai yang tercantum dalam kolom "y_true" merupakan referensi atau titik pembanding untuk mengevaluasi sejauh mana prediksi dari setiap model mendekati nilai sebenarnya.

Dengan membandingkan nilai prediksi dari masing-masing model dengan nilai "y_true", Kita dapat mengukur kinerja dan akurasi model dalam melakukan prediksi. Semakin mendekati nilai "y_true", semakin akurat prediksi model.

Setelah melalui berbagai tahapan evaluasi diputuskan bahwa model terbaik yang akan digunakan adalah LassoCV sesuai dengan perhitungan matrix yang telah dijabarkan diatas. Berikut hasil akhir dari 4 Model terbaik.

Tabel 5. _Final Result of Model_
|index|Model\_Name|mse|r2|rmse|
|---|---|---|---|---|
|0|LassoCV|5\.7576947625830465e-05|0\.9982615994842791|0\.007587947523924402|
|1|NuSVR|5\.584341223514932e-05|0\.9983139395082199|0\.007472844989369799|
|2|ElasticNetCV|5\.83802975620606e-05|0\.9982373442223897|0\.0076407000701546055|
|3|LassoLarsCV|5\.832448681004554e-05|0\.998239029296783|0\.007637046995406375|

Setelah mencoba prediksi untuk data test, akurasi yang dihasilkan menggunakan Model LassoCV sudah sesuai Ekspektasi, Dalam konteks ini, "sesuai ekspektasi" berarti bahwa hasil prediksi yang dihasilkan menggunakan Model LassoCV cukup akurat dan mendekati nilai aktual. Meskipun masih ada perbedaan antara prediksi dan data aktual, perbedaan tersebut mungkin masih dalam kisaran yang dapat diterima atau dianggap sebagai tingkat toleransi yang dapat diterima.

Dalam kasus ini, langkah-langkah yang diambil adalah menggunakan Model LassoCV untuk memprediksi tingkat pengangguran. Meskipun hasil prediksi tidak sepenuhnya persis dengan data aktual, tetapi prediksi tersebut masih dapat menjadi patokan yang berguna untuk menentukan tingkat standar pengangguran yang akan diberlakukan. Dengan menggunakan model ini, diharapkan bahwa langkah-langkah yang diambil untuk mencegah peningkatan tingkat pengangguran dapat lebih efektif..

## Daftar Referensi 

### Referensi

[1]C. N. Rianda, “Analisis Dampak Pengangguran Berpengaruh Terhadap individual,” AT-TASYRI’: JURNAL ILMIAH PRODI MUAMALAH, p. 17, 2020. doi:10.47498/tasyri.v12i01.358 . 

[2] Greatlearning. “A Complete understanding of LASSO Regression.” https://www.mygreatlearning.com/ [accessed Jul. 12 2023]

[3] J. Brownlee, “How to develop elastic net regression models in Python,” MachineLearningMastery.com, https://machinelearningmastery.com/elastic-net-regression-in-python/ [accessed Jul. 12, 2023]. 

[4] Boisberranger. J. D, et al., "Scikit Learn Documentations." https://scikit-learn.org/stable/ [accessed Jul. 12 2023]


