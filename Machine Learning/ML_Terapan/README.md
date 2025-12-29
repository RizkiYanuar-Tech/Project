# Laporan Proyek Machine Learning - Muhammad Rizki Yanuar

## Domain Proyek

Krisis saat ini telah menyebabkan banyak lembaga keuangan di seluruh dunia mengambil langkah-langkah penting untuk menghindari risiko gagal bayar dari pelanggan yang meminjam dana. Maraknya kasus default utang konsumen mendorong para ahli untuk meninjau kembali standar dan praktik yang selama ini digunakan, guna memastikan perlindungan yang memadai bagi perusahaan dari peristiwa serupa di masa depan [1].

Bagi setiap bank atau lembaga keuangan, pengelolaan pinjaman dan pengendalian leverage merupakan tugas krusial dalam menjaga stabilitas dan efisiensi operasional. Tanpa model bisnis pinjaman-ke-simpanan yang dirancang dengan baik, bank tidak dapat menjalankan fungsinya secara optimal. Seiring kemajuan teknologi, mekanisme pemberian dan penanganan pinjaman mengalami transformasi signifikan, salah satunya melalui penerapan pembelajaran mesin (machine learning) dan ilmu data (data science)[2].

## Referensi
[1]	K. Gupta, B. Chakrabarti, A. Ansari, S. S. Rautray, and M. Pandey, “Loanification-Loan Approval Classification using Machine Learning Algorithms.” [Online]. Available: https://ssrn.com/abstract=3833303

[2]	A. Mahgoub, “Optimizing Bank Loan Approval with Binary Classification Method and Deep Learning Model,” Open Journal of Business and Management, vol. 12, no. 03, pp. 1970–2001, 2024, doi: 10.4236/ojbm.2024.123104.

## Business Understanding

Dalam industri perbankan dan keuangan, salah satu tantangan terbesar adalah mengidentifikasi nasabah yang berpotensi gagal bayar (default) sejak tahap awal proses aplikasi pinjaman. Keputusan yang tidak tepat dalam pemberian kredit dapat berdampak buruk terhadap stabilitas keuangan lembaga. Oleh karena itu, diperlukan sistem prediksi yang mampu mengevaluasi risiko kredit calon peminjam secara akurat dan efisien.

Proyek ini bertujuan untuk mengembangkan model klasifikasi berbasis machine learning yang dapat memprediksi kemungkinan terjadinya gagal bayar pinjaman, dengan memanfaatkan data historis pelanggan dan berbagai atribut finansial lainnya. Model ini diharapkan dapat membantu lembaga keuangan dalam membuat keputusan pemberian pinjaman yang lebih tepat dan berbasis data.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Banyak individu yang menerima pinjaman keuangan padahal individu tersebut tidak layak untuk diberikan pinjaman sehingga menyebabkan  mengalami kerugian

- Belum adanya sistem prediksi otomatis berbasis pembelajaran mesin yang secara akurat dapat mengklasifikasikan seseorang layak mendapatkan pinjaman uang atau tidak.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Identifikasi fitur yang paling penting yang mempengaruhi apakah orang tersebut layak untuk mendapatkan pinjaman atau tidak

- Mengembangkan model klasifikasi berbasis machine learning yang mampu memprediksi apakah seseorang kemungkinan akan menerima pinjaman atau tidak

### Solution statements
- Implementasi dan membandingkan model klasifikasi untuk menemukan model terbaik
- Melakukan hyperparameter tuning pada model untuk mengoptimalkan hasil klasifikasi dan meningkatkan akurasi prediksi

## Data Understanding
Dataset yang saya gunakan berasal dari kaggle: [Loan Approval Classification](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data). Dataset ini berisikan 45000 data dengan 14 atribut, dimana pada data tidak ditemukan adanya *missing value* dan tidak ada data yang terduplikat, berikut atribut yang dimiliki:

### Variabel-variabel pada Loan Approval Classification dataset adalah sebagai berikut:
- person_age: Umur orang
- person_gender: Jenis kelamin orang
- person_education: Pendidikan terakhir
- person_income: Pendapatan tahunan
- person_emp_exp: Pengalaman bekerja -tahun
- person_home_ownership: Status kepemilikan rumah
- loan_amnt: Total pinjaman yang diminta
- loan_intent: Tujuan pinjaman
- loan_int_rate: Suku bunga pinjaman
- loan_percent_income: Jumlah pinjaman sebagai persentase pendapatan tahunan
- cb_person_cred_hist_length: Lama riwayat kredit dalam tahun
- credit_score: Nilai kredit orang
- previous_loan_defaults_on_file: Indikator tunggakan pinjaman sebelumnya
- loan_status: Status persetujuan pinjaman

## Data Preparation
- Encoding: Transformasi data dengan mengubah nilai kategori ke bentuk numerik.
- Outliers: Dilakukan pengechekan agar tidak mengganggu kinerja model
- Split Data: Membagi data latih dan data uji dengan proporsi 80:20

## Modeling
Pada tahap ini, beberapa algoritma klasifikasi machine learning digunakan untuk memprediksi apakah seseorang mengalami penyakit liver atau tidak.

  1. Random Forest
     kumpulan dari banyak pohon keputusan (decision trees) yang dilatih dengan data acak dan subset fitur yang berbeda. Hasil akhir ditentukan berdasarkan voting mayoritas dari semua pohon.

  Kelebihan
  
  a. Akurasi yang lebih tinggi daripada decision tree tunggal
  
  b. Lebih tahan terhadap overfitting
  
  c. Memberikan feature importance
  
  Kekurangan 
  
  a. Kurang interpretatif dibanding decision tree tunggal
  
  b. Sedikit lebih lama dalam pelatihan

  2. Gradient Boosting
     Gradient Boosting adalah sebuah teknik yang menggabungkan beberapa model yang lemah (weak model) menjadi sebuah model yang kuat.

  Kelebihan

  a. Akurasi yang tinggi: Gradient Boosting sering menghasilkan model yang akurat dan kuat, terutama ketika digunakan pada data yang kompleks dan tidak terstruktur.

  b. Kecepatan komputasi yang cepat

  Kekurangan:
  
  a. Memerlukan tuning yang cermat: Algoritma ini memerlukan tuning parameter yang cermat untuk mendapatkan model yang optimal.
  
  b. Mudah overfitting: Gradient Boosting dapat cenderung overfit pada data training jika tidak dilakukan pengaturan parameter yang baik.

## Hyperparameter Tuning

Hyperparameter tuning dilakukan pada model untuk mendapatkan parameter terbaik yang mampu membantu model untuk meningkatkan akurasi, metode yang digunakan adalah RandomSearchCV dimana metode ini akan melakukan pengambilan parameter secara acak. Parameter yang akan digunakan sebagai berikut:

Random Forest:
- 'n_estimators': [100, 200] -> Jumlah decision tree yang dibuat dalam random forest, semakin banyak decision tree dapat meningkatkan kinerja model tetapi meningkatkan biaya pelatihan dan prediksi komputasi.

- 'max_depth': [20, 30] -> Kedalaman maksimum decision tree dalam random forest, semakin tinggi nilai max_depth maka semakin kompleks decision tree dan semakin banyak kemungkinan split yang dilakukan

- 'criterion': ['gini', 'entropy', 'log_loss'] -> Mengukur kualitas pemisahan pada setiap cabang pohon

Gradient Boosting
- 'loss': ['log_loss', 'exponential'] -> Bagaimana model memperkirakan dan memperbaiki kesalahan prediksi

- 'learning_rate': [0.01, 0.1] -> Seberapa besar langkah yang diambil untuk memperbarui bobot model selama pelatihan, menentukan seberapa cepat atau lambat model belajar dari data pelatihan. Learning_rate ini merupakan ukuran dari perubahan bobot pada setiap iterasi berdasarkan gradien dari loss function

- 'n_estimators': [100, 200] -> Jumlah decision tree yang dibuat dalam random forest, semakin banyak decision tree dapat meningkatkan kinerja model tetapi meningkatkan biaya pelatihan dan prediksi komputasi.

- 'max_depth': [20, 30] -> Kedalaman maksimum decision tree dalam random forest, semakin tinggi nilai max_depth maka semakin kompleks decision tree dan semakin banyak kemungkinan split yang dilakukan
  
## Evaluation

- Akurasi: Mengukur proporsi prediksi yang benar dari keseluruhan prediksi.

- Recall (Sensitivity): Mengukur seberapa baik model dapat mendeteksi kasus positif.

- Precision: Mengukur proporsi prediksi positif yang benar-benar positif.

- F1-Score: Harmonik Rata-rata dari precision dan recall, yang cocok untuk label tidak seimbang.

Hasil akurasi pada model Random Forest sebelum hyperparameter tuning:
![Base Model Random Forest](https://github.com/user-attachments/assets/2d5ab4f3-b76d-4ef2-bb7e-8d0c088d4a4e)

Setelah hyperparameter tuning:
![Hyparameter](https://github.com/user-attachments/assets/47b81f6c-c081-43ad-aedd-1a245ab05837)

![Hyperparameter Random Forest](https://github.com/user-attachments/assets/42662731-e3e5-4f30-b4d5-60322398ec33)

Ditemukan bahwa parameter terbaik adalah 'n_estimators': 200, 'max_depth': 20, 'criterion': 'log_loss'. Tetapi jika dibandingkan dengan base model random forest, akurasi yang diberikan tidak mengalami perbedaan yang signifikan dimana base model mendapatkan akurasi 0.9257% pada testing, sedangkan ketika menggunakan hyperparameter hanya mendapatkan akurasi 0.9268%.

![Confusion Matrix](https://github.com/user-attachments/assets/5af259e5-050e-4139-b63d-d48ea05751a7)

## Kesimpulan
Berdasarkan hasil klasifikasi menggunakan random forest mendapatkan akurasi 0.926%, dimana menunjukkan bahwa model mampu menghasilkan klasifikasi dengan baik antara orang yang layak untuk diberikan pinjaman dan tidak layak untuk diberikan pinjaman. Ditemukan beberapa hubungan antar fitur yang mempengaruhi diterima atau ditolaknya pengajuan pinjaman
