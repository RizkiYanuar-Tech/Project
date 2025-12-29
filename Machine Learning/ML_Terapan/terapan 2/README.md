![image](https://github.com/user-attachments/assets/708d3907-7066-46ce-9fee-8d83cbd769d5)# Laporan Proyek Machine Learning Submission 2 - Muhammad Rizki Yanuar

## Domain Proyek
Perpustakaan memiliki ruang rak yang terbatas, tetapi masih memiliki cukup buku yang cukup banyak hingga pemilihan buku menjadi sulit dan memakan waktu, tetapi jumlah buku dan pengguna tidak cukup untuk menghasilkan teknik kolaboratif tradisional yang mengandalkan data dalam jumlah besar[1].

Apalagi Penggunaan internet untuk mencari informasi cenderung meningkat, terutama peningkatann pada pencarian buku di perpustakan atau penggunaan perpustakaan untuk mencari buku. Sehingga dikembangkan dalam sistem perpustakaan untuk meningkatkan efektivitas pencarian informasi untuk memenuhi kepuasaan pengguna, salah satu teknik yang digunakan adalah sistem rekomendasi[2]. 

Sistem rekomendasi buku dirancang dengan mempertimbangkan kecepatan perubahan waktu. Dengan menggunakan catatan peminjaman perpustakaan, dimana sistem bisa menyarankan kepada pustakawan buku apa yang harus dibeli, atapun untuk pengguna dapat membantu untuk menyarankan untuk membeli atau membaca buku yang sesuai untuknya dengan mempertimbangkan berbagai kriteria seperti preferensi, biaya dan fitur lain[3].

## Referensi

[1]	Proceedings of the 12th ACMIEEE-CS joint conference on Digital Libraries. ACM Digital Library, 2013.

[2]	Innovative Computing Technology (INTECH), 2014 Fourth International Conference on : date 13-15 Aug. 2014. IEEE, 2014.

[3]	K. Anwar, J. Siddiqui, and S. Saquib Sohail, “Machine Learning Techniques for Book Recommendation: An Overview.” [Online]. Available: https://ssrn.com/abstract=3356349

## Business Understanding

Perpustakaan saat ini menghadapi tantangan dalam pengelolaan koleksi buku yang besar dengan ruang rak yang terbatas dan jumlah pengguna yang tidak terlalu banyak. Di sisi lain, pengguna semakin bergantung pada teknologi dan internet untuk mencari informasi, termasuk pencarian buku. Untuk meningkatkan layanan dan efisiensi pencarian informasi di perpustakaan, dibutuhkan sistem cerdas yang mampu memberikan rekomendasi buku secara personal.

Penerapan sistem rekomendasi buku dapat membantu pengguna menemukan buku yang relevan tanpa harus mencari secara manual, serta membantu pustakawan dalam pengambilan keputusan terkait pembelian atau penambahan koleksi berdasarkan preferensi dan riwayat peminjaman. Hal ini akan meningkatkan kepuasan pengguna dan efisiensi operasional perpustakaan.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Sistem pencarian buku masih bersifat umum dan belum mempertimbangkan preferensi pengguna secara personal seperti genre favorit, biaya, dan minat sebelumnya.

- Pustakawan kesulitan menentukan buku apa yang sebaiknya dibeli atau dipertahankan dalam koleksi karena tidak adanya sistem yang menganalisis riwayat peminjaman dan tren minat pengguna.

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Meningkatkan efisiensi pencarian buku di perpustakaan dengan menyediakan sistem rekomendasi yang relevan dan personal.

- Meningkatkan kepuasan pengguna melalui saran buku yang sesuai dengan preferensi, minat, atau riwayat bacaan mereka.

### Solution statements
- Dikembangkan sistem rekomendasi buku berbasis data rating dan preferensi pengguna.

- Menyertakan modul analitik untuk membantu pustakawan dalam menentukan buku yang paling relevan untuk ditambahkan ke koleksi, berdasarkan pola minat pengguna dan tren peminjaman.

## Data Understanding
Dataset yang saya gunakan berasal dari kaggle: [goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k?select=books.csv). Dataset ini berisikan 6 file csv yaitu:
- book_tags.csv
- books.csv
- ratings.csv
- sample_book.xml
- tags.csv
- to_read.csv

Dari 6 file csv tersebut, saya hanya menggunakan  file 4 csv yaitu:
- books.csv = Metadata buku
- ratings.csv = rating user terhadap buku
- tags.csv = tags nama buku
- book_tags.csv = tag pada buku

books.csv berisikan 10000 data, 23 kolom 

ratings.csv berisikan 981756 data, 3 kolom 

tags.csv berisi 34252 data, 2 kolom 

book_tags.csv berisi 999912 data, 3 kolom

### Variabel-variabel pada Loan Approval Classification dataset adalah sebagai berikut:
books.csv
- id
- book_id
- best_book_id
- work_id
- books_count
- isbn
- isbn13
- authors
- original_publication_year
- original_title
- title
- language_code
- average_rating
- ratings_count
- work_ratings_count
- work_text_reviews_count
- ratings_1
- ratings_2
- ratings_3
- ratings_4
- ratings_5
- image_url
- small_image_url

ditemukan missing value dikolom:
*   isbn = 700 data
*   isbn13 = 585 data
*   original_publication_year = 21 data
*   original_title = 585 data
*   language_code = 1084 data

Pada data books ditemukan adanya outlier pada kolom average_rating, dimana ditemukan adanya nilai dibawah batas minimal (3.4) dan nilai diatas batas maksimal (4.5)

------------------------------------------------------------------------
ratings.csv
- book_id
- user_id
- rating

Pada data ratings tidak ditemukan missing value, dan data ratings memiliki duplicated data sebanyak 1644 data

-------------------------------------------------------------------------
tags.csv
- tag_id
- tag_name

Pada data tags tidak ditemukan adanya data missing value dan duplicated

------------------------------------------------------------------------
book_tags.csv
- goodreads_book_id
- tag_id
- count

Sedangkan pada data book_tags ditemukan adanya data duplicated sebanyak 6 data, ditemukan juga nama kolom yang tidak konsisten untuk digabung berdasarkan book_id yang sama

## Data Preparation
- Handling Missing Values: Mengatasi nilai kosong pada data
- Handling Duplicated: Menghapus data duplikat
- Rename Columns: Merubah nama kolom untuk memudahkan dalam penggabungan data
- Handling Outlier: Mengubah nilai outlier ke batas maksimal atau minimum menggunakan teknik capping
- Data Integration: Menggabungkan data menjadi 1 DataFrame
- Drop Columns: Menghapus kolom yang tidak digunakan
- Splitting Data: Data akan dibagi menjadi 2, data train dan test menggunakan library sklearn

## Modeling
Pada tahap ini menggunakan metode Collaborative Filtering, untuk memberikan rekomendasi berdasarkan kesamaan histori dengan user lain

  1. Collaborative Filtering
  Memanfaatkan data histori pengguna untuk memberikan rekomendasi berdasarkan kesamaan dengan histori pengguna lain

  Kelebihan
  
  a. Tidak membutuhkan data tambahan seperti deskripsi produk atau genre film
  
  b. Teknik ini dapat menemukan hubungan atau kesamaan yang tidak langsung terlihat antara pengguna dan item.
  
  Kekurangan 
  
  a. (Cold Start) Sulit memberikan rekomendasi kepada pengguna baru atau item baru yang belum memiliki data peringkat yang cukup
  
  b. (Sparsity) Sebagian besar pengguna hanya berinteraksi dengan sebagian kecil item, sehingga data menjadi jarang dan rekomendasi menjadi kurang akurat.

  Output Top-N Rekomendasi
  Output Top 5 Rekomendasi Buku 
| Title | Predicted_Rating |
| ----- | ---------------- |
|  Dune Messiah (Dune Chronicles #2)  |  Predicted Rating: 4.97 |
|  J.R.R. Tolkien 4-Book Boxed Set: The Hobbit and The Lord of the Rings | Predicted Rating. 4.02 |
|  One Hundred Years of Solitude | Predicted Rating: 4.00 |
|  The Phantom Tollbooth | Predicted Rating: 3.98 |
|  Atlas Shrugged | Predicted Rating: 3.96 |
  
## Evaluation

- RMSE: Akurasi dalam memprediksi rating yang akan diberikan pengguna pada buku

- Recall@K: Mengukur dari kumpulan buku, berapa buku yang cocok untuk pengguna dan terdapat dalam daftar rekomendasi

- Precision@K: Dari buku yang dijadikan rekomendasi, berapa buku yang relevan dengan user

- NDCG@K: Mengukur kualitas rekomendasi berdasarkan urutan item yang direkomendasikan memiliki bobot tinggi pada item relevan berada di posisi atas.

## Hasil
RMSE yang dihasilkan pada data latih dan data uji

![RMSE](https://github.com/user-attachments/assets/d01becf0-4308-4c69-981c-b174bef7995f)


Dengan RMSE: 0.1239 atau 1240 dijelaskan bahwa model ternyata dengan baik dapat memprediksi rating yang diberikan user terhadap buku yang direkomendasikan.

Precision@5, Recall@5, NDCG@5 

![Precision@K, Recall@K, NDCG@K](https://github.com/user-attachments/assets/92a3944b-7d45-404c-8285-0b691679b031)

Sedangkan pada recall@k menjelaskan bahwa ternyata dari kumpulan buku yang dimiliki hanya sebesar 0.0400% buku yang cocok direkomendasikan, sedangkan precision@k menjelaskan bahwa ternyata dari buku yang sudah diambil untuk direkomendasikan hanya sebesar 0.3068% item yang relevan dengan user, dan model hanya mampu mengurutkan item dengan bobot terbaik berada di paling atas hanya sebesar 0.4053%

## Kesimpulan
Model collaborative filtering dapat memberikan rekomendasi yang cukup relevan kepada user walaupun hanya sebesar 0.3068%, model juga dapat memprediksi rating yang diberi user dengan lumayan baik.
