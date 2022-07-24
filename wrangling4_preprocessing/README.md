# Script ini dibuat untuk simple data cleaning pada data mobil dan berbagai fiturnya

# Output dari script ini adalah data frame yang dapat digunakan untuk melakukan prediksi harga mobil dengan pemodelan tertentu

# Proses pembersihan data dimulai dari:
- mengubah format kolom tanggal menjadi datetime
- menghapus kolom yang tidak diperlukan
- menghapus satuan pada kolom seperti $ dan km
- menghapus data outlier
- mengimputasi baris bernilai NaN
- melakukan normalisasi data numerikal
- melakukan encoding pada data categorical
- menyimpan data frame yang telah bersih

# Cara menggunakan:
- unduh semua elemen pada project ini
- mulai dengan aktivasi script Scripts\activate 
- menginstall package yang diperlukan python setup.py develop
- menjalankan script python main.py