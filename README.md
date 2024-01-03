# Laporan Proyek Machine Learning
### Nama : Andri Setiawan
### Nim : 211351018
### Kelas : Malam B

## Domain Proyek

The Groceries Market Basket Dataset, yang dapat ditemukan di sini. Dataset ini berisi 9835 transaksi oleh pelanggan yang berbelanja kebutuhan. Data ini mencakup 169 item unik.
Data ini cocok untuk melakukan penambangan data untuk analisis keranjang pasar yang memiliki beberapa variabel


## Business Understanding

Dalam bidang keilmuan data mining, terdapat suatu metode yang dinamakan association rule. Metode ini bertujuan untuk menunjukkan nilai asosiatif antara jenis-jenis produk yang dibeli oleh pelanggan sehingga terlihatlah suatu pola berupa produk apa saja yang sering dibeli oleh palanggan tersebut.

### Problem Statements

Melalui analisis dataset, kita akan menjelajahi pola pembelian pelanggan, mengidentifikasi item yang sering muncul bersama, dan mengungkap asosiasi yang dapat memberikan wawasan bisnis dan mendukung pengambilan keputusan.

### Goals

Dengan adanya model machine learning ini di harapkan pelaku usaha lebih mudah dalam menentukan paket penjualan item, diskon dan stok barang yang di dasarkan pada kebiasaan pembelian customer

  ### Solution statements
  - Menganalisis pola pembelian pelanggan dan mengidentifikasi item yang sering muncul bersama.
  - Mengungkap asosiasi yang kuat antara item untuk mendukung peluang penjualan lintas produk.
  - Mengoptimalkan strategi pemasaran berdasarkan pola dan asosiasi yang ditemukan.

## Data Understanding
Dataset yang digunakan dalam analisis ini terdiri dari 9835 transaksi yang mencerminkan perilaku belanja pelanggan untuk kebutuhan kelontong. Ini mencakup sejumlah item kelontong, dengan total 169 item unik yang tercakup. Data ini merekam catatan transaksi pelanggan yang berbelanja kebutuhan kelontong, dan setiap baris sesuai dengan satu transaksi. Dataset ini memiliki total 33 kolom. Kolom "Item(s)" menunjukkan jumlah item yang dibeli dalam setiap transaksi, sementara kolom-kolom berikutnya, dari "Item 1" hingga "Item 32," mencantumkan item-individu yang dibeli dalam transaksi tersebut.
[Groceries Market Basket Dataset](https://www.kaggle.com/datasets/irfanasrullah/groceries).

### Variabel-variabel pada Mobile Price Prediction Dataset adalah sebagai berikut:
Jenis inputan type data pada dataset ini yakni integer, kecuali untuk resolusi dan kecepatan CPU
- Price = Harga dari device tersebut (int64) 
- Sale = Tingkat penjualan device tersebut (int64)  
- weight = Berat device tersebut (float64)
- resoloution = tingkat resolusi (float64)
- ppi = tingkat kepadatan pixels (int64)  
- cpu core = jumlah core cpu (int64 ) 
- cpu freq = kecepatan cpu (float64)
- internal mem = kapasitas memori internal (float64)
- ram = kapasitas ram (float64)
- RearCam = resolusi kamera bealakang (float64)
- Front_Cam = resolusi kamera depan (float64)
- battery = kapasitas baterai (int64)  
- thickness = ketebalan device (float64)

## Data Preparation
Pertama tama kita import dulu library python yang ingin di gunakan
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')
```
Selanjutnya kita buka dataset nya

```bash
df = pd.read_csv('mobile-price-prediction/Cellphone.csv')
df.head()
```
Dengan perintad di atas maka dataset akan otomatis terbaca dan akan menampilkan 5 kolom awal.
Selanjutnya bisa kita cek untuk dataset nya berapa jumlah baris dan kolomnya

```bash
df.shape
```
```bash
(161, 14)
```
nah bisa dilihat jika dataset tersebut terdiri dari 161 baris dan 14 kolom

Selanjutnya kita bisa visualisasikan data tersebut dengan sebuah grafik, kita bisa tuliskan
```bash
plt.figure(figsize=(20,15))
j = 1
for i in df.iloc[:,:-1].columns:
    plt.subplot(5,3,j)
    sns.histplot(df[i], stat = "density", kde = True , color = "red")
    j+=1
plt.show()
```
Maka akan muncul
![grafik](https://github.com/andrisetiawan03/uts/assets/148999404/0fdb8277-6e90-457d-94d5-71de3d7fad1a)

atau kita visualisasi kan dengan heat map
```bash
plt.figure(figsize=(8,8))
corr = df.drop(["Product_id"], axis =1 ).corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask=mask, linewidths=.5, annot=True)
plt.show()
```
![output](https://github.com/andrisetiawan03/uts/assets/148999404/ea48237d-f512-464c-a505-1cf58a4e14e0)



Jika sudah selesai pada tahapan ini maka proses bisa dilanjutkan dengan membuat algoritma permodelan


## Modeling
Model yang digunakan adalah model regresi linear, karena output dari proyek ini adalah sebuah estimasi<br>
Pertama bisa kita simpan dulu untuk nilai X dan Y nya

```bash
features = ['ram','cpu core','internal mem','battery','Front_Cam','RearCam','resoloution','cpu freq']
x = df[features]
y = df['Price']
x.shape, y.shape
```
Nah, sudah di lihat diatas untuk nilai X nya apa saja dan nilai Y nya hanya kolom Price.
Selanjutnya bisa dilanjutkan dengan melakukan data training sebanyak 70%
```bash
x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70)
y_test.shape
```
Jika sudah selesai maka bisa di lanjutkan dengan memasukan rumus regresi linear nya

```bash
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(X_test)
```
nah jika sudah sampai pada tahap ini maka proses modeling sudah selesai dan bisa dilakukan pengetesan melalui inputan data array
```bash
input_data = np.array([[12,8,128,5000,14,56,5.1,3.6]])

prediction = lr.predict(input_data)
print('Estimasi harga ponsel :', prediction)
```
Nah nanti akan keluar untuk estimasinya.
Jika sudah selesai, maka kita bisa import model ini dengan menggunakan pickle
```bash
import pickle
filename = 'estimasi_harga_HP.sav'
pickle.dump(lr,open(filename,'wb'))
```
Maka model yang tadi akan tersave dalam estimasi_harga_HP.sav yang bisa kita sambungkan ke file tampilan streamlit

## Evaluation
Proses Evaluasi menggunakan metode Akurasi
```bash
score = lr.score(X_test, y_test)
print('akurasi model regresi linier = ', score)
```
maka akan muncul
```bash
akurasi model regresi linier =  0.9171990085700209
```
Diperoleh tingkat akurasinya 91%, untuk model regreai linear saya rasa cocok untuk menggunakan nilai acuan akurasi. Apalagi ketika akurasinya sudah diatas 70%.

## Deployment
[Link Streamlit untuk Project UAS](https://uas-ml-apriori.streamlit.app/)
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/c215c98e-c02a-4c5f-b749-6ba8ce6893c3)

