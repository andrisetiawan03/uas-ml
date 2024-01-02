# Laporan Proyek Machine Learning
### Nama : Andri Setiawan
### Nim : 211351018
### Kelas : Malam B

## Domain Proyek

Proyek ini memudahkan pengggunanya untuk melihat estimasi harga ponsel berdasarkan spesifikasi yang di inginkan, sehingga bisa menjadi acuan untuk dana yang harus di siapkan


## Business Understanding

Dengan adanya apliakasi ini, memudahkan penggunanya untuk menyiapkan dana guna mendapat ponsel dengan spesifikasi yang di inginkan dengan cara mengkalkulasikan estimasi nya menggunakan regresi linear

Bagian laporan ini mencakup:

### Problem Statements

Sering kali kita tertipu sales ponsel yang menjual ponsel dengan harga tidak masuk akal tapi spesifikasi dipertanyakan

### Goals

Dengan adanya model machine learning ini di harapkan kita terhindar dari hal tersebut karena sudah punya patokan untuk harga nya berdsasarkan rincian spesifikasinya.

  ### Solution statements
  - Untuk pemodelan estimasi, digunakan algoritma regresi linear
  - Proses pengkalkulasian di lakukan berdasarkan estimasi data yang di sajikan berdasarkan rincian spesifikasinya

## Data Understanding
Proyek ini didasarkan pada dataset yang diambil dari kaggle perihal estimasi harga ponsel berdasarkan rincian spesifikasinya. Untuk dataset nya bisa di ambil disini<br>

[Ponsel Price](https://www.kaggle.com/datasets/mohannapd/mobile-price-prediction).

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
[Link Streamlit untuk Project UTS](https://estimasiponsel.streamlit.app/)
![alt text](https://github.com/andrisetiawan03/uts/blob/main/tampilan.png)
