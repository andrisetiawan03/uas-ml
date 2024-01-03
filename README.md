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
Dataset yang digunakan dalam analisis ini terdiri dari 9835 transaksi yang mencerminkan perilaku belanja pelanggan untuk kebutuhan kelontong. Ini mencakup sejumlah item kelontong, dengan total 169 item unik yang tercakup. Data ini merekam catatan transaksi pelanggan yang berbelanja kebutuhan kelontong, dan setiap baris sesuai dengan satu transaksi. Dataset ini memiliki total 33 kolom. Kolom "Item(s)" menunjukkan jumlah item yang dibeli dalam setiap transaksi, sementara kolom-kolom berikutnya, dari "Item 1" hingga "Item 32," mencantumkan item-individu yang dibeli dalam transaksi tersebut.<br>
[Groceries Market Basket Dataset](https://www.kaggle.com/datasets/irfanasrullah/groceries).

### Variabel-variabel pada Mobile Price Prediction Dataset adalah sebagai berikut:
Hanya tersedia satu variable yakni untuk list item saja, dan terdapar 33 kolom

## Data Preparation
Pertama tama kita import dulu library python yang ingin di gunakan
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
```
Selanjutnya kita buka dataset nya

```bash
df = pd.read_csv("/content/groceries/groceries - groceries.csv")
df.head()
```
Dengan perintad di atas maka dataset akan otomatis terbaca dan akan menampilkan 5 kolom awal.
Selanjutnya bisa kita cek untuk dataset nya berapa jumlah baris dan kolomnya

```bash
df.shape
```
Kita bisa menentukan 20 produk terlaris
```bash
plt.rcParams['figure.figsize']=20,7
sns.countplot(data=df, x=df['Item 1'],
             order = df['Item 1'].value_counts().head(20).index,
             palette='cool')
plt.xticks(rotation=90)
plt.xlabel('Product')
plt.title('Top 20 frequently bought products')
plt.show()
```
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/19cf73be-1f89-470a-b2c6-c6cc2c189612)

kita bisa melihat distribusi transaksi
```bash
plt.figure(figsize=(8, 6))

plt.boxplot(num_items_per_transaction, vert=False, patch_artist=True)

plt.title("Distribution of Transaction Sizes")
plt.xlabel("Number of Items")
plt.grid(axis="x")
plt.yticks([])
plt.tight_layout()

plt.show()
```
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/c764c06b-1570-475b-ba36-e179a0daafe6)

Kita lihat persentasenya
```bash
min_size = min(num_items_per_transaction)
max_size = max(num_items_per_transaction)

bins = list(range(min_size, max_size + 2))

plt.hist(num_items_per_transaction, bins=bins, edgecolor="black", align="left", rwidth=0.8)
plt.title("Transaction Size Distribution")
plt.xlabel("Number of Items in Transaction")
plt.ylabel("Frequency")

item_count = df["Item(s)"].value_counts()
total_transactions = len(df["Item(s)"])
percentage_item_purchases = (item_count / total_transactions) * 100

height_first_bar = plt.gca().patches[0].get_height()

plt.annotate(f"{round(percentage_item_purchases[1])}%",
             xy=(bins[0] + 0.5, height_first_bar),
             xytext=(0, 3),
             textcoords="offset points",
             ha='center',
             fontsize=10)

plt.show()
```
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/dfe5ac84-734f-436a-b7d0-949906135d09)

Kita lihat jumlah item yang di beli dalam 10 pembelian terbanyak
```bash
item_counts = df.iloc[:, 1:].stack().value_counts()

sorted_items_desc = item_counts.sort_values(ascending=False)

top_10_items = sorted_items_desc.head(10)

plt.figure(figsize=(10, 6))

bar_color = "#9AC2C5"

bars = plt.barh(range(len(top_10_items)), top_10_items.values, color=bar_color)

plt.yticks(range(len(top_10_items)), top_10_items.index)

plt.gca().invert_yaxis()

plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)


plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

for index, value in enumerate(top_10_items.values):
    plt.text(value + 5, index, str(value), ha="left", va="center")


plt.grid(False)

plt.ylabel("")
plt.xlabel("")

plt.title("Top 10 Most Frequent Items")

plt.tight_layout()
plt.show()
```
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/369717a2-3386-42d1-b945-bbb085c464cc)

Kita juga bisa memvisualkan top 5 item yang di beli tapi tidak berelasi dengan item lainya
```bash
item_columns = df.columns[1:33]

standalone_purchases = df[df["Item(s)"] == 1][item_columns]

standalone_item_counts = standalone_purchases.stack().value_counts()
top_standalone_items = standalone_item_counts.head(5)

plt.figure(figsize=(10, 6))


bar_color = "#7EB5D6"

plt.barh(top_standalone_items.index, top_standalone_items.values, color=bar_color, height=0.5)

plt.ylabel("")
plt.xlabel("")

plt.title("Top 5 Most Frequent Standalone Items")


for index, value in enumerate(top_standalone_items.values):
    plt.text(value, index, str(value), ha="left", va="center", color="black")

plt.gca().invert_yaxis()

plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

plt.grid(False)

plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

plt.tight_layout()

plt.show()
```
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/d6d907a8-6bbe-47b5-b082-92d37c21737c)




Jika sudah selesai pada tahapan ini maka proses bisa dilanjutkan dengan membuat algoritma permodelan


## Modeling
Kita lakukan modeling sembari menghapus data kosong
```bash
item_columns = df.columns[1:33]

transactions = df[item_columns].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

onehot_transactions = pd.DataFrame(transactions)

onehot_encoded = pd.get_dummies(onehot_transactions.unstack()).groupby(level=1).max()
```
Selanjutnya kita tentukan nilai support nya
```bash
min_support_values = [0.07, 0.05, 0.03, 0.02]

confidence_levels = list(np.arange(0.05, 1.05, 0.05))

num_rules_lists = []

for min_support in min_support_values:
    frequent_itemsets = apriori(onehot_encoded, min_support=min_support, use_colnames=True)
    rules_list = []
    for confidence_level in confidence_levels:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence_level)
        num_rules = len(rules)
        rules_list.append(num_rules)
    num_rules_lists.append(rules_list)

plt.figure(figsize=(10, 6))

colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

for i, min_support in enumerate(min_support_values):
    plt.plot(confidence_levels, num_rules_lists[i], marker="o", color=colors[i], label=f"Min Support: {min_support}")

plt.xlabel("Confidence Level")
plt.ylabel("Number of Rules")
plt.title("Number of Rules vs. Confidence Level for Different Minimum Support")

plt.xticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1])

plt.grid(True, linestyle="--", alpha=0.7)

plt.legend()
plt.show()
```
  ![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/94160bb6-fcb4-42a9-ba9f-8c6709656e55)

Selanjutnya kita hitung item yang di beli
```bash
frequent_itemsets = apriori(onehot_encoded, min_support=0.03, use_colnames=True)

sorted_frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

sorted_frequent_itemsets["length"] = sorted_frequent_itemsets["itemsets"].apply(len)

with pd.option_context("display.max_rows", None,
                       "display.max_columns", None,
                       "display.precision", 3,
                       ):
  print(sorted_frequent_itemsets)
```


## Evaluation
Karena ini adalah apriori algorithm kita evaluasi berdasarkan nilai Association Rules
```bash
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

sorted_association_rules = association_rules_df.sort_values(by="lift", ascending=False).reset_index(drop=True)

print("\nAssociation Rules:")
sorted_association_rules
```
![Screenshot (104)](https://github.com/andrisetiawan03/uas-ml/assets/148999404/2de8184c-c60e-4328-b81f-1043b254f1ca)


## Deployment
[Link Streamlit untuk Project UAS](https://uas-ml-apriori.streamlit.app/)
![image](https://github.com/andrisetiawan03/uas-ml/assets/148999404/c215c98e-c02a-4c5f-b749-6ba8ce6893c3)

