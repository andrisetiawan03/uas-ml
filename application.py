import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_csv("groceries - groceries.csv")

item_columns = df.columns[1:33]

transactions = df[item_columns].apply(lambda row: row.dropna().tolist(), axis=1).tolist()

onehot_transactions = pd.DataFrame(transactions)

onehot_encoded = pd.get_dummies(onehot_transactions.unstack()).groupby(level=1).max()

st.title("Association rules pada data belanja dengan algoritma Apriori")

def user_input_feature():
    item_list = df['Item 1'].unique().tolist()
    item = st.selectbox("Item", item_list)
    return item

item = user_input_feature()


frequent_itemsets = apriori(onehot_encoded, min_support=0.03, use_colnames=True)

sorted_frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

sorted_frequent_itemsets["length"] = sorted_frequent_itemsets["itemsets"].apply(len)

with pd.option_context("display.max_rows", None,
                       "display.max_columns", None,
                       "display.precision", 3,
                       ):
  print(sorted_frequent_itemsets)

support = 0.01

metric = "lift"
min_treshold = 1

rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_treshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)

# ... (previous code)

def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    filtered_data = data.loc[data["antecedents"] == item_antecedents]

    if not filtered_data.empty:
        return list(filtered_data.iloc[0, :])
    else:
        return []

# Move the indentation of the following block outside the function
st.markdown("Hasil Rekomendasi : ")
result = return_item_df(item)
if result:
    st.success(f"Jika Konsumen Membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")
else:
    st.warning("Tidak ditemukan rekomendasi untuk item yang dipilih")

