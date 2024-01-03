import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.sparse import csr_matrix

# Load the dataset
file_path = "groceries - groceries.csv"
df = pd.read_csv(file_path)

# Apriori Algorithm
def apply_apriori(df, min_support=0.01, min_confidence=0.5):
    # Convert the dataset to a one-hot encoded format
    df_encoded = pd.get_dummies(df.drop('Item(s)', axis=1).astype(str))

    # Apply the Apriori algorithm
    frequent_itemsets = apriori(df_encoded.astype(bool), min_support=min_support, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    return frequent_itemsets, rules

# Streamlit app
st.title("Grocery Items Explorer with Apriori Algorithm")

# Display the dataset in a table
st.write("## Raw Dataset")
st.dataframe(df)

# Apply Apriori algorithm
min_support = st.slider("Select minimum support:", 0.01, 0.5, 0.01, 0.01)
min_confidence = st.slider("Select minimum confidence:", 0.1, 1.0, 0.5, 0.1)

frequent_itemsets, rules = apply_apriori(df, min_support, min_confidence)

# Display frequent itemsets
st.write("## Frequent Itemsets")
st.dataframe(frequent_itemsets)

# Display association rules
st.write("## Association Rules")

# Allow the user to select antecedents and consequents
antecedents = st.multiselect("Select antecedents:", df.columns[1:])
consequents = st.multiselect("Select consequents:", df.columns[1:])

# Filter rules based on user selection
filtered_rules = rules[
    (rules['antecedents'].apply(lambda x: set(antecedents).issubset(set(x)))) &
    (rules['consequents'].apply(lambda x: set(consequents).issubset(set(x))))
]

st.dataframe(filtered_rules)
