import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

file_path = r'D:\Uni Related\Data Mining\Horizontal_Format.xlsx'

df = pd.read_excel(file_path)

# Create a DataFrame
df = pd.DataFrame(df)

frequent_itemsets = {}


# Function to perform pruning
def support_pruning(df_before_pruning, min_sup):
    for index, row in df_before_pruning.iterrows():
        count = 0  # Reset count for each row
        for col in df_before_pruning.columns[1:]:  # Start from the second column to exclude 'items'
            if row[col] == 1:  # Check if the value is True
                count += 1
        if count < min_sup:
            # Drop the row if the count is less than min_sup
            df_before_pruning = df_before_pruning.drop(index)
    # Reset the index after dropping rows
    df_before_pruning.reset_index(drop=True, inplace=True)
    df_pruned = df_before_pruning
    return df_pruned


def insert_frequent_itemsets(df_frequent_itemsets):
    for index, row in df_frequent_itemsets.iterrows():
        count = 0
        for col in df_frequent_itemsets.columns[1:]:  # Start from the second column to exclude 'items'
            if row[col]:  # Check if the value is True
                count += 1
        frequent_itemsets[df_frequent_itemsets.iloc[index, 0]] = count


def generate_frequent_itemsets(df_pruned):
    df_after_generation = pd.DataFrame()
    # df_of_items['items'] = df_after_pruning['items']
    # num_rows = len(df_after_pruning)
    items = df_pruned['items'].tolist()
    # combined_values = []
    combinations = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            combinations.append(items[i] + items[j])
    df_after_generation['items'] = combinations

    return df_after_generation


# Split the 'items' column into a list of items
df['items'] = df['items'].str.split(',')

# Explode the list of items into separate rows
df = df.explode('items')

# Group by 'items' and aggregate the transactions into lists for each item
grouped_df = df.groupby('items')['TiD'].agg(list).reset_index()

# Use TransactionEncoder to encode the 'TiD' column
te = TransactionEncoder()
encoded_data = te.fit_transform(
    grouped_df['TiD'].apply(lambda x: [str(x)] if isinstance(x, int) else [str(i) for i in x]))

# Create a new DataFrame with the encoded 'TiD' data
encoded_df = pd.concat([grouped_df['items'].reset_index(drop=True), pd.DataFrame(encoded_data, columns=te.columns_)],
                       axis=1)

encoded_df.replace({True: 1, False: 0}, inplace=True)

min_sup = int(input("Enter min Support"))
min_conf = int(input("Enter min Confidence"))

df_after_pruning = support_pruning(encoded_df, min_sup)
insert_frequent_itemsets(df_after_pruning)
df_after_generating = generate_frequent_itemsets(df_after_pruning)

print(df_after_pruning)
print(frequent_itemsets)
print(df_after_generating)
