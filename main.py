import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import ECLAT_implementation

file_path = r'D:\Uni Related\Data Mining\trial.xlsx'

df = pd.read_excel(file_path)

# Create a DataFrame
input_df = pd.DataFrame(df)


def main():
    # Detect Type
    detected_type = ECLAT_implementation.check_type_of_data(input_df)
    if detected_type == 'Horizontal':
        vertical_df = ECLAT_implementation.horizontal_to_vertical(input_df)
    else:
        vertical_df = ECLAT_implementation.modify_vertical(input_df)
    # take input
    min_sup = int(input("Enter Minimum Support Count"))
    min_conf = int(input("Enter min Confidence in Percentage"))
    min_conf = min_conf / 100
    # support pruning 1st time
    df_after_pruning = ECLAT_implementation.support_pruning(vertical_df, min_sup)
    # insert in frequent item-sets
    ECLAT_implementation.insert_frequent_itemsets(df_after_pruning)
    # generate candidate item-sets of length 2
    df_after_generating_l2 = ECLAT_implementation.generate_candidate_itemsets_l2(df_after_pruning)
    temp_df = df_after_generating_l2
    result = ECLAT_implementation.mine_frequent_itemsets(temp_df, min_sup)
    print(ECLAT_implementation.frequent_itemsets)
    ECLAT_implementation.generate_association_rules(min_conf)
    print('Strong Rules')
    print(ECLAT_implementation.print_strong_rules())

    #ECLAT_implementation.calculate_lift()
    #ECLAT_implementation.extract_unique_combinations()
    #ECLAT_implementation.calculate_lift()
    #print(ECLAT_implementation.all_combinations_list)
    # print(result)
    #for key, value in ECLAT_implementation.association_rules.items():
        #print(f'{key}: {value}')


# If vertical --> Encode
# if horizontal --> transform then encode
# input
# support pruning
# insert in frequent itemsets
# generate candidate itemsets C2
#  ----- In Recursion -----
# support pruning
# insert in frequent itemsets
# generate candidate itemsets Lk
if __name__ == "__main__":
    main()

# Split the 'items' column into a list of items
# df['items'] = df['items'].str.split(',')

# Explode the list of items into separate rows
# df = df.explode('items')

# Group by 'items' and aggregate the transactions into lists for each item
# grouped_df = df.groupby('items')['TiD'].agg(list).reset_index()

# print(grouped_df)

# Use TransactionEncoder to encode the 'TiD' column
# te = TransactionEncoder()
# encoded_data = te.fit_transform(
# grouped_df['TiD'].apply(lambda x: [str(x)] if isinstance(x, int) else [str(i) for i in x]))

# Create a new DataFrame with the encoded 'TiD' data
# encoded_df = pd.concat([grouped_df['items'].reset_index(drop=True), pd.DataFrame(encoded_data, columns=te.columns_)],
# axis=1)

# encoded_df.replace({True: 1, False: 0}, inplace=True)

# print(encoded_df)

# min_sup = int(input("Enter min Support"))
# min_conf = int(input("Enter min Confidence"))

# type = ECLAT_implementation
# df_after_pruning = ECLAT_implementation.support_pruning(encoded_df, min_sup)  # L1
# ECLAT_implementation.insert_frequent_itemsets(df_after_pruning)
# df_after_generating_l2 = ECLAT_implementation.generate_frequent_itemsets_l2(df_after_pruning)
# df_after_pruning_l2 = ECLAT_implementation.support_pruning(df_after_generating_l2, min_sup)
# ECLAT_implementation.insert_frequent_itemsets(df_after_pruning_l2)

# combinations = ECLAT_implementation.generate_frequent_itemsets_lk(df_after_pruning_l2)
# print(df_after_pruning)
# print(frequent_itemsets)
# print(df)
# print(type)
# print(combinations)

# 1 --> support pruning
# 2 --> insert
# 3 --> generate
