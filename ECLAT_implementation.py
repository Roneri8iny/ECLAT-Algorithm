import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations

association_rules = {}
strong_association_rules = {}
frequent_itemsets = {}

te = TransactionEncoder()


# frequent_itemsets_df = pd.DataFrame()

# Function to perform pruning
def support_pruning(df_before_pruning, min_sup):
    for index, row in df_before_pruning.iterrows():
        count = 0  # Reset count for each row
        # for col in df_before_pruning.columns[1:]:  # Start from the second column to exclude 'items'
        count = row.iloc[1:].sum()
        # print(count)
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


def generate_candidate_itemsets_l2(df_pruned):
    df_after_generation = pd.DataFrame()
    result_list = df_pruned.values.tolist()
    # df_of_items['items'] = df_after_pruning['items']
    # num_rows = len(df_after_pruning)
    items = df_pruned['items'].tolist()
    transactions = df_pruned.drop('items', axis=1)
    transactions_list = transactions.iloc[0:, :].values.tolist()
    # combined_values = []
    combinations_items = []
    combinations_trans = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            combinations_items.append(items[i] + ',' + items[j])
    df_after_generation['items'] = combinations_items
    for i in range(len(transactions_list)):
        for j in range(i + 1, len(transactions_list)):
            result = [int(transactions_list[i][k]) & int(transactions_list[j][k]) for k in
                      range(len(transactions_list[i]))]
            combinations_trans.append(result)
    temp_df = pd.DataFrame(combinations_trans)
    temp_df.columns = range(1, len(temp_df.columns) + 1)
    result_df = pd.concat([df_after_generation, temp_df], axis=1)
    return result_df


def generate_combinations(items):
    item_combinations = []
    for r in range(1, len(items)):
        for subset in combinations(items, r):
            remaining = list(set(items) - set(subset))
            item_combinations.append((subset, remaining))
    return item_combinations


def get_support_count(itemset):
    return frequent_itemsets.get(itemset, 0)


freq_item_list = []
combinations_list = []
strong_association_rules = []


def generate_association_rules(min_confidence):
    min_confidence = min_confidence / 100
    frequent_itemsets_df = pd.DataFrame.from_dict(frequent_itemsets, orient='index',
                                                  columns=['frequent_itemsets']).reset_index()
    # items_list = [(key, value) for key, value in frequent_itemsets.items()]
    frequent_itemsets_df.columns = ['frequent_itemsets', 'support_count']
    for index, row in frequent_itemsets_df.iterrows():
        temp_items = row['frequent_itemsets'].split(',')
        freq_item_list[:] = temp_items
        temp_combinations = generate_combinations(freq_item_list)
        combinations_list[:] = temp_combinations
        for combination in combinations_list:
            lhs = combination[0]
            support_count_lhs = get_support_count(','.join(lhs))
            support_count_key = get_support_count(row['frequent_itemsets'])
            confidence = support_count_key / support_count_lhs
            if confidence >= min_confidence:
                strong_rule = {'lhs': lhs, 'confidence': confidence, 'rhs': combination[1]}
                strong_association_rules.append(strong_rule)
            print(f"Rule: {combination[0]} -> {combination[1]} : Confidence = {confidence}")
    # print(frequent_itemsets_df)
    # rows_with_apple = df.loc[df['B'] == 'apple']
    # for i in range(len(df_index_oriented)):
    # for index, row in df_index_oriented.iterrows():
    # value = row.iloc[0]
    '''
    for key, value in items_list:
        #items = items_list.split(',')
        #print(items)
        if len(key) > 1:
            for i in range(1, len(key)):
                for subset in itertools.combinations(key, i):
                    # subset_str = ''.join(sorted(subset))
                    # lhs = subset_str
                    # rhs = ''.join(sorted(set(key) - set(subset)))
                    subset_frozenset = frozenset(subset)
                    lhs = subset_frozenset
                    rhs = frozenset(key) - subset_frozenset

                    print(lhs)
                    print(rhs)
                    # Assuming 'frequent_itemsets' is a dictionary containing support counts
                    confidence = frequent_itemsets.get(key) / frequent_itemsets.get(lhs)
                    ar_key = "{}->{}".format(lhs, rhs)
                    association_rules[ar_key] = confidence
                    if confidence >= min_confidence:
                        strong_association_rules[ar_key] = confidence
    '''


def generate_candidate_itemsets_l2(df_pruned):
    df_after_generation = pd.DataFrame()
    result_list = df_pruned.values.tolist()
    # df_of_items['items'] = df_after_pruning['items']
    # num_rows = len(df_after_pruning)
    items = df_pruned['items'].tolist()
    transactions = df_pruned.drop('items', axis=1)
    transactions_list = transactions.iloc[0:, :].values.tolist()
    # combined_values = []
    combinations_items = []
    combinations_trans = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            combinations_items.append(items[i] + ',' + items[j])
    df_after_generation['items'] = combinations_items
    for i in range(len(transactions_list)):
        for j in range(i + 1, len(transactions_list)):
            result = [int(transactions_list[i][k]) & int(transactions_list[j][k]) for k in
                      range(len(transactions_list[i]))]
            combinations_trans.append(result)
    temp_df = pd.DataFrame(combinations_trans)
    temp_df.columns = range(1, len(temp_df.columns) + 1)
    result_df = pd.concat([df_after_generation, temp_df], axis=1)
    return result_df


def generate_candidate_itemsets_lk(df_pruned_l2):
    # df_after_generation_lk = pd.DataFrame()
    result_list = df_pruned_l2.values.tolist()
    combinations_items = []
    combinations_trans = []
    result_df = pd.DataFrame()
    # df_after_generation = pd.DataFrame()
    # temp_df = pd.DataFrame()
    # print(result_list)
    for item1 in result_list:
        for item2 in result_list:
            if item1 != item2:
                substrings1 = item1[0].split(',')
                substrings2 = item2[0].split(',')
                substring1 = substrings1[1]  # Substring after first comma in string1
                substring2 = substrings2[-2]
                intersection = set(substring1) & set(substring2)
                if len(intersection) == len(substring1):  # Ensure two elements match
                    combined_items = ', '.join(sorted(set(substrings1 + substrings2)))
                    combinations_items.append(combined_items)
                    result = [int(item1[k + 1]) & int(item2[k + 1]) for k in range(len(item1) - 1)]
                    combinations_trans.append(result)

    result_df['items'] = combinations_items
    temp_df = pd.DataFrame(combinations_trans)
    temp_df.columns = range(1, len(temp_df.columns) + 1)
    result_df = pd.concat([result_df, temp_df], axis=1)

    return result_df


def check_type_of_data(initial_df):
    column_names = initial_df.columns.tolist()
    # Assign column names to variables
    column_1 = column_names[0]
    column_2 = column_names[1]
    if column_1.lower() == 'tid'.lower() and column_2.lower() == 'items'.lower():
        return "Horizontal"
    else:
        return "Vertical"


def horizontal_to_vertical(horizontal_df):
    # Split the 'items' column into a list of items
    horizontal_df['items'] = horizontal_df['items'].str.split(',')

    # Explode the list of items into separate rows
    horizontal_df = horizontal_df.explode('items')

    # Group by 'items' and aggregate the transactions into lists for each item
    grouped_df = horizontal_df.groupby('items')['TiD'].agg(list).reset_index()

    # Use TransactionEncoder to encode the 'TiD' column

    encoded_data = te.fit_transform(
        grouped_df['TiD'].apply(lambda x: [str(x)] if isinstance(x, int) else [str(i) for i in x]))

    # Create a new DataFrame with the encoded 'TiD' data
    encoded_df = pd.concat(
        [grouped_df['items'].reset_index(drop=True), pd.DataFrame(encoded_data, columns=te.columns_)],
        axis=1)

    encoded_df.replace({True: 1, False: 0}, inplace=True)
    return encoded_df


def modify_vertical(vertical_df):
    vertical_df = vertical_df.explode('TiD')
    # Converting the 'TiD' column to string
    vertical_df['TiD'] = vertical_df['TiD'].astype(str)

    # Splitting the 'TiD' values into separate columns
    vertical_df = vertical_df.pivot_table(index='items', columns='TiD', aggfunc=lambda x: '1').fillna('0')
    # Resetting the index to make 'items' a regular column
    vertical_df = vertical_df.reset_index()
    modified_vertical_df = vertical_df
    return modified_vertical_df


def mine_frequent_itemsets(temp_df, min_sup):
    temp_df = support_pruning(temp_df, min_sup)
    insert_frequent_itemsets(temp_df)
    temp_df_after_generating = generate_candidate_itemsets_lk(temp_df)
    if len(temp_df) < 2 or len(temp_df_after_generating) == 0:
        return temp_df
    else:
        return mine_frequent_itemsets(temp_df_after_generating, min_sup)
