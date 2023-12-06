import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from itertools import combinations, chain

association_rules = {}
frequent_itemsets = {}

te = TransactionEncoder()

freq_item_list = []
combinations_list = []
strong_association_rules = []
all_combinations_list = []


# unique_combinations = []


# unique_combinations = []

# Function to perform pruning
def support_pruning(df_before_pruning, min_sup):
    for index, row in df_before_pruning.iterrows():
        count = 0  # Reset count for each row
        count = row.iloc[1:].sum()
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


'''
def generate_candidate_itemsets_l2(df_pruned):
    df_after_generation = pd.DataFrame()
    result_list = df_pruned.values.tolist()
    items = df_pruned['items'].tolist()
    transactions = df_pruned.drop('items', axis=1)
    transactions_list = transactions.iloc[0:, :].values.tolist()
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

'''


def generate_combinations(items):
    item_combinations = []
    for r in range(1, len(items)):
        for subset in combinations(items, r):
            remaining = list(set(items) - set(subset))
            item_combinations.append((subset, remaining))
    return item_combinations


def get_support_count(itemset):
    return frequent_itemsets.get(itemset, 0)


def rule_exists(rule):
    found = False
    for item in all_combinations_list:
        if item == rule:
            found = True
            break
    if found:
        return True
    else:
        return False


def generate_association_rules(min_confidence):
    min_confidence = min_confidence
    original_rules_list = []
    frequent_itemsets_df = pd.DataFrame.from_dict(frequent_itemsets, orient='index',
                                                  columns=['frequent_itemsets']).reset_index()
    frequent_itemsets_df.columns = ['frequent_itemsets', 'support_count']
    for index, row in frequent_itemsets_df.iterrows():
        temp_items = row['frequent_itemsets'].split(',')
        freq_item_list[:] = temp_items
        temp_combinations = generate_combinations(freq_item_list)
        combinations_list[:] = temp_combinations
        all_combinations_list.extend(combinations_list)  #(combination[0], combination[1] )
        for combination in combinations_list:
            lhs = sorted(combination[0])
            rhs = sorted(combination[1])
            support_count_lhs = get_support_count(','.join(lhs))
            support_count_rhs = get_support_count(','.join(rhs))
            support_count_key = get_support_count(row['frequent_itemsets'])
            confidence = support_count_key / support_count_lhs
            reverse_rule = (tuple(rhs), tuple(lhs))
            original_rule = (tuple(lhs), tuple(rhs))
            original_rules_list.append(original_rule)
            #original_flattened_list = [item for sublist in original_rules_list for item in sublist]
            if rule_exists(reverse_rule) == False:
                denominator = float(support_count_lhs * support_count_rhs)
                lift = support_count_key / denominator
                print(f"Rule: {lhs} -> {rhs} : Lift = {lift}")

            if confidence >= min_confidence:
                strong_rule = {'lhs': lhs, 'confidence': confidence, 'rhs': combination[1]}
                strong_association_rules.append(strong_rule)
            print(f"Rule: {combination[0]} -> {combination[1]} : Confidence = {confidence}")


def generate_candidate_itemsets_l2(df_pruned):
    df_after_generation = pd.DataFrame()
    result_list = df_pruned.values.tolist()
    items = df_pruned['items'].tolist()
    transactions = df_pruned.drop('items', axis=1)
    transactions_list = transactions.iloc[0:, :].values.tolist()
    combinations_items = []
    combinations_trans = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            combinations_items.append(items[i] + ',' + items[j])
    df_after_generation['items'] = combinations_items
    for i in range(len(transactions_list)):
        for j in range(i + 1, len(transactions_list)):
            result = [int(transactions_list[i][k]) & int(transactions_list[j][k]) for k in range(len(transactions_list[i]))]
            combinations_trans.append(result)
    temp_df = pd.DataFrame(combinations_trans)
    temp_df.columns = range(1, len(temp_df.columns) + 1)
    result_df = pd.concat([df_after_generation, temp_df], axis=1)
    return result_df


def generate_candidate_itemsets_lk(df_pruned_l2):
    result_list = df_pruned_l2.values.tolist()
    combinations_items = []
    combinations_trans = []
    result_df = pd.DataFrame()
    for item1 in result_list:
        for item2 in result_list:
            if item1 != item2:
                substrings1 = item1[0].split(',') # A -> B   A,B   B,C
                substrings2 = item2[0].split(',')
                substring1 = substrings1[1]  # Substring after first comma in string1
                substring2 = substrings2[-2]
                intersection = set(substring1) & set(substring2)  ## find the midle part of the string
                if len(intersection) == len(substring1):  # Ensure two elements match
                    combined_items = ','.join(sorted(set(substrings1 + substrings2)))
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


def print_strong_rules():
    for rule in strong_association_rules:
        lhs = ', '.join(rule['lhs'])
        rhs = ', '.join(rule['rhs'])
        confidence = rule['confidence']
        print(f"{lhs} -> {rhs} confidence = {confidence}")
    return 'Done'


'''
def extract_unique_combinations():
    seen_combinations = set()
    for combination in all_combinations_list:
        if len(combination) == 2 and len(combination[0]) == 1 and len(combination[1]) >= 1:
            item_1, item_2 = combination[0][0], combination[1][0]
            forward_combination = f"{item_1} -> {item_2}"
            reverse_combination = f"{item_2} -> {item_1}"

            if reverse_combination not in seen_combinations:
                unique_combinations.append(combination)
                seen_combinations.add(forward_combination)

    # Print or use unique_combinations as needed
    for combo in unique_combinations:
        # print('combo')
        print(combo)
    # return unique_combinations

'''
'''
def generate_unique_combinations(items):

    for item in items:
        if len(item) > 1:
            all_unique_combinations = chain.from_iterable(combinations(item, r) for r in range(1, len(item)))

            for combination in all_unique_combinations:
                lhs = frozenset(combination)
                rhs = frozenset(item - lhs)
    return item_combinations



def calculate_lift():
    #unique_combinations = []
    #indices = []
    copy_all_combinations = all_combinations_list.copy()
    frequent_itemsets_df = pd.DataFrame.from_dict(frequent_itemsets, orient='index',
                                                  columns=['frequent_itemsets']).reset_index()
    frequent_itemsets_df.columns = ['frequent_itemsets', 'support_count']
    # print(copy_all_combinations)
    for item, support in frequent_itemsets.items():
        #temp_items = row['frequent_itemsets'].split(',')
        #freq_item_list[:] = temp_items
        #for item in freq_item_list:
        item_without_commas = tuple(x for x in item if x != ',')

        if len(item_without_commas) > 1:
            item_set = set(item_without_commas)
            all_unique_combinations = chain.from_iterable(combinations(item_set, r) for r in range(1, len(item_set) + 1))
            for combination in all_unique_combinations:
                print(combination)
            for combination in all_unique_combinations:
                combination_without_commas = tuple(x for x in combination if x != ',')
                lhs = frozenset(sorted(combination_without_commas))
                rhs = frozenset(sorted(item_set - lhs))
                support_count_lhs = get_support_count(','.join(lhs))
                support_count_rhs = get_support_count(','.join(rhs))
                #support_count_key = get_support_count(row['frequent_itemsets'])
                lift = support / (support_count_lhs * support_count_rhs)
                if lift > 1:
                    print(f"Rule: {combination[0]} -> {combination[1]} : Lift = {lift} : Positive Correlation")
                elif lift < 1:
                    print(f"Rule: {combination[0]} -> {combination[1]} : Lift = {lift} : Negative Correlation")
                else:
                    print(f"Rule: {combination[0]} -> {combination[1]} : Lift = {lift} : No Correlation")


    for i in range(len(copy_all_combinations)):
        current = copy_all_combinations[i]
        for j in range(i + 1, len(copy_all_combinations)):
            next_item = copy_all_combinations[j]
            #is_unique = True
            original_current = (sorted(current[0]), sorted(current[1]))
            reverse_next = (sorted(next_item[1]), sorted(next_item[0]))  # Create a reverse tuple
            if reverse_next not in unique_combinations and original_current not in unique_combinations:
                unique_combinations.append(current)
                break
    
    print(unique_combinations)
    
    for index, row in frequent_itemsets_df.iterrows():
        for combination in unique_combinations:
            lhs = sorted(combination[0])
            rhs = sorted(combination[1])
            support_count_lhs = get_support_count(','.join(lhs))
            support_count_rhs = get_support_count(','.join(rhs))
            support_count_key = get_support_count(row['frequent_itemsets'])
            lift = support_count_key / (support_count_lhs * support_count_rhs)
            if lift > 1:
                print(f"Rule: {combination[0]} -> {combination[1]} : Lift = {lift} : Positive Correlation")
            elif lift < 1:
                print(f"Rule: {combination[0]} -> {combination[1]} : Lift = {lift} : Negative Correlation")
            else:
                print(f"Rule: {combination[0]} -> {combination[1]} : Lift = {lift} : No Correlation")
'''
