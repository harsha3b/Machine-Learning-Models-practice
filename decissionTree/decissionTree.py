import pandas as pd
import math
import argparse
#
# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--data", help="Data File")
# args = parser.parse_args()
# file = args.data


def readcsv():  # read the data
    # csv_data = pd.read_csv(file, header=None)
    csv_data = pd.read_csv("car.csv", header=None)
    return csv_data


def decision_tree(data, depth=1):
    ig, best_attr = split_info(data)
    att_list = data[best_attr].unique()
    for value in att_list:
        x = att_entropy(best_attr, data, value)
        subData = sub_data(data, best_attr, value)
        if (x != 0 and ig != 0):
            print(f'{depth},{best_attr}={value},{x},no_leaf ')
            decision_tree(subData, depth=depth + 1)
        else:
            print(f'{depth},{best_attr}={value},{x},{subData.iloc[0, -1]}')


def entropy(train_data):  # calculate the entropy

    rows = train_data.shape[0]
    entropy_S = 0
    labels = data.iloc[:, -1].unique()
    log = labels.shape[0]

    for rec in labels:
        classes = train_data[train_data.iloc[:, -1] == rec].shape[0]
        if classes != 0:
            p = classes / rows
            class_entropy = - p * (math.log(p, log))
            entropy_S += class_entropy
    return entropy_S


def att_entropy(col, data_d, attribute_value):
    value_data = data_d[data_d[col] == attribute_value]  # filtering rows with that attribute_value
    attribute_value_entropy = entropy(value_data)  # calculating entropy for the attribute value
    return attribute_value_entropy


def sub_data(data, attr, att_value):
    sub_set = data[data[attr] == att_value]
    return sub_set


def split_info(training_data):
    best_split = None
    ig = 0  # information gain
    class_entropy = entropy(training_data)
    total_row = training_data.shape[0]

    for col in training_data.columns:

        if col != training_data.columns[-1]:

            att_list = training_data[col].unique()
            att_info = 0
            for value in att_list:
                att_count = training_data[training_data[col] == value].shape[0]
                value_entropy = att_entropy(col, training_data, value)
                att_probability = att_count / total_row
                att_info += att_probability * value_entropy
                cur_ig = class_entropy - att_info
        if cur_ig > ig:
            best_split = col
            ig = cur_ig
    return ig, best_split

data = readcsv()
data.columns = ["att" + str(x) for x in range(len(data.iloc[1, :]))]

print(f'0,root,{entropy(data)},no_leaf')
decision_tree(data)
