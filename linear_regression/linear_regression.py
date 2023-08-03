import numpy as np
import argparse

# parser=argparse.ArgumentParser(description='Linear regression using Gradient descent')
# parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Input file path')
# parser.add_argument('-l', '--eta', type=float, metavar='', required=True, help='Learning rate')
# parser.add_argument('-t', '--threshold', type=float, metavar='', required=True, help='Threshold')
# args=parser.parse_args()
#
# file_location=args.data
# learning_rate=args.eta
# threshold=args.threshold

csv_data = np.genfromtxt("random3.csv", delimiter=",")

#csv_data = np.genfromtxt(file_location, delimiter=",")
columns = int(csv_data.shape[1])
rows = int(csv_data.shape[0])

old_weights = np.zeros(columns)
new_weights = np.zeros(columns)
gradients = np.zeros(columns)
x_values = np.zeros(columns)
y = 0
squared_error = 0
changeInError = 1
count = 0
learning_rate=0.00005
threshold = 0.0001



#Getting the value of x from input file
def get_x_values(csv_data, row, column):
    x_values[0] = 1
    col_index = 0
    while col_index < column - 1:
        x_values[col_index + 1] = csv_data[row, col_index]
        col_index += 1
    return x_values

#Getting the value of y from input file
def get_y_values(csv_data, row, column):

    y = csv_data[row, column - 1]
    return y


#copying the weights
def copying_weights(new_weights, old_weights, columns):
    col_index = 0
    while col_index < columns:
        old_weights[col_index] = new_weights[col_index]
        col_index += 1
    return old_weights


# f(xi)=wi*xi calculation
def function(old_weights, x_values):
    y_dash = 0
    for w, x in zip(old_weights, x_values):
        y_dash += w * x

    return y_dash


while changeInError > threshold:
    gradients = np.zeros(columns)
    old_squared_error = squared_error
    squared_error = 0
    row_index = 0

    while row_index < rows:
        old_weights = copying_weights(new_weights, old_weights, columns)
        x_values = get_x_values(csv_data, row_index, columns)
        y=get_y_values(csv_data, row_index, columns)
        result = function(old_weights, x_values)

        col_index = 0
        while col_index < columns:
            gradients[col_index] += x_values[col_index] * (y - result)
            col_index += 1

        squared_error += (y - result) ** 2
        row_index += 1

    col_index = 0
    while col_index < columns:
        new_weights[col_index] += gradients[col_index] * learning_rate
        col_index += 1
    changeInError = abs(old_squared_error - squared_error)
    print("{0},{1},{2},{3},{4}".format(count,old_weights[0],old_weights[1],old_weights[2],squared_error))
    count += 1