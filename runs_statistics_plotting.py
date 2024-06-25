import matplotlib.pyplot as plt
import csv
import os

with open(os.path.join('Output', 'runs_statistics.csv')) as csvfile:
    csv_reader = csv.reader(csvfile)
    x_list = []
    y_list = []
    for i, row in enumerate(csv_reader):
        if i in range(29, 37):
            x_list.append(int(row[4]))
            y_list.append(float(row[8]))
    print(x_list, y_list)
    plt.xticks(x_list)
    plt.plot(x_list, y_list, 'x--', label='mandrill')
    plt.xlabel("Number of Frames")
    plt.ylabel("Final NMSE")
    plt.legend()
    plt.show()