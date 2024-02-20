import csv
import os
import PIL.Image
import matplotlib.pyplot as plt

delentropy = {}
delentropy_scatter = [[] for i in range(9)]
nmse_scatter = [[] for i in range(9)]
with open(os.path.join('entropy investigations', 'delentropy_results.csv')) as csv_file_delentropy:
    csv_file_read_delentropy = csv.reader(csv_file_delentropy, delimiter=',')
    next(csv_file_read_delentropy)
    for row in csv_file_read_delentropy:
        if row[0] not in delentropy:
            delentropy[row[0]] = float(row[1])

with open(os.path.join('entropy investigations', 'entropy_investigation.csv')) as csv_file_nmse:
    csv_file_read_nmse = csv.reader(csv_file_nmse, delimiter=',')
    next(csv_file_read_nmse)
    for row in csv_file_read_nmse:
        delentropy_scatter[int(row[2])].append(delentropy[row[0]])
        nmse_scatter[int(row[2])].append(float(row[3]))


fig, ax = plt.subplots()
for i in range(1, 9):
    ax.scatter(delentropy_scatter[i], nmse_scatter[i], label="holo bit depth = " + str(i))
plt.xlabel("Delentropy of the target image")
plt.ylabel("NMSE of the reconstruction to the target image")
ax.legend()
plt.show()