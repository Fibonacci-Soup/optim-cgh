import csv
import os
import matplotlib.pyplot as plt

image_filenames = []
delentropy_scatter = [[] for i in range(9)]
entropy_scatter = [[] for i in range(9)]
nmse_scatter = [[] for i in range(9)]

with open(os.path.join('entropy investigations', 'holo_information_investigation_GS_Fresnel0.1.csv')) as holo_info_csv_file:
    holo_info_csv_file_reader = csv.reader(holo_info_csv_file, delimiter=',')
    next(holo_info_csv_file_reader)
    for row in holo_info_csv_file_reader:
        if int(row[3]) == 1:
            image_filenames.append(row[0])
        delentropy_scatter[int(row[3])].append(float(row[2]))
        entropy_scatter[int(row[3])].append(float(row[1]))
        nmse_scatter[int(row[3])].append(float(row[4]))


fig, ax = plt.subplots()
for i in range(1, 9):
    ax.scatter(delentropy_scatter[i], nmse_scatter[i], label="hologram bit depth: " + str(i))
plt.xlabel("Target image delentropy")
plt.ylabel("NMSE between reconstruction and target image")
ax.legend()
plt.show()

fig, ax = plt.subplots()
for i in range(1, 9):
    ax.scatter(entropy_scatter[i], nmse_scatter[i], label="hologram bit depth: " + str(i))
plt.xlabel("Target image entropy")
plt.ylabel("NMSE between reconstruction and target image")
ax.legend()
plt.show()

for i in range(4, len(image_filenames), 5): # only plot for a portion of data
    plt.plot([bit_depth for bit_depth in range(1, 9)], [nmse_scatter[bit_depth][i] for bit_depth in range(1, 9)], label=image_filenames[i])

plt.xlabel("Hologram bit depth")
plt.ylabel("NMSE between reconstruction and target image")
plt.legend(loc='upper right', ncol=5, fontsize=9)
plt.show()