import csv
import os
# import PIL.Image
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
    with open('holo_information_investigation.csv', 'w', newline='') as output_file:
        file_writer = csv.writer(output_file)
        file_writer.writerow(['image_filename', 'image_entropy', 'image_delentropy', 'hologram_bit_depth', 'NMSE', 'hologram_entropy'])
        for row in csv_file_read_nmse:
            delentropy_scatter[int(row[2])].append(delentropy[row[0]])
            nmse_scatter[int(row[2])].append(float(row[3]))
            file_writer.writerow([row[0], row[1], delentropy[row[0]], row[2], row[3], row[4]])



fig, ax = plt.subplots()
for i in range(1, 9):
    ax.scatter(delentropy_scatter[i], nmse_scatter[i], label="hologram bit depth: " + str(i))
plt.xlabel("Target image delentropy")
plt.ylabel("NMSE between reconstruction and target image")
ax.legend()
plt.show()