import csv
import os
import PIL.Image
import matplotlib.pyplot as plt

file_jpg_compression_rate = {}
jpeg_size_scatter = [[] for i in range(9)]
nmse_scatter = [[] for i in range(9)]

with open('entropy_investigation.csv') as csv_file:
    csv_file_read = csv.reader(csv_file, delimiter=',')

    for row in csv_file_read:
        img_path = os.path.join('..', 'Target_images', 'DIV2K_train_HR', row[0])
        if os.path.isfile(img_path):
            if row[0] not in file_jpg_compression_rate.keys():
                this_image = PIL.Image.open(img_path)
                this_image.save("temp_png.png", "PNG", optimize=True)
                file_jpg_compression_rate[row[0]] = os.path.getsize("temp_jpeg.jpg") / os.path.getsize(img_path) * 100
            jpeg_size_scatter[int(row[2])].append(file_jpg_compression_rate[row[0]])
            nmse_scatter[int(row[2])].append(float(row[3]))

fig, ax = plt.subplots()
for i in range(1, 9):
    ax.scatter(jpeg_size_scatter[i], nmse_scatter[i], label="holo bit depth = " + str(i))
plt.xlabel("Target image jpeg compression rate (%)")
plt.ylabel("NMSE")
ax.legend()
plt.show()