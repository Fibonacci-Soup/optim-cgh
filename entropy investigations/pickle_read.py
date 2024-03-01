import numpy
import random
import matplotlib.pyplot as plt
import pickle
with open('entropy investigations/variables_entropy_investigation.pkl', 'rb') as f:
    [image_filenames, entropy_scatter, nmse_scatter] = pickle.load(f)

    for i in range(0+4, len(image_filenames), 5):
        # print(image_filenames[i])
        # for bit_depth in range(len(entropy_scatter[0])):
        plt.plot([bit_depth for bit_depth in range(1, 9)], [nmse_scatter[bit_depth][i*5] for bit_depth in range(1, 9)], label=image_filenames[i])

    plt.xlabel("Hologram bit depth")
    plt.ylabel("NMSE between reconstruction and target image")
    plt.legend(loc='upper right', ncol=5, fontsize=9)
    plt.show()

    # fig, ax = plt.subplots()
    # for i in entropy_scatter.keys():
    #     print(i)
    #     ax.scatter(entropy_scatter[i], nmse_scatter[i], label="hologram bit depth: " + str(i))
    #     print("bit depth: ", i, "\tmean: ", numpy.mean(nmse_scatter[i]), "\tstd", numpy.std(nmse_scatter[i]))
    # plt.xlabel("Target image entropy")
    # plt.ylabel("NMSE between reconstruction and target image")
    # ax.legend()
    # plt.show()
