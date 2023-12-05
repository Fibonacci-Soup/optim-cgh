import numpy
import random
import matplotlib.pyplot as plt
import pickle
with open('variables_entropy_investigation.pkl', 'rb') as f:
    [image_filenames, entropy_scatter, nmse_scatter] = pickle.load(f)

    # for i in range(0, len(image_filenames)):
    #     print(image_filenames[i])
    #     # for bit_depth in range(len(entropy_scatter[0])):
    #     plt.plot([bit_depth for bit_depth in range(1, 9)], [nmse_scatter[bit_depth][i*5] for bit_depth in range(1, 9)], label=image_filenames[i] + " (entropy = {:.2f})".format(entropy_scatter[1][i*5]))

    # plt.xlabel("bit depth")
    # plt.ylabel("NMSE of reconstruction")
    # plt.legend(loc='upper right', ncol=3, fontsize=8)
    # plt.show()

    fig, ax = plt.subplots()
    for i in entropy_scatter.keys():
        print(i)
        ax.scatter(entropy_scatter[i], nmse_scatter[i], label="holo bit depth = " + str(i))
        print("bit depth: ", i, "\tmean: ", numpy.mean(nmse_scatter[i]), "\tstd", numpy.std(nmse_scatter[i]))
    plt.xlabel("Target image entropy")
    plt.ylabel("NMSE")
    ax.legend()
    plt.show()
