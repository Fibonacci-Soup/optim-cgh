import matplotlib.pyplot as plt
import pickle
NUM_ITERATIONS = 1000


with open('nmse_list_LBFGS_RE_all.pickle', 'rb') as handle:
    nmse_list_LBFGS_RE_all = pickle.load(handle)

for i, num_frames in enumerate([24, 12, 8, 6, 4, 3, 2, 1]):
    plt.plot(range(1, NUM_ITERATIONS + 1), nmse_list_LBFGS_RE_all[i], '-', label="Number_of_frames: {}".format(num_frames))

plt.xlabel("iterarions")
plt.ylabel("PSNR(dB)")
plt.legend()
plt.show()