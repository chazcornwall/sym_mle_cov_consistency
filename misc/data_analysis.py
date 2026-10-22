import numpy as np
import matplotlib.pyplot as plt

array = np.load("data/meas_close_20261022_111609_anees.npy")

plt.hist(array[0,0])
plt.show()