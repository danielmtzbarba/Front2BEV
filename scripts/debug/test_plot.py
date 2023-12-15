import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from pathlib import Path
from dan.utils import load_pkl_file

logs_path = Path('D:/Logs/Dan-2023-Front2BEV/')

log_file = load_pkl_file(str(logs_path / 'F2B_VAE_3K.pkl'))

ydata = np.array(log_file['batches']['loss'])
xdata = np.array([i for i in range(len(ydata))])

# Plot the actual data
plt.plot(xdata, ydata, ".", label="Data");

# This is the function we are trying to fit to the data.
def func(x, a, b, c):
     return a * np.exp(-b * x) + c

# The actual curve fitting happens here
optimizedParameters, pcov = curve_fit(func, xdata, ydata);

# Use the optimized parameters to plot the best fit
plt.plot(xdata, func(xdata, *optimizedParameters), label="fit");

# Show the graph
plt.legend();
plt.show();
