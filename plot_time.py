import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import pandas as pd 

errors_dpt = []
errors_rom_8 = []
times_dpt = []
times_rom8 = []
times_exact = []
for length in [6, 12, 120, 1200]:
    npzfile = np.load(f"times_{length}.npz")
    errors_dpt.append(npzfile["arr_1"])
    errors_rom_8.append(npzfile["arr_3"])
    times_dpt.append(npzfile["arr_5"])
    times_rom8.append(npzfile["arr_7"])
    times_exact.append(npzfile["arr_8"])
    print(errors_dpt[-1])

print(100*(np.array(times_rom8)-np.array(times_exact))/np.array(times_exact))


import ipdb; ipdb.set_trace()

