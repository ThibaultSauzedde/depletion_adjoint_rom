import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
import pandas as pd 

sns.set_theme()
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex=True)

plt.rc('mathtext', fontset='custom')
plt.rc('mathtext', rm='Helvetica')
plt.rc('mathtext', it='Helvetica:italic')
plt.rc('mathtext', bf='Helvetica:bold')
plt.rc('font', family="Helvetica")

plt.rc('text.latex', preamble=r'\usepackage{xfrac}')

npzfile = np.load("errors_final_weighted.npz")
print(npzfile)

errors_rom = npzfile["arr_0"]
errors_rom_8 = npzfile["arr_1"]
errors_dpt = npzfile["arr_2"]
isot_list = npzfile["arr_3"]

errors_rom = pd.DataFrame(errors_rom, index=isot_list)
errors_rom_8 = pd.DataFrame(errors_rom_8, index=isot_list)
errors_dpt = pd.DataFrame(errors_dpt, index=isot_list)
rom_name = """Adjoint Based Reduced
Order Model (size = 13)"""
dpt_name = "Depletion Perturbation Theory"
value_keys = "values"
values_name = r"$\sfrac{n_i-n_i^{ref}}{n_i^{ref}} \; (\%)$ "
data = pd.concat([errors_rom, errors_dpt], keys=[rom_name, dpt_name])

data.columns.name = "error"
data.index.names = ["Model", "Isotopes"]
data = data.stack()
data = data.reset_index()
data.columns = ["Reconstruction model", 'Isotopes', 'error', value_keys]
fig, ax = plt.subplots(figsize=(8, 6.7))
sns.boxplot(data=data.reset_index(), y="Isotopes", x=value_keys,
             hue="Reconstruction model", ax=ax)
ax.set_ylabel(
    "Isotopes",
    rotation='horizontal', y=1.02, labelpad=-40, fontsize=MEDIUM_SIZE)
ax.set_xlabel(values_name, fontsize=BIGGER_SIZE)
ax.legend(title=r"\underline{Reconstruction model}", title_fontsize=SMALL_SIZE,
framealpha=1., edgecolor=(0,0,0))
isot_names = [r'$^{235}$U', r'$^{238}$U', r'$^{239}$U', r'$^{239}$Np',
               r'$^{239}$Pu', r'$^{240}$Pu', r'$^{241}$Pu', r'$^{242}$Pu', r'$^{135}$I',
 r'$^{135}$Xe', r'$^{149}$Nd', r'$^{149}$Pm', r'$^{149}$Sm']

plt.yticks(range(len(isot_list)), isot_names)
fig.tight_layout()
fig.savefig(f"./error_final_weighted.png", dpi=600)

rom_name = """Adjoint Based Reduced
Order Model (size = 8)"""
dpt_name = "Depletion Perturbation Theory"
# data = pd.concat([errors_rom_8, errors_dpt], keys=[rom_name, dpt_name])
data = pd.concat([errors_rom_8], keys=[rom_name])

data.columns.name = "error"
data.index.names = ["Model", "Isotopes"]
data = data.stack()
data = data.reset_index()
data.columns = ["Reconstruction model", 'Isotopes', 'error', value_keys]
fig, ax = plt.subplots(figsize=(8, 6.7))
sns.boxplot(data=data.reset_index(), y="Isotopes", x=value_keys,
             hue="Reconstruction model",  ax=ax)
ax.set_ylabel(
    "Isotopes",
    rotation='horizontal', y=1.02, labelpad=-40, fontsize=MEDIUM_SIZE)
ax.set_xlabel(values_name, fontsize=BIGGER_SIZE)
ax.legend(title=r"\underline{Reconstruction model}",
           title_fontsize=SMALL_SIZE, loc='best', bbox_to_anchor=(0.5,1.1), 
framealpha=1., edgecolor=(0,0,0))
isot_names = [r'$^{235}$U', r'$^{238}$U', r'$^{239}$U', r'$^{239}$Np',
               r'$^{239}$Pu', r'$^{240}$Pu', r'$^{241}$Pu', r'$^{242}$Pu', r'$^{135}$I',
 r'$^{135}$Xe', r'$^{149}$Nd', r'$^{149}$Pm', r'$^{149}$Sm']

plt.yticks(range(len(isot_list)), isot_names)
fig.tight_layout()
fig.savefig(f"./error_rom8_final_weighted.png", dpi=600)