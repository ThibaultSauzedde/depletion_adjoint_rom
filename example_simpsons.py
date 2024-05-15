import matplotlib.pyplot as plt
import numpy as np
import copy
import scipy.linalg 
import os 
import seaborn as sns 
import time
from scipy import integrate

sns.set_theme()
SMALL_SIZE = 24
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
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


def getM(xs, phi):
    M = np.zeros((len(isot_list), len(isot_list)))
    M[0, 0] = -phi.dot(xs["U235"]["siga"])

    M[1, 1] = -phi.dot(xs["U238"]["siga"])

    M[2, 1] = +phi.dot(xs["U238"]["sigc"])
    M[2, 2] = -xs["U239"]["lambda"]

    M[3, 2] = xs["U239"]["lambda"]
    M[3, 3] = -xs["Np239"]["lambda"]

    M[4, 3] = xs["Np239"]["lambda"]
    M[4, 4] = -phi.dot(xs["Pu239"]["siga"])

    M[5, 4] = phi.dot(xs["Pu239"]["sigc"])
    M[5, 5] = -phi.dot(xs["Pu240"]["siga"])

    M[6, 5] = phi.dot(xs["Pu240"]["sigc"])
    M[6, 6] = -phi.dot(xs["Pu241"]["siga"]) - xs["Pu241"]["lambda"]

    M[7, 6] = phi.dot(xs["Pu241"]["sigc"])
    M[7, 7] = -phi.dot(xs["Pu242"]["siga"])

    M[8, 0] = xs["I135"]["gamma"] * phi.dot(xs["U235"]["sigf"])
    M[8, 4] = xs["I135"]["gamma"] * phi.dot(xs["Pu239"]["sigf"])
    M[8, 6] = xs["I135"]["gamma"] * phi.dot(xs["Pu241"]["sigf"])
    M[8, 8] = -xs["I135"]["lambda"]

    M[9, 8] = xs["I135"]["lambda"] 
    M[9, 9] = -phi.dot(xs["Xe135"]["siga"])-xs["Xe135"]["lambda"]

    M[10, 0] = xs["Nd149"]["gamma"] * phi.dot(xs["U235"]["sigf"])
    M[10, 4] = xs["Nd149"]["gamma"] * phi.dot(xs["Pu239"]["sigf"])
    M[10, 6] = xs["Nd149"]["gamma"] * phi.dot(xs["Pu241"]["sigf"])
    M[10, 10] = -xs["Nd149"]["lambda"]

    M[11, 10] = xs["Nd149"]["lambda"]
    M[11, 11] = -xs["Pm149"]["lambda"]

    M[12, 11] = xs["Pm149"]["lambda"]
    M[12, 12] = -phi.dot(xs["Xe135"]["siga"])  # todo: fix it, should be Sm149

    return M

def randomize_parameters(xs, phi, std = 1.):
    xs = copy.deepcopy(xs)
    phi= copy.deepcopy(phi)

    for isot, xs_i in xs.items():
        if "siga" in xs_i.keys():    
            xs[isot]["siga"][0] = xs[isot]["siga"][0] * (1 + np.random.randn()*std/100.)
            xs[isot]["siga"][1] = xs[isot]["siga"][1] * (1 + np.random.randn()*std/100.)

        if "sigf" in xs_i.keys():
            xs[isot]["sigf"][0] = xs[isot]["sigf"][0] * (1 + np.random.randn()*std/100.)
            xs[isot]["sigf"][1] = xs[isot]["sigf"][1] * (1 + np.random.randn()*std/100.)
        
        if "sigc" in xs_i.keys():
            xs[isot]["sigc"][0] = xs[isot]["sigc"][0] * (1 + np.random.randn()*std/100.)
            xs[isot]["sigc"][1] = xs[isot]["sigc"][1] * (1 + np.random.randn()*std/100.)
        
        if "gamma" in xs_i.keys():
            xs[isot]["gamma"] = xs[isot]["gamma"] * (1 + np.random.randn()*std/100.)
        
        if "lambda" in xs_i.keys():
            xs[isot]["lambda"] = xs[isot]["lambda"] * (1 + np.random.randn()*std/100.)
        
        if "n0" in xs_i.keys():
            xs[isot]["n0"] = xs[isot]["n0"] * (1 + np.random.randn()*std/100.)
    
    phi[0] = phi[0] * (1 + np.random.randn()*std/100.)
    phi[1] = phi[1] * (1 + np.random.randn()*std/100.)

    return xs, phi


xs = {}

xs["U235"] = {}
xs["U235"]["siga"] = [1.8, 100.]
xs["U235"]["sigf"] = [1.5, 55.]
xs["U235"]["n0"] = 3.5 * 1e23 

xs["U238"] = {}
xs["U238"]["siga"] = [0.35, 2.]
xs["U238"]["n0"] = 96.5 * 1e23

xs["U239"] = {}
xs["U239"]["lambda"] = 7.25 * 1e-4 * np.sqrt(2)
xs["U239"]["n0"] = 0.

xs["Np239"] = {}
xs["Np239"]["lambda"] = 5.03 * 1e-6 * np.sqrt(2)
xs["Np239"]["n0"] = 0.

xs["Pu239"] = {}
xs["Pu239"]["siga"] = [2.0, 190.]
xs["Pu239"]["sigf"] = [1.8, 120.]
xs["Pu239"]["n0"] = 0.

xs["Pu240"] = {}
xs["Pu240"]["siga"] = [0.2, 110.]
xs["Pu240"]["n0"] = 0.

xs["Pu241"] = {}
xs["Pu241"]["siga"] = [2.5, 180.]
xs["Pu241"]["sigf"] = [0.5, 140.]
xs["Pu241"]["lambda"] = 2.11 *1e-9 * np.sqrt(2)
xs["Pu241"]["n0"] = 0.

xs["Pu242"] = {}
xs["Pu242"]["siga"] = [0.7, 70.]
xs["Pu242"]["n0"] = 0.

xs["I135"] = {}
xs["I135"]["gamma"] = 0.064
xs["I135"]["lambda"] = 4.25 *1e-5 * np.sqrt(2)
xs["I135"]["n0"] = 0.

xs["Xe135"] = {}
xs["Xe135"]["siga"] = [0.0, 2. * 1e5]
xs["Xe135"]["lambda"] = 3.03 *1e-5 * np.sqrt(2)
xs["Xe135"]["n0"] = 0.

xs["Nd149"] = {}
xs["Nd149"]["gamma"] = 0.0109
xs["Nd149"]["lambda"] = 1.61 *1e-4 * np.sqrt(2)
xs["Nd149"]["n0"] = 0.

xs["Pm149"] = {}
xs["Pm149"]["lambda"] = 5.24 *1e-6 * np.sqrt(2)
xs["Pm149"]["n0"] = 0.

xs["Sm149"] = {}
xs["Sm149"]["siga"] = [0.0, 1e3]
xs["Sm149"]["n0"] = 0.
print(xs)
# import ipdb; ipdb.set_trace()
for isot, xs_i in xs.items():

    if "siga" in xs_i.keys():
        xs[isot]["siga"] =np.array(xs[isot]["siga"])*1e-24

        if "sigf" in xs_i.keys():
            xs[isot]["sigf"] =np.array(xs[isot]["sigf"])*1e-24
            xs[isot]["sigc"] = xs[isot]["siga"] - xs[isot]["sigf"]
        else:
            xs[isot]["sigc"] = xs[isot]["siga"]


t0 = 0.
tf = 5*24*60*60
dt = 60*60*20
# t = np.arange(t0, tf, dt)
t = np.linspace(t0, tf, int(tf/dt))
# print(t)
# print(5*24*60*60)


isot_list = ['U235', 'U238', 'U239', 'Np239', 'Pu239', 'Pu240', 'Pu241', 'Pu242', 'I135', 'Xe135', 'Nd149', 'Pm149', 'Sm149'] #list(xs.keys())

n0 = np.zeros(len(isot_list))
for i, isot in enumerate(isot_list):
    n0[i] = xs[isot]["n0"]


phi = np.array([0.6667 *1e14, 0.2 *1e15])
M = getM(xs, phi)
# import ipdb; ipdb.set_trace()
n = np.zeros((len(isot_list), len(t)))
# print(M.shape)   
# print(n0.shape) 
for i, t_i in enumerate(t):
    n[:, i] = scipy.linalg.expm(M*(t_i-t0)).dot(n0)

# for i, isot in enumerate(isot_list):
#     fig, ax = plt.subplots()
#     ax.plot(t/(24*60*60), n[i], label=isot)
#     ax.legend()
# plt.show()


#test n_t calculation 
# qi = np.zeros(len(isot_list))
# qi[5] = 1.
# nt_j = np.zeros((len(isot_list), len(t)))
# for k, t_k in enumerate(t):
#     nt_j[:, k] = scipy.linalg.expm(-(M.T)*(t_k-tf))@qi
#     print(nt_j[:, k].dot(n[:, k]))

# for i, isot in enumerate(isot_list):
#     fig, ax = plt.subplots()
#     ax.plot(t/(24*60*60), nt_j[i], label=isot)
#     ax.legend()
# plt.show()

# Calculation of the reduced basis with a range finding algorithm 
# calculate the snapshots 

s_size = 13
trials_size = 10
Snap = np.zeros((s_size, len(isot_list))) #todo: mettre des deltas plutôt !
trials = []
for i in range(s_size+trials_size):
    xs_i, phi_i = randomize_parameters(xs, phi)
    n0_i = np.zeros(len(isot_list))
    for j, isot in enumerate(isot_list):
        n0_i[j] = xs_i[isot]["n0"]

    M_i = getM(xs_i, phi_i)

    nf_i =  scipy.linalg.expm(M_i*(tf-t0)).dot(n0_i)
    if i < s_size:
        Snap[i] = nf_i
    else:
        trials.append(nf_i)

if os.path.exists("basis"):
    npzfile = np.load("basis.npz")
    Snap = npzfile["Snap"]
    trials = npzfile["trials"]
else:
    np.savez("basis", Snap, trials)


#normalize the concentrations by the max for each isotopes 
# mean_coeff = []
# var_coeff = []
# for i in range(len(isot_list)):
#     mean_coeff.append(np.mean(Snap[:, i]))
#     var_coeff.append(np.std(Snap[:, i]))
#     Snap[:, i] = (Snap[:, i] - mean_coeff[-1]) / var_coeff[-1]
    # print(np.mean(Snap[:, i]), np.var(Snap[:, i]), np.std(Snap[:, i]))

U, s, Vh = scipy.linalg.svd(Snap, full_matrices=False)
if s_size <= len(isot_list):
    Q = Vh
else:
    Q = U

# for i, q_i in enumerate(Q):
#     for j, q_j in enumerate(Q):
#         print(i, j, q_i.dot(q_j))

# for i in range(len(isot_list)):
#     Q[:, i] = Q[:, i] * var_coeff[i] + mean_coeff[i]


prec = []
for i in range(trials_size):
    trials[i] = trials[i] / np.linalg.norm(trials[i])  

trials_res = copy.deepcopy(trials)
trials_norm = np.zeros(trials_size)
precisions = np.zeros(s_size)
for k in range(s_size):
    for i in range(trials_size):
        trials_res[i] -= Q[k].dot(trials[i]) * Q[k]
        trials_norm[i] = np.linalg.norm(trials_res[i])  
    precisions[k] = (10 * np.sqrt(2 / np.pi)) * np.max(trials_norm)

print("precisions = ", precisions)
fig, ax = plt.subplots(figsize=(8, 6.7))
ax.scatter(range(1, len(precisions)+1), precisions)
ax.set_xlabel("Size of the basis")
ax.set_ylabel("Theoretical error ($cm^{-3}$)", rotation='horizontal', y=1.05, labelpad=-130)
ax.set_yscale('log')
ax.xaxis.set_ticks(range(1, len(precisions)+1))
fig.tight_layout()
fig.savefig(f"./basis_precision.png", dpi=600)
# We try this basis of two vectors ! 

Q_ = Q #Q[:12, :]
nt_rom = []
for qi in Q_: 
    nt_i = np.zeros((len(isot_list), len(t)))
    for i, t_i in enumerate(t):
        nt_i[:, i] = scipy.linalg.expm(-M.T*(t_i-tf))@qi
    nt_rom.append(nt_i)

nt_dpt = []
for j in range(len(isot_list)):
    qi = np.zeros(len(isot_list))
    qi[j] = 1.
    nt_j = np.zeros((len(isot_list), len(t)))
    for k, t_k in enumerate(t):
        nt_j[:, k] = scipy.linalg.expm(-M.T*(t_k-tf))@qi
    nt_dpt.append(nt_j)

#recons rom
nb_test = 100

perturb = np.zeros((len(isot_list), nb_test))
errors_rom = np.zeros((len(isot_list), nb_test))
errors_dpt = np.zeros((len(isot_list), nb_test))
errors_rom = np.zeros((len(isot_list), nb_test))
errors_rom_8 = np.zeros((len(isot_list), nb_test))

times_exact = np.zeros((len(isot_list), nb_test))
times_rom = np.zeros((len(isot_list), nb_test))
times_rom8 = np.zeros((len(isot_list), nb_test))
times_dpt = np.zeros((len(isot_list), nb_test))

for i in range(nb_test):
    xs_i, phi_i = randomize_parameters(xs, phi, std=1.)
    n0_i = np.zeros(len(isot_list))
    for j, isot in enumerate(isot_list):
        # n0_i[j] = xs[isot]["n0"]
        n0_i[j] = xs_i[isot]["n0"]

    M_i = getM(xs_i, phi_i)

    time0 =time.time()

    nf_i =  scipy.linalg.expm(M_i*(tf-t0)).dot(n0_i)

    delta_time0 = time.time()-time0
    times_exact[:, i] = delta_time0

    dnf = nf_i - n[:, -1]
    
    dn0 = n0_i - n0
    dM = M_i - M

    # using the rom ! 
    time0 =time.time()

    a = []
    dnf_rom = np.zeros(len(isot_list))
    for j, qi in enumerate(Q_):
        ntdMn = np.zeros(nt_rom[j].shape[1])
        for k in range(len(ntdMn)):
            ntdMn[k] = nt_rom[j][:, k].dot(dM.dot(n[:, k]))

        int_ntdMn = integrate.simps(ntdMn, t)
        a.append(int_ntdMn + nt_rom[j][:, 0].dot(dn0))
        dnf_rom += a[-1] * qi
    
    nf_rom = n[:, -1] + dnf_rom

    delta_time0 = time.time()-time0
    times_rom[:, i] = delta_time0

    # using the rom ! 
    time0 =time.time()
    a = []
    dnf_rom = np.zeros(len(isot_list))
    for j, qi in enumerate(Q_[:8, :]):
        ntdMn = np.zeros(nt_rom[j].shape[1])
        for k in range(len(ntdMn)):
            ntdMn[k] = nt_rom[j][:, k].dot(dM.dot(n[:, k]))

        int_ntdMn = integrate.simps(ntdMn, t)
        a.append(int_ntdMn + nt_rom[j][:, 0].dot(dn0))
        dnf_rom += a[-1] * qi
    
    nf_rom_8 = n[:, -1] + dnf_rom

    delta_time0 = time.time()-time0
    times_rom8[:, i] = delta_time0

    #using dpt
    time0 =time.time()

    dnf_dpt = np.zeros(len(isot_list))
    for j in range(len(isot_list)):
        ntdMn = np.zeros(nt_dpt[j].shape[1])
        for k in range(len(ntdMn)):
            ntdMn[k] = nt_dpt[j][:, k].dot(dM.dot(n[:, k]))
        
        
        # if j == 4:
        #     # import ipdb; ipdb.set_trace()
        #     fig, ax = plt.subplots(3, 1,  figsize=(8.9, 4.5*3), sharex=True)
        #     ax[0].plot(t/(24*60*60), n[4], label=r'$^{239}$Pu')
        #     ax[0].set_ylabel(r"$n_i(t)$ ($cm^{-3}$)", fontsize=MEDIUM_SIZE)
        #     ax[0].legend()
        #     ax[1].plot(t/(24*60*60), nt_dpt[4][1], label=r'$^{238}$U')
        #     ax[1].plot(t/(24*60*60), nt_dpt[4][2], label=r'$^{239}$U')
        #     ax[1].plot(t/(24*60*60), nt_dpt[4][3], label=r'$^{239}$Np')
        #     ax[1].plot(t/(24*60*60), nt_dpt[4][4], label=r'$^{239}$Pu')
        #     ax[1].set_ylabel(r"$n_i^{\dagger}(t)$ ($cm^{-3}$)",  fontsize=MEDIUM_SIZE)
        #     ax[1].legend()
        #     ax[2].plot(t/(24*60*60), ntdMn)
        #     ax[2].set_ylabel(r"$n_i^\dagger(t) \delta M n(t)$", fontsize=MEDIUM_SIZE)
        #     ax[2].set_xlabel("time (days)", fontsize=MEDIUM_SIZE)
        #     ax[2].fill_between(t/(24*60*60), 0., ntdMn, alpha=0.5)
        #     ax[2].text(2.5, np.min(ntdMn) + (np.max(ntdMn) - np.min(ntdMn))/2., r"$\int_{t_0}^{t_f} n_i^\dagger(t) \delta M n(t) dt$",
        # horizontalalignment='center', fontsize=20)
        #     fig.tight_layout()
        #     fig.savefig(f"./ni_ni_dagger/ni_ni_dagger_{i}.png", dpi=600)
        #     # import ipdb; ipdb.set_trace()
            
        int_ntdMn = integrate.simps(ntdMn, t)
        dnf_dpt[j] = (int_ntdMn + nt_dpt[j][:, 0].dot(dn0))
    
    nf_dpt = n[:, -1] + dnf_dpt

    delta_time0 = time.time()-time0
    times_dpt[:, i] = delta_time0

    print("test ", i)

    # print("dnf_rom = ", dnf_rom)
    # print("nf_rom-nf / nf (%) = ", 100* (nf_rom-nf_i)/nf_i)

    # print("dnf_dpt = ", dnf_dpt)
    # print("nf_dpt-nf / nf (%) = ", 100* (nf_dpt-nf_i)/nf_i)

    perturb[:, i] = 100* (n[:, -1]-nf_i)/nf_i
    errors_rom[:, i] =  100* (nf_rom-nf_i)/nf_i
    errors_rom_8[:, i] =  100* (nf_rom_8-nf_i)/nf_i
    errors_dpt[:, i] =  100* (nf_dpt-nf_i)/nf_i

print("time")
print("dpt : ", times_dpt.mean())
print("rom : ", times_rom.mean())
print("rom8 : ", times_rom8.mean())
print("exact : ", times_exact.mean())

print("max pert")
print("dpt : ", errors_dpt.max(axis=1))
print("rom : ", errors_rom.max(axis=1))
print("rom8 : ", errors_rom_8.max(axis=1))
print("exact : ", perturb.max(axis=1))

np.savez(f"times_{len(t)}", len(t), errors_dpt.max(axis=1), errors_rom.max(axis=1), errors_rom_8.max(axis=1), perturb.max(axis=1),
                        times_dpt.mean(), times_rom.mean(), times_rom8.mean(), times_exact.mean())


import ipdb; ipdb.set_trace()
np.savez("errors", errors_rom, errors_rom_8, errors_dpt, isot_list)
