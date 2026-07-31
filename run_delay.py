import numpy as np
from scipy.io import loadmat
from scipy import signal
from time import time
import matplotlib.pyplot as plt

import ei_kuramoto_delay as km

np.random.seed(335)

sc_data = loadmat("/home/elida/Desktop/stroke/fernando_stroke/data/raw/ANTONIO_20regiones_DET_PROB.mat")
C_raw, D_raw = sc_data["Cprob"], sc_data["Dprob"]
N = len(C_raw)
K_scalar,MD = 400,0.024

#matrices to load in the model, module sets inhibitory connections to zero
groups = np.zeros(N)
C_norm = km.normalize_connectivity_ref(C_raw)
K_used = km.build_K_matrix_ref(C_norm, K_scalar=K_scalar)
K_initial = K_used.copy()
D_base = km.build_delay_matrix_ref(D_raw, C_norm)

#sample frequencies and initial conditions
omega = km.build_natural_frequencies_ref(N, freq_mean_hz=40, freq_std_hz=2)
initial_phases = km.generate_initial_phases_ref(N)


T,dt,cutoff = 60,0.001,20000


print("Running!")
t1 = time()
mask = ~np.eye(N, dtype=bool)
target_mean, target_std = km.build_target_mean_std_per_node(K_used, groups)   # <- per-node, not global

phases_t, K_final, K_ee_std_t = km.run_kuramoto(
    omega, groups, K_used, T=T, dt=dt,
    D_base=D_base, MD=MD,
    initial_phases=initial_phases,
    plasticity_on=False,                 
    plasticity_rate=0.01, sharpness=2.0,
    target_mean=target_mean, target_std=target_std,   # (N,) arrays now, not scalars
    modulation_rate=0.5
)

runtime = time()-t1
print(f"Run!, time = {runtime:.3f} s")
#%%
phases_t = phases_t.T
ts = np.sin(phases_t[cutoff:])


fs = 1000
nperseg = fs * 5
noverlap = nperseg // 2

f, PSD = signal.welch(ts, fs, nperseg=nperseg, noverlap=noverlap,axis=0)


#%%

plt.figure(1)
plt.clf()

ax = plt.subplot(231)
ax.set_title(f"T={T:.2f} s, dt = {dt:.3f}\nruntime = {runtime:.3f} s")
ax.plot(f,PSD.mean(axis=1))
ax.set_xlim((-1,60))

ax = plt.subplot(234)
ax.set_title("initial C")
im = ax.imshow(K_initial)
plt.colorbar(im,ax=ax)

ax = plt.subplot(235)
ax.set_title("final C")
im = ax.imshow(K_final)
plt.colorbar(im,ax=ax)

ax = plt.subplot(236)
ax.set_title("final - initial")
im = ax.imshow(K_final - K_initial)
plt.colorbar(im,ax=ax)

plt.tight_layout()
plt.show()