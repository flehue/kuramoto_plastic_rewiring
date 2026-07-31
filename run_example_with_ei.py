import numpy as np
import minimal_EIkuramoto as kuramoto
import matplotlib.pyplot as plt

np.random.seed(123)

N = 100
T, dt = 50, 0.01
frac_exc = 0.8

# natural frequencies (E and I pools)
w_e_mean, w_e_std = 0.0, 0.1
w_i_mean, w_i_std = 0.0, 0.1

# coupling strengths (block means)
K_ee_mean, K_ee_std = 1.0, 0.1
K_ei_mean = 3.0
K_ie_mean = 3.0
K_ii_mean = 3.0

frequencies, groups = kuramoto.init_ei_population(N, w_e_mean, w_e_std, w_i_mean, w_i_std, frac_exc)
N_exc = int(np.sum(groups == 0))
K = kuramoto.build_K_matrix(groups, K_ee_mean, K_ee_std, K_ei_mean, K_ie_mean, K_ii_mean)

##lesion?
K[:,4] = 0
K[4,:] = 0
print(groups)

K_initial = K.copy()
K_ee_initial = K_initial[np.ix_(groups == 0, groups == 0)].copy()

phases_t, K_final, K_ee_std_t = kuramoto.run_kuramoto(
    frequencies, groups, K, T, dt,
    plasticity_rate=0.01, sharpness=2,
    target_mean=K_ee_mean / N_exc, target_std=K_ee_std / N_exc,
    modulation_rate=0.5
)

# Order parameter R(t) for the excitatory population: measures how
# synchronized the E units are (R=1 fully synced, R=0 fully desynced).
R_exc = np.abs(np.sum(np.exp(1j * phases_t[groups == 0, :]), axis=0) / N_exc)

t_axis = np.arange(0, T, dt)

#%%
plt.figure(1)
plt.clf()
# fig, axes = plt.subplots(1, 3, figsize=(14, 4),)

ax = plt.subplot(231)
ax.plot(t_axis, R_exc)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Time")
ax.set_ylabel("R (excitatory synchrony)")
ax.set_title("Order parameter over time")

ax = plt.subplot(232)
ax.plot(t_axis, K_ee_std_t)
ax.set_xlabel("Time")
ax.set_ylabel("Std(K_ee)")
ax.set_title("Spread of E-E weights over time")


ax = plt.subplot(233)
K_ee_final = K_final[np.ix_(groups == 0, groups == 0)].copy()
np.fill_diagonal(K_ee_final, np.nan)
ax.hist(K_ee_final.flatten(), bins=40)
ax.set_xlabel("K_ee (final)")
ax.set_ylabel("Count")
ax.set_title("Final K_ee distribution")

ax = plt.subplot(234)
ax.set_title("initial K")
im = ax.imshow(K_ee_initial, cmap="jet")
plt.colorbar(im, ax=ax)

ax = plt.subplot(235)
ax.set_title("final K")
im = ax.imshow(K_ee_final,cmap="jet")
plt.colorbar(im,ax=ax)



plt.tight_layout()
# plt.savefig("/mnt/user-data/outputs/minimal_EIkuramoto_result.png", dpi=150)
plt.show()