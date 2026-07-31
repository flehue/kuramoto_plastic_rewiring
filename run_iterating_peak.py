# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 07:03:42 2026

@author: flehu
"""

import numpy as np
from scipy.io import loadmat
from scipy import signal
from time import time
import matplotlib.pyplot as plt
import os

import ei_kuramoto_delay as km

np.random.seed(335)


#%%
matrix_folder = r"C:\Users\flehu\OneDrive\Escritorio\postdocs\wael_post\fernando_stroke\plastic_rewiring\toy_model_elida"


sc_data = loadmat(os.path.join(matrix_folder,"matrices_syn_seed_1002_con ajuste.mat"))
C_raw, D_raw = sc_data["W_syn"], sc_data["D_syn"]

P_block = loadmat(os.path.join(matrix_folder,"P_block.mat"))
labels = P_block["Labels"].flatten()




#%%

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


#%%


T,dt,cutoff = 10,0.001,100


print("Running!")

target_mean, target_std = km.build_target_mean_std_per_node(K_used, groups)   # <- per-node, not global



#%%

fs = 1000
nperseg = fs * 5
noverlap = nperseg // 2


def observe_epoch(K):
    
    K_used = km.build_K_matrix_ref(C_norm, K_scalar=K)
    
    phases_t, K_final, K_ee_std_t = km.run_kuramoto(
        omega, groups, K_used, T=T, dt=dt,
        D_base=D_base, MD=MD,
        initial_phases=initial_phases,
        plasticity_on=False,                 
        plasticity_rate=0.01, sharpness=2.0,
        target_mean=target_mean, target_std=target_std,   # (N,) arrays now, not scalars
        modulation_rate=0.5
    )  
    phases_t = phases_t.T
    ts = np.sin(phases_t[cutoff:])

    f, PSD = signal.welch(ts, fs, nperseg=nperseg, noverlap=noverlap,axis=0)
    
    mean_peak = np.array([f[np.argmax(PSD[:,i])] for i in range(246)]).mean()

    return mean_peak
    

t1 = time()

def iterate_to_global(target,total_T,epoch_T,K_init=400,K_over_Hz = -1/0.05):
    N_epochs = total_T//epoch_T
    
    K = K_init
    
    print(f"target = {target:.2f} Hz")
    for n in range(N_epochs):
        if n==0:
            mean_peak = observe_epoch(K)
        else:
            difference_peak = mean_peak - target
            print(f"epoch {n}, mean_peak = {mean_peak:.2f}, K ={K:.2f}")
            K -= difference_peak*K_over_Hz
            mean_peak = observe_epoch(K)
    
    difference_peak = mean_peak - target
    print(f"finished {N_epochs} epochs,target= {target:.2f}, mean_peak = {mean_peak:.2f} diff = {difference_peak}, K = {K:.2f}")
            
        
iterate_to_global(15,20,2)
        
        
runtime = time()-t1
print(f"Run!, time = {runtime:.3f} s")
        


#%%


