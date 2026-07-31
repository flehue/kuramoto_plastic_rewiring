import math
import numpy as np
from numba import njit


def normalize_connectivity_ref(C_raw):
    """
    Matches Kuramoto.load_struct_connectivity()'s normalization:
    off-diagonal mean forced to 1, diagonal forced to 0.
    (This is also exactly what `C[mask] /= C[mask].mean()` does in your
    own analysis script -- this is just that same step, reusable.)
    """
    N = C_raw.shape[0]
    C = C_raw.copy().astype(float)
    mask = ~np.eye(N, dtype=bool)
    C[mask] /= C[mask].mean()
    np.fill_diagonal(C, 0.0)
    return C


def build_K_matrix_ref(C_normalized, K_scalar):
    """
    Matches: global_coupling = K_scalar / N; K_used = global_coupling * C.
    THIS is the fix for the K*C bug -- the reference model always
    divides by N, it never uses raw K directly.

    C_normalized must already be normalized (see normalize_connectivity_ref
    -- call that FIRST). Returns the matrix ready to pass as `K` into
    run_kuramoto.
    """
    N = C_normalized.shape[0]
    global_coupling = K_scalar / N
    return global_coupling * C_normalized


def build_delay_matrix_ref(D_raw, C_normalized):
    """
    Matches applyMean_Delay(): normalizes D_raw so its mean over
    CONNECTED edges (C_normalized != 0) equals exactly 1. Pass this as
    D_base into run_kuramoto together with MD = your desired mean delay
    in seconds
    """
    connected = C_normalized != 0
    D = D_raw.copy().astype(float)
    D = D / D[connected].mean()
    np.fill_diagonal(D, 0.0)
    return D


def build_natural_frequencies_ref(N, freq_mean_hz, freq_std_hz):
    freqs_hz = np.random.normal(freq_mean_hz, freq_std_hz, size=N)
    return 2 * np.pi * freqs_hz


def generate_initial_phases_ref(N):
    return np.random.uniform(0, 2 * np.pi, size=N)


def init_ei_population(N, w_e_mean, w_e_std, w_i_mean, w_i_std, frac_exc=0.8):
    """Split N units into Excitatory (0) / Inhibitory (1) groups and
    draw a natural frequency for each unit from a group-specific Gaussian."""
    N_exc = int(frac_exc * N)
    N_inh = N - N_exc
    groups = np.concatenate([np.zeros(N_exc), np.ones(N_inh)])
    np.random.shuffle(groups)

    frequencies = np.random.normal(w_e_mean, w_e_std, size=N)
    frequencies[groups == 1] = np.random.normal(w_i_mean, w_i_std, size=N_inh)
    return frequencies, groups


# ---------------------------------------------------------------------
# 2. Build the coupling matrix K (one value per E/I block)
#    Also setup-only, plain Python/NumPy.
# ---------------------------------------------------------------------
def build_K_matrix(groups, K_ee_mean, K_ee_std, K_ei_mean, K_ie_mean, K_ii_mean):
    """K[i, j] = influence of unit j on unit i.
    Excitatory-to-* connections are positive, inhibitory-to-* are negative.
    Each block is normalized by the size of the presynaptic group so that
    total incoming drive stays roughly O(1) regardless of N."""
    N = len(groups)
    N_exc = int(np.sum(groups == 0))
    N_inh = int(np.sum(groups == 1))
    K = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if groups[i] == 0 and groups[j] == 0:      # E <- E
                K[i, j] = np.random.normal(K_ee_mean, K_ee_std) / N_exc
            elif groups[i] == 1 and groups[j] == 0:     # I <- E
                K[i, j] = np.random.normal(K_ei_mean, 0) / N_exc
            elif groups[i] == 0 and groups[j] == 1:     # E <- I  (inhibitory: negative)
                K[i, j] = -np.random.normal(K_ie_mean, 0) / N_inh
            elif groups[i] == 1 and groups[j] == 1:     # I <- I  (inhibitory: negative)
                K[i, j] = -np.random.normal(K_ii_mean, 0) / N_inh
    return K


# ---------------------------------------------------------------------
# 2b. Delay matrix for E-E connections ONLY
#     IMPORTANT SHAPE CHANGE: returns a FULL (N, N) matrix now (zero
#     outside the E-E block), not a compact (N_exc, N_exc) one.
#     Numba's nopython mode can't do the "gather" indexing needed to
#     work with a compact submatrix cleanly, so everything in this file
#     now stays at full (N, N) size and relies on `groups` checks
#     inside plain loops instead of fancy indexing.
#     (If you already run everyone as "E" -- e.g. groups = np.zeros(N)
#     for a pure structural-connectome run -- this is no different from
#     before: N_exc == N, so full and compact are the same thing.)
# ---------------------------------------------------------------------
def build_delay_matrix_ee(groups, delay_mean, delay_std):
    """
    Build a BASE delay matrix for excitatory-to-excitatory (E-E)
    connections only. Returned as a full (N, N) matrix: entries outside
    the E-E block are exactly zero and are never used as a delay by
    run_kuramoto (E-I, I-E, I-I are always instantaneous, unconditionally).

    Units = time (same units you use for T and dt). This is the BASE
    structure (as if MD = 1); run_kuramoto rescales it internally as
    D_used = MD * D_base.

    If you already have a real delay matrix (e.g. distance/tractography
    based), just build a full (N, N) array yourself with zeros outside
    the E-E block (or all real values, if everyone is "E") and pass
    that in directly instead of using this function.
    """
    N = len(groups)
    D_full = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if groups[i] == 0 and groups[j] == 0:
                D_full[i, j] = abs(np.random.normal(delay_mean, delay_std))
    D_full = (D_full + D_full.T) / 2   # symmetric delay structure
    np.fill_diagonal(D_full, 0.0)
    return D_full


# ---------------------------------------------------------------------
# 3. Hebbian plasticity rule for K_ee only (Numba-jitted)
# ---------------------------------------------------------------------
@njit(cache=True)
def hebbian_update_Kee(K, groups, phase_diff_ee, plasticity_rate, sharpness):
    """Strengthen E-E connections between units whose relevant phase
    difference is small: gain = exp(-sharpness * |phase difference|).
    Only entries where BOTH i and j are excitatory are ever touched.

    phase_diff_ee : (N, N) array of wrapped |phase differences|,
        precomputed by run_kuramoto (using delayed presynaptic phases
        where a delay is in effect). Entries outside the E-E block are
        never read.
    """
    N = K.shape[0]
    for i in range(N):
        if groups[i] != 0:
            continue
        for j in range(N):
            if groups[j] != 0 or i == j:
                continue
            gain = math.exp(-sharpness * phase_diff_ee[i, j])
            K[i, j] += plasticity_rate * gain
    return K

def build_target_mean_std_per_node(K_initial, groups):
    N = K_initial.shape[0]
    target_mean = np.zeros(N)
    target_std = np.zeros(N)
    for i in range(N):
        if groups[i] != 0:
            continue
        row_idx = [j for j in range(N) if groups[j] == 0 and j != i]
        row_vals = K_initial[i, row_idx]
        target_mean[i] = row_vals.mean()
        target_std[i] = row_vals.std()
    return target_mean, target_std

# ---------------------------------------------------------------------
# 3b. Homeostatic normalization for K_ee only (Numba-jitted)
# ---------------------------------------------------------------------
@njit(cache=True)
def homeostatic_normalize_Kee(K, groups, target_mean, target_std, modulation_rate):
    """Pull each row of K_ee back toward ITS OWN (target_mean[i],
    target_std[i]), at a limited rate (modulation_rate). Only the E-E
    block is touched.
 
    target_mean, target_std : (N,) arrays -- ONE TARGET PER NODE
    """
    N = K.shape[0]
    for i in range(N):
        if groups[i] != 0:
            continue
 
        row_sum = 0.0
        row_count = 0
        for j in range(N):
            if groups[j] != 0 or i == j:
                continue
            row_sum += K[i, j]
            row_count += 1
        if row_count == 0:
            continue
        row_mean = row_sum / row_count
 
        var_sum = 0.0
        for j in range(N):
            if groups[j] != 0 or i == j:
                continue
            var_sum += (K[i, j] - row_mean) ** 2
        row_std = math.sqrt(var_sum / row_count)
        if row_std == 0.0:
            continue
 
        for j in range(N):
            if groups[j] != 0 or i == j:
                continue
            K_target = (K[i, j] - row_mean) * (target_std[i] / row_std) + target_mean[i]
            K[i, j] += modulation_rate * (K_target - K[i, j])
    return K


# ---------------------------------------------------------------------
# 4. Kuramoto integration loop (Numba-jitted, E-E-only delay,
#    easy plasticity on/off switch)
# ---------------------------------------------------------------------
@njit(cache=True)
def run_kuramoto(frequencies, groups, K, T, dt,
                  D_base, MD,
                  initial_phases,
                  plasticity_on,
                  plasticity_rate, sharpness,
                  target_mean, target_std, modulation_rate):
    """
    Euler integration of the Kuramoto phase equations. Fully Numba
    nopython-compatible: everything is explicit loops over plain
    scalar-indexed arrays -- no np.ix_, no fancy/gather indexing, no
    None defaults (Numba needs one concrete type per variable).

    initial_phases : (N,) float64 array, REQUIRED.
        Phases at t=0. Generate these with generate_initial_phases_ref(N)
        (or your own array) BEFORE calling this function, and control
        their seed with an ordinary np.random.seed() call beforehand.
        Random phases are deliberately NOT generated inside this
        function: Numba keeps a completely separate internal RNG state
        from NumPy's global one, so np.random.seed() in your calling
        code would have had NO effect on phases generated in here --
        silently breaking reproducibility. Passing them in externally
        also matches the reference class exactly (its initial_phases()
        is plain NumPy too, not inside anything compiled).

    Delay handling
    --------------
    D_base : (N, N) float64 array.
        Only entries where BOTH i and j are excitatory are ever used as
        a delay -- everything else is ignored outright, regardless of
        its value. Pass an all-zero (N, N) array for "no delay at all".
    MD : float64.
        Delay actually used = MD * D_base[i, j]. Set MD = 0.0 to force
        E-E instantaneous even with a nonzero D_base, or scale it up to
        stretch every E-E delay proportionally.
    E-I, I-E, I-I connections are ALWAYS instantaneous, unconditionally.

    Plasticity switch
    -----------------
    plasticity_on : bool.
        True  -> Hebbian update + homeostatic normalization run on K_ee
                 every step, as before.
        False -> K is never modified, at all, for the whole simulation.
                 target_mean / target_std are unused in this case --
                 pass 0.0 as a placeholder, Numba still needs a float.
    """
    N = frequencies.shape[0]
    n_steps = int(T / dt)

    # ---- delay, in integer steps of dt (E-E only; 0 elsewhere) ----
    delay_steps = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            d = int(round(MD * D_base[i, j] / dt))
            if d < 0:
                d = 0
            delay_steps[i, j] = d

    #initial phases
    phases = initial_phases.copy()

    phases_t = np.zeros((N, n_steps))
    for i in range(N):
        phases_t[i, 0] = phases[i]

    K_ee_std_t = np.zeros(n_steps)

    for t in range(1, n_steps):
        t_prev = t - 1   # `phases` currently holds phases_t[:, t_prev]

        sin_sum = np.zeros(N)
        phase_diff_ee = np.zeros((N, N))   # only E-E

        for i in range(N):
            theta_i = phases[i]
            for j in range(N):
                if i == j:
                    continue

                if groups[i] == 0 and groups[j] == 0:
                    # E-E: possibly-delayed lookup into stored history
                    tt = t_prev - delay_steps[i, j]
                    if tt < 0:
                        tt = 0
                    theta_j = phases_t[j, tt]
                    diff = theta_i - theta_j
                    phase_diff_ee[i, j] = abs(math.atan2(math.sin(diff), math.cos(diff)))
                else:
                    # E-I / I-E / I-I: always instantaneous
                    theta_j = phases[j]

                sin_sum[i] += K[i, j] * math.sin(theta_j - theta_i)

        #integrate phases
        for i in range(N):
            phases[i] = (phases[i] + dt * (frequencies[i] + sin_sum[i])) % (2 * np.pi)
            phases_t[i, t] = phases[i]

        #plasticity on K_ee
        if plasticity_on:
            K = hebbian_update_Kee(K, groups, phase_diff_ee, plasticity_rate, sharpness)
            K = homeostatic_normalize_Kee(K, groups, target_mean, target_std, modulation_rate)

        # ---- track spread of K_ee this step ----
        ee_sum = 0.0
        ee_sq = 0.0
        ee_n = 0
        for i in range(N):
            if groups[i] != 0:
                continue
            for j in range(N):
                if groups[j] != 0 or i == j:
                    continue
                ee_sum += K[i, j]
                ee_sq += K[i, j] * K[i, j]
                ee_n += 1
        if ee_n > 0:
            m = ee_sum / ee_n
            var = ee_sq / ee_n - m * m
            if var < 0.0:
                var = 0.0
            K_ee_std_t[t] = math.sqrt(var)

    return phases_t, K, K_ee_std_t
