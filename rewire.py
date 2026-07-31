folder = "toy_model_elida/"
import sys
sys.path.append(folder)
import Toy_model_v_final_okok as toy_model
import numpy as np

def lesion_matrix(C,i):
    
    out = C.copy()
    out[i] = 0
    out[:,i] = 0
    return out



def new_weight(weight, target_row, mode="same"):
    """
    ============================================================
    WEIGHT INITIALIZATION — edit this function to explore rules
    ============================================================
    Determines the weight assigned to a new rewired connection.

    Parameters
    ----------
    weight : float
        Original weight of the lost connection.
    target_row : np.ndarray
        Current row (or col) of the target node in rewired_C,
        used for statistics-based rules.
    mode : str
        "same"  → keep the original lost weight (default)
        "mean"  → mean of existing non-zero weights in target row
        "min"   → minimum of existing non-zero weights in target row
        "max"   → maximum of existing non-zero weights in target row
        "noise" → small random weight (mean of target row + Gaussian noise)

    Returns
    -------
    float : the weight to assign to the new connection
    """
    nonzero = target_row[target_row > 0]

    if mode == "same":
        return weight

    elif mode == "mean":
        return np.mean(nonzero) if len(nonzero) > 0 else weight

    elif mode == "min":
        return np.min(nonzero) if len(nonzero) > 0 else weight

    elif mode == "max":
        return np.max(nonzero) if len(nonzero) > 0 else weight

    elif mode == "noise":
        base = np.mean(nonzero) if len(nonzero) > 0 else weight
        return max(0, base + np.random.normal(0, base * 0.1))

    else:
        raise ValueError(f"Unknown weight mode: {mode}")


def rewire_matrix(original_C, lesioned_i, label_vector, weight_mode="same", seed=None):
    """
    Lesion a node and rewire its lost connections via collateral sprouting
    within the same module.

    Parameters
    ----------
    original_C : np.ndarray (N, N)
        Original (pre-lesion) structural connectivity matrix.
        Convention: C[i, j] = connection weight FROM j TO i.
    lesioned_i : int
        Index of the node to lesion.
    label_vector : np.ndarray (N,)
        Module/community label for each node.
    weight_mode : str
        Weight initialization rule passed to new_weight().
        One of: "same", "mean", "min", "max", "noise".
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    lesioned_C : np.ndarray (N, N)
        Matrix with lesioned node's row and column zeroed.
    rewired_C : np.ndarray (N, N)
        Matrix after rewiring lost connections within the module.
    """

    rng = np.random.default_rng(seed)
    N = original_C.shape[0]

    # --- Step 1: build lesioned matrix ---
    lesioned_C = original_C.copy()
    lesioned_C[lesioned_i, :] = 0
    lesioned_C[:, lesioned_i] = 0

    # --- Step 2: identify same-module nodes (excluding lesioned node) ---
    lesioned_module = label_vector[lesioned_i]
    same_module_mask = (label_vector == lesioned_module)
    same_module_mask[lesioned_i] = False
    same_module_nodes = np.where(same_module_mask)[0]

    if len(same_module_nodes) == 0:
        print(f"Warning: node {lesioned_i} is alone in its module. Returning lesioned matrix only.")
        return lesioned_C, lesioned_C.copy()

    rewired_C = lesioned_C.copy()

    # --- Step 3: AFFERENT rewiring ---
    # original_C[lesioned_i, source] = weight coming FROM source TO lesioned node
    # Rerouted to: a random node within the lesioned module receives from source instead
    for source in range(N):
        if source == lesioned_i:
            continue
        weight = original_C[lesioned_i, source]
        if weight == 0:
            continue

        source_module = label_vector[source]
        if source_module == lesioned_module:
            # INTRA-MODULE AFFERENT:
            # source (same module) was sending to lesioned_i (same module)
            # → reroute to another node within the module (excluding source itself)
            candidates = same_module_nodes[same_module_nodes != source]
        else:
            # INTER-MODULE AFFERENT:
            # source (different module) was sending to lesioned_i
            # → reroute to any node within the lesioned module
            candidates = same_module_nodes

        if len(candidates) == 0:
            continue

        target = rng.choice(candidates)
        # ============================================================
        # WEIGHT INITIALIZATION (afferent)
        # ============================================================
        rewired_C[target, source] += new_weight(weight, rewired_C[target, :], mode=weight_mode)

    # --- Step 4: EFFERENT rewiring ---
    # original_C[dest, lesioned_i] = weight going FROM lesioned node TO dest
    # Rerouted to: a random node within the lesioned module sends to dest instead
    for dest in range(N):
        if dest == lesioned_i:
            continue
        weight = original_C[dest, lesioned_i]
        if weight == 0:
            continue

        dest_module = label_vector[dest]
        if dest_module == lesioned_module:
            # INTRA-MODULE EFFERENT:
            # lesioned_i (same module) was sending to dest (same module)
            # → reroute from another node within the module (excluding dest itself)
            candidates = same_module_nodes[same_module_nodes != dest]
        else:
            # INTER-MODULE EFFERENT:
            # lesioned_i was sending to dest (different module)
            # → reroute from any node within the lesioned module
            candidates = same_module_nodes

        if len(candidates) == 0:
            continue

        proxy = rng.choice(candidates)
        # ============================================================
        # WEIGHT INITIALIZATION (efferent)
        # ============================================================
        rewired_C[dest, proxy] += new_weight(weight, rewired_C[:, proxy], mode=weight_mode)

    return lesioned_C, rewired_C


# --- Example usage ---
if __name__ == "__main__":
    
    from scipy.io import loadmat
    
    N = 246
    C = np.random.rand(N, N)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 0)

    labels  = loadmat(folder+"P_block.mat")["Labels"].flatten()
    lesioned_i = 2

    lesioned, rewired = rewire_matrix(C, lesioned_i, labels, weight_mode="same", seed=42)

    print(f"Lesioned node: {lesioned_i}, module: {labels[lesioned_i]}")
    print(f"Row {lesioned_i} in lesioned (should be 0): {lesioned[lesioned_i, :]}")
    print(f"Row {lesioned_i} in rewired  (should be 0): {rewired[lesioned_i, :]}")
    print(f"\nTotal weight original : {C.sum():.4f}")
    print(f"Total weight lesioned : {lesioned.sum():.4f}")
    print(f"Total weight rewired  : {rewired.sum():.4f}")    
    print(f"Total weight rewired  : {rewired.sum():.4f}")