import os
import csv
import numpy as np
import pandas as pd
from core_mc import probe_equilibration, detect_equil_index, simulate_at_T, quad_peak_T

def run_for_L(N, T_probe=0.78, max_probe_sweeps=6000, prod_sweeps=12000, seed=42):
    print(f"\n=== Running L={N} ===")

    # --- Step 0: Probe equilibration ---
    E_tr, S_tr, acc_tr = probe_equilibration(T=T_probe, N=N, max_sweeps=max_probe_sweeps, seed=seed)
    idx_E = detect_equil_index(E_tr, window=100)
    idx_S = detect_equil_index(S_tr, window=100)
    burnin = max(idx_E, idx_S)
    equil_sweeps = int(burnin * 1.2)
    print(f"[L={N}] Equilibrium detected at sweep ≈ {burnin}, using equil_sweeps = {equil_sweeps}")

    # --- Step 1: Temperature scan ---
    T_list = np.arange(0.6, 0.90, 0.02)
    S_means, S_errs, chis, capacities = [], [], [], []
    results_S, results_E = {}, {}

    for T in T_list:
        print(f"   → Simulating at T={T:.3f}")
        S_mean, S_sem, acc_rate, amp, chi, capacity, S_samples, E_samples, acc_rate_equil, k, expected = simulate_at_T(
            T, N, equil_sweeps, prod_sweeps, seed
        )
        S_means.append(S_mean); S_errs.append(S_sem)
        chis.append(chi); capacities.append(capacity)
        results_S[T] = S_samples; results_E[T] = E_samples

    # --- Step 2: Estimate Tc from χ(T) ---
    Tc_chi = quad_peak_T(T_list, chis)
    Tc_cv  = quad_peak_T(T_list, capacities)
    print(f"\n[L={N}] Estimated Tc(χ) ≈ {Tc_chi:.4f},  Tc(Cv) ≈ {Tc_cv:.4f}")

    # --- Step 3: Save results ---
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/raw_L{N}", exist_ok=True)

    for T in T_list:
        np.savez_compressed(f"results/raw_L{N}/T{T:.3f}.npz",
                            S_samples=results_S[T],
                            E_samples=results_E[T])

    df = pd.DataFrame({
        "T": T_list,
        "S_mean": S_means,
        "S_err": S_errs,
        "chi": chis,
        "Cv": capacities
    })
    df.to_csv(f"results/L{N}_scan.csv", index=False)

    with open("results/Tc_vs_L.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([N, Tc_chi, Tc_cv])

    print(f"[L={N}] Saved results → results/L{N}_scan.csv + updated Tc_vs_L.csv")
    return Tc_chi, Tc_cv
