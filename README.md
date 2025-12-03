# 2D XY Model Monte-Carlo Simulation (BKT Transition)

 model using:
- **Metropolis Monte-Carlo algorithm**
- **Periodic boundary conditions**
- **Finite-size scaling**
- **Nematic correlation function**
- **Thermal averages over long Monte-Carlo runs**

---

## Physics Background

The 2D XY model exhibits a **Berezinskii–Kosterlitz–Thouless (BKT)** topological phase transition at
a critical temperature around:

\[
T_c(\infty) \approx 0.655 \pm 0.015
\]

Features of the BKT transition:
- No spontaneous magnetization for \(T>0\)
- Quasi-long-range order below \(T_c\)
- Vortex–antivortex unbinding at \(T_c\)

---

##  What We Compute

| Quantity | Symbol | Purpose |
|---------|--------|---------|
| Order parameter | ⟨S⟩ | Detect degree of alignment |
| Susceptibility | χ(T) | Locate transition peak |
| Heat capacity | Cᵥ(T) | Check thermodynamic anomaly |
| Nematic correlation | g₂(r) | Confirm algebraic vs exponential decay |
| Auto-correlation | ACF | Check statistical independence |
| Snapshots | θ(x,y) | Identify vortex behavior |

Simulated lattice sizes:

\[
L = 20, 30, 40, 50, 60, 80
\]

Temperature range:

\[
T = 0.60 \rightarrow 0.90 \quad (\Delta T = 0.01)
\]

---

