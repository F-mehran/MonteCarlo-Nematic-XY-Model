import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.optimize import curve_fit
epsilon = 1.0

def initgrid(N, seed):
    np.random.seed(seed)
    theta= np.random.rand(N,N)*2*np.pi
    return theta

def neighbours(i,j,N):
    nbrs= [((i+1) % N,j) , ((i-1) % N , j), (i,(j-1) % N) , (i, (j+1) % N)]
    return nbrs

def Pair_Energy(theta_i,theta_j):
    delta= theta_i - theta_j
    cosd= np.cos(delta)
    P2= 0.5* (3*cosd**2 -1 )
    return -epsilon * P2

def Local_Energy(i,j,theta,N):
    E=0
    for (ii,jj) in neighbours(i,j,N):
        E += Pair_Energy(theta[i,j],theta[ii,jj])
    return E

def Whole_Energy (theta,N):
    E=0
    for i in range(N):
        for j in range(N):
            E += Local_Energy(i,j,theta,N)
    return 0.5*E

def metropolis_step(N,theta,amp,T):
    i= np.random.randint(N)
    j=np.random.randint(N)
    old = Local_Energy(i,j,theta,N)
    old_angle= theta[i,j].copy()
    delta= (np.random.rand()-0.5)*amp
    theta[i,j]= (theta[i,j]+ delta) % (2*np.pi)
    new = Local_Energy(i,j,theta,N)
    dE= new - old
    if dE < 0 or np.random.rand()< np.exp(-dE/T):
        return dE, True #accepted
    else:
        theta[i,j]= old_angle
        return dE, False   
    
def Order_Parameter(theta):
    c2= np.cos(2*theta).mean()
    s2= np.sin(2*theta).mean()
    S= np.sqrt(c2**2 + s2**2)
    return S

def probe_equilibration(T, N, max_sweeps=8000, seed=0):
    np.random.seed(seed)
    theta = initgrid(N, seed)
    E_total = Whole_Energy(theta,N)

    target_low, target_high = 0.30, 0.45
    AMP_MIN, AMP_MAX = 0.05, np.pi
    amp = 0.8

    E_trace, S_trace, acc_trace = [], [], []
    acc = att = 0

    for step in range(max_sweeps * N * N):
        dE, accepted = metropolis_step(N,theta,amp,T)
        att += 1
        if accepted:
            E_total += dE
            acc += 1

        # each sweep
        if (step + 1) % (N * N) == 0:
            rate = acc / max(att, 1)
            # regulate amp
            if step < (max_sweeps * N * N // 2):
                if rate > target_high: amp *= 1.2
                elif rate < target_low: amp *= 0.8
            else:
                if rate > target_high: amp = min(AMP_MAX, amp * 1.1)
                elif rate < target_low: amp = max(AMP_MIN, amp * 0.9)

            E_trace.append(E_total / (N*N))        # energy per each site
            S_trace.append(Order_Parameter(theta)) # order parameter
            acc_trace.append(rate)
            acc = att = 0

    return np.array(E_trace), np.array(S_trace), np.array(acc_trace)

def detect_equil_index(series, window=50, eps_mean=1e-3, eps_slope=1e-4):
    import numpy as np
    x = np.asarray(series)
    if len(x) < window*2: return len(x)-1  #data is few.

    
    def slope(y):
        n = len(y); t = np.arange(n)
        a, b = np.polyfit(t, y, 1)  # y ≈ a t + b
        return a

    for i in range(window, len(x)-window):
        m1 = x[i-window:i].mean()
        m2 = x[i:i+window].mean()
        s2 = slope(x[i:i+window])
        if abs(m2 - m1) < eps_mean and abs(s2) < eps_slope:
            return i  #the first spot that is constant.
    return len(x)-1

def simulate_at_T(T, N, equil_sweeps, prod_sweeps, seed=0):
    np.random.seed(seed)
    theta = initgrid(N, seed=seed)
    E_total = Whole_Energy(theta,N)

    target_low, target_high = 0.30, 0.45
    amp = 0.8
    acc = 0; att = 0
    rate = 0
    AMP_MIN = 0.05
    AMP_MAX = 5 * np.pi / 4

    #  Equilibration 
    for step in trange(equil_sweeps * N * N, desc=f"Equil T={T}", leave=False):
        dE, accepted = metropolis_step(N, theta, amp, T)
        att += 1
        if accepted:
            E_total += dE
            acc += 1

        if (step + 1) % (N * N) == 0:
            rate = acc / max(att, 1)

            if step < (equil_sweeps * N * N // 2):
                if rate > target_high: amp *= 1.2
                elif rate < target_low: amp *= 0.8
            else:
                if rate > target_high: amp = min(AMP_MAX, amp * 1.1)
                elif rate < target_low: amp = max(AMP_MIN, amp * 0.9)

            acc = 0; att = 0

    acc_rate_equil = rate

    #  Adaptive sampling based on acceptance rate 
    if acc_rate_equil > 0.6:
        sample_every = 1 * N * N
    elif acc_rate_equil > 0.4:
        sample_every = 2 * N * N
    else:
        sample_every = 3 * N * N

    k = sample_every // (N * N)
    expected = int(prod_sweeps / max(k, 1))

    # --- Production ---
    S_samples = []
    Energy_samples = []
    acc = 0; att = 0
    for step in trange(prod_sweeps * N * N, desc=f"Prod T={T}", leave=False):
        dE, accepted = metropolis_step(N, theta, amp, T)
        att += 1
        if accepted:
            E_total += dE
            acc += 1

        if (step + 1) % sample_every == 0:
            S_samples.append(Order_Parameter(theta))
            Energy_samples.append(E_total)

    acc_rate = acc / max(att, 1)

    # --- Statistics ---
    S_samples = np.array(S_samples)
    E_arr = np.array(Energy_samples)

    S_mean = np.mean(S_samples)
    varS = np.var(S_samples, ddof=1)
    varE = np.var(E_arr, ddof=1)

    chi = (N * N) * varS / T
    capacity = varE / (N * N * T**2)


    S_sem = S_samples.std(ddof=1) / np.sqrt(len(S_samples))

    return S_mean, S_sem, acc_rate, amp, chi, capacity, S_samples, Energy_samples, acc_rate_equil, k, expected



# Quadratic fitting function 
def quad(x, a, b, c):
    return a*x**2 + b*x + c

def quad_peak_T(T_list, y_values):
    """Finds the peak position (Tc) by fitting a quadratic curve near the maximum."""
    i_peak = np.argmax(y_values)
    i1 = max(0, i_peak-2)
    i2 = min(len(T_list), i_peak+3)
    T_fit = np.array(T_list[i1:i2])
    y_fit = np.array(y_values[i1:i2])
    popt, _ = curve_fit(quad, T_fit, y_fit)
    a, b, c = popt
    Tc_fit = -b / (2*a)
    return Tc_fit
