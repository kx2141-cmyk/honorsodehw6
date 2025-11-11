# Let's implement and run the model described by the user.
# We'll (1) compute the steady states analytically for chosen parameters,
# (2) simulate the ODE system with Euler's method to verify stability,
# (3) demonstrate switching by making alpha(t) time-dependent,
# and (4) save the code so the user can download and reuse it.

import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# Model definitions
# ----------------------

def f_piecewise(pn, k, q):
    """Transcription activation via relief of inhibition."""
    if pn < q:
        return k / (k + q - pn)
    else:
        return 1.0

def simulate(params, T=200.0, dt=0.01, x0=None, alpha_schedule=None, seed=0):
    """
    Euler simulation of the 4D cascade.
    alpha_schedule: function t -> alpha(t). If None, uses constant params['alpha'].
    """
    random.seed(seed)
    n_steps = int(T/dt)
    t = np.linspace(0, T, n_steps+1)

    # unpack parameters
    alpha = params['alpha']
    beta = params['beta']
    gamma_m = params['gamma_m']
    delta_m = params['delta_m']
    gamma_p = params['gamma_p']
    delta_p = params['delta_p']
    k = params['k']
    q = params['q']

    # initial conditions
    if x0 is None:
        x = np.array([0.0, 0.0, 0.0, 0.0])  # [mn, mc, pc, pn]
    else:
        x = np.array(x0, dtype=float)

    traj = np.zeros((n_steps+1, 5))  # t, mn, mc, pc, pn
    traj[0, :] = [0.0, *x]

    for i in range(n_steps):
        ti = t[i]
        # handle alpha(t)
        if alpha_schedule is not None:
            alpha_t = alpha_schedule(ti)
        else:
            alpha_t = alpha

        mn, mc, pc, pn = x

        # compute f(pn)
        fval = f_piecewise(pn, k, q)

        # Euler updates
        dmn = alpha_t * fval - gamma_m * mn
        dmc = gamma_m * mn - delta_m * mc
        dpc = beta * mc - gamma_p * pc
        dpn = gamma_p * pc - delta_p * pn

        x = x + dt * np.array([dmn, dmc, dpc, dpn])
        traj[i+1, :] = [t[i+1], *x]

    return traj

# ----------------------
# Chosen parameters to yield 3 steady states
# ----------------------
# We choose Q = q/k = 3 (so q=3, k=1) and A in (Q, (1+Q)^2/4) = (3,4).
# Let A = 3.5 by setting alpha = 3.5, beta=1, delta_m=1, delta_p=1.
# Other rates gamma_m, gamma_p set to 1 for simplicity.
params = {
    'alpha': 3.5,
    'beta': 1.0,
    'gamma_m': 1.0,
    'delta_m': 1.0,
    'gamma_p': 1.0,
    'delta_p': 1.0,
    'k': 1.0,
    'q': 3.0
}
Q = params['q']/params['k']
A = params['alpha']*params['beta']/(params['k']*params['delta_m']*params['delta_p'])

# ----------------------
# Analytical steady states for these parameters
# ----------------------

def steady_states(Q, A):
    """Return the (valid) steady states Pn for given Q, A (rescaled Pn = pn/k)."""
    states = []
    # Case F=1: Pn = A, valid only if Pn >= Q
    if A >= Q:
        states.append(A)  # high state

    # Case Pn < Q: solve Pn^2 - (1+Q)Pn + A = 0 if discriminant >= 0
    D = (1+Q)**2 - 4*A
    if D >= 0:
        sqrtD = math.sqrt(D)
        r_plus = ((1+Q) + sqrtD)/2.0
        r_minus = ((1+Q) - sqrtD)/2.0
        # Valid only if < Q
        if r_minus < Q:
            states.append(r_minus)
        if r_plus < Q:
            states.append(r_plus)
    # Sort
    return sorted(states)

Pn_states = steady_states(Q, A)
pn_states = [params['k']*s for s in Pn_states]  # convert to original units

# ----------------------
# Verify stability numerically by sim near each steady state
# ----------------------
def run_and_plot_near(pn_target, label, offset=0.1):
    """
    Start near a given pn steady state and plot the resulting time series to test stability.
    We'll randomize mn, mc, pc small amounts and pn near target.
    """
    trials = []
    for j in range(3):  # a few trials
        pn0 = pn_target + (2*random.random()-1.0)*offset
        x0 = [random.uniform(0, 0.25), random.uniform(0, 0.25), random.uniform(0, 0.25), pn0]
        traj = simulate(params, T=120.0, dt=0.01, x0=x0, alpha_schedule=None, seed=j)
        trials.append(traj)

    # Plot pn over time for these trials
    plt.figure()
    for tr in trials:
        plt.plot(tr[:,0], tr[:,4], linewidth=1.5)
    plt.axhline(pn_target, linestyle='--')
    plt.title(f'Trajectories near {label} steady state (target pn≈{pn_target:.3f})')
    plt.xlabel('t')
    plt.ylabel('pn')
    plt.show()

# Run stability checks
# lower (stable), middle (unstable), upper (stable) predicted.
# With Q=3, A=3.5: we expect pn ≈ [1.293, 2.707, 3.5]
low, mid, high = pn_states[0], pn_states[1], pn_states[2]

run_and_plot_near(low, "LOW (predicted stable)", offset=0.2)
run_and_plot_near(mid, "MIDDLE (predicted UNstable)", offset=0.2)
run_and_plot_near(high, "HIGH (predicted stable)", offset=0.2)

# ----------------------
# Switching demos by pulsing alpha(t)
# ----------------------

# 1) Switch LOW -> HIGH by temporarily raising alpha above A_max = ((1+Q)^2)/4 = 4
A_max = ((1+Q)**2)/4.0
alpha_hi = 4.5  # > A_max
alpha_base = params['alpha']

def alpha_pulse_up(t, t1=20.0, t2=60.0):
    if t1 <= t <= t2:
        return alpha_hi
    return alpha_base

# Start at low state
x0_low = [0.0, 0.0, 0.0, low]
traj_up = simulate(params, T=120.0, dt=0.01, x0=x0_low,
                   alpha_schedule=lambda tt: alpha_pulse_up(tt, 20.0, 60.0))

plt.figure()
plt.plot(traj_up[:,0], traj_up[:,4], linewidth=1.5)
plt.axvspan(20.0, 60.0, alpha=0.2)
plt.axhline(low, linestyle='--')
plt.axhline(high, linestyle='--')
plt.title('Switching LOW → HIGH via temporary increase in alpha')
plt.xlabel('t')
plt.ylabel('pn')
plt.show()

# 2) Switch HIGH -> LOW by temporarily lowering alpha below Q
alpha_lo = 2.5  # < Q=3 so the high F=1 state is invalid during the pulse

def alpha_pulse_down(t, t1=20.0, t2=60.0):
    if t1 <= t <= t2:
        return alpha_lo
    return alpha_base

x0_high = [0.0, 0.0, 0.0, high]
traj_down = simulate(params, T=120.0, dt=0.01, x0=x0_high,
                     alpha_schedule=lambda tt: alpha_pulse_down(tt, 20.0, 60.0))

plt.figure()
plt.plot(traj_down[:,0], traj_down[:,4], linewidth=1.5)
plt.axvspan(20.0, 60.0, alpha=0.2)
plt.axhline(low, linestyle='--')
plt.axhline(high, linestyle='--')
plt.title('Switching HIGH → LOW via temporary decrease in alpha')
plt.xlabel('t')
plt.ylabel('pn')
plt.show()

# ----------------------
# Save a reusable Python script
# ----------------------
script_path = "/mnt/data/genetic_switch_relief_inhibition.py"
with open(script_path, "w") as f:
    f.write(r'''import math
import random
import numpy as np

def f_piecewise(pn, k, q):
    if pn < q:
        return k / (k + q - pn)
    else:
        return 1.0

def simulate(params, T=200.0, dt=0.01, x0=None, alpha_schedule=None, seed=0):
    random.seed(seed)
    n_steps = int(T/dt)
    t = np.linspace(0, T, n_steps+1)

    alpha = params['alpha']
    beta = params['beta']
    gamma_m = params['gamma_m']
    delta_m = params['delta_m']
    gamma_p = params['gamma_p']
    delta_p = params['delta_p']
    k = params['k']
    q = params['q']

    if x0 is None:
        x = np.array([0.0, 0.0, 0.0, 0.0])
    else:
        x = np.array(x0, dtype=float)

    traj = np.zeros((n_steps+1, 5))
    traj[0, :] = [0.0, *x]

    for i in range(n_steps):
        ti = t[i]
        if alpha_schedule is not None:
            alpha_t = alpha_schedule(ti)
        else:
            alpha_t = alpha

        mn, mc, pc, pn = x
        fval = f_piecewise(pn, k, q)

        dmn = alpha_t * fval - gamma_m * mn
        dmc = gamma_m * mn - delta_m * mc
        dpc = beta * mc - gamma_p * pc
        dpn = gamma_p * pc - delta_p * pn

        x = x + dt * np.array([dmn, dmc, dpc, dpn])
        traj[i+1, :] = [t[i+1], *x]

    return traj

if __name__ == "__main__":
    params = {
        'alpha': 3.5,
        'beta': 1.0,
        'gamma_m': 1.0,
        'delta_m': 1.0,
        'gamma_p': 1.0,
        'delta_p': 1.0,
        'k': 1.0,
        'q': 3.0
    }
    # Example: run a quick sim
    traj = simulate(params, T=10.0, dt=0.01, x0=[0,0,0,0])
    print("Final state:", traj[-1, :])
''')

script_path
