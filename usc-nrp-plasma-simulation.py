# %%
"""
BOLOS is used to solve the Boltzmann equation for a given reduced electric field and gas temperature.
This determines the electron energy distribution function (EEDF).
"""


import numpy as np
import scipy.constants as co
from bolos import parser, solver, grid

# Direct input values - modify these as needed
input_file = "data/biagi-v7.1-He"  # Path to your cross-sections file
debug = False  # Set to True for debugging output
en_value = 100.0  # Reduced field (in Td)
temp_value = 300.0  # Gas temperature (in K)

# Use a linear grid from 0 to 60 eV with 500 intervals.
gr = grid.LinearGrid(0, 60., 500)

# Initiate the solver instance
bsolver = solver.BoltzmannSolver(gr)

# Parse the cross-section file in BOSIG+ format and load it into the
# solver.
with open(input_file) as fp:
    bsolver.load_collisions(parser.parse(fp))

# Set the conditions.  And initialize the solver
# bsolver.target['N2'].density = 0.8
bsolver.target['He'].density = 1.0
bsolver.kT = temp_value * co.k / co.eV
bsolver.EN = en_value * solver.TOWNSEND
bsolver.init()

# Start with Maxwell EEDF as initial guess.  Here we are starting with
# with an electron temperature of 2 eV
f0 = bsolver.maxwell(2.0)

# Solve the Boltzmann equation with a tolerance rtol and maxn iterations.
f0 = bsolver.converge(f0, maxn=200, rtol=1e-4)

# Second pass: with an automatic grid and a lower tolerance.
# mean_energy = bsolver.mean_energy(f0)
newgrid = grid.QuadraticGrid(0, 1000, 1000)
bsolver.grid = newgrid
bsolver.init()

f1 = bsolver.grid.interpolate(f0, gr)
f1 = bsolver.converge(f1, maxn=200, rtol=1e-5)

# You can also iterate over all processes or over processes of a certain
# type.
print("\nREACTION RATES:\n")
for t, p in bsolver.iter_all():
    print("{:<40}   {:.3e} m^3/s".format(str(p), bsolver.rate(f1, p)))

# Calculate the mobility and diffusion.
print("\nTRANSPORT PARAMETERS:\n")
print("mobility * N   = {:.4e}  1/m/V/s".format(bsolver.mobility(f1)))
print("diffusion * N  = {:.4e}  1/m/s".format(bsolver.diffusion(f1)))
print("average energy = {:.4e}  eV".format(bsolver.mean_energy(f1)))

# Interpolate f1 back to the boundary grid
from dataclasses import dataclass
from typing import Optional
import numpy as np

EEDF_f = np.interp(bsolver.grid.b, bsolver.grid.c, f1, left=0.0, right=0.0)
EEDF_grid = bsolver.grid.b


# %%
# Cantera take the EEDF
import cantera as ct

from extensible_two_temp_plasma import TwoTempPlasmaRate
# read input file
plasma = ct.Solution("data/helium-oxygen-hydrogen-plasma.yaml", phase="basic-plasma")

# %%
class ReactorOde:
    def __init__(self, plasma):
        # Parameters of the ODE system and auxiliary data are stored in the
        # ReactorOde object.
        self.gas = plasma
        self.P = plasma.P

    def __call__(self, t, y):
        """the ODE function, y' = f(t,y) """

        self.gas.concentrations = y

        wdot = self.gas.net_production_rates
        dndt = wdot

        return np.hstack((dndt))

# %%
# setup for the pulse simulation
import scipy.integrate

plasma.TPX = 300, 0.01 * ct.one_atm, "He:1.0, He+:1e-6, e:1e-6"
y0 = np.hstack(plasma.concentrations)

# Set up objects representing the ODE and the solver
ode = ReactorOde(plasma)
solver = scipy.integrate.ode(ode)
solver.set_integrator('vode', method='bdf', with_jacobian=True)
solver.set_initial_value(y0, 0.0)

# pulse time and steps
pulse_duration = 1e-8  # 10 ns pulse
dt_pulse = 1e-11
nSteps_pulse = int(pulse_duration / dt_pulse)

# decay time and steps
cycle_duration = 1e-5  # 10 μs cycle (pulse + decay) 10k Hz
dt_decay = 1e-8
nSteps_decay = int((cycle_duration - pulse_duration) / dt_decay)

# print out the time & steps values
print(f"pulse_duration = {pulse_duration}")
print(f"cycle_duration = {cycle_duration}")
print(f"dt_pulse = {dt_pulse}")
print(f"nSteps_pulse = {nSteps_pulse}")
print(f"dt_decay = {dt_decay}")
print(f"nSteps_decay = {nSteps_decay}")

# %%
for p_index in range(10):
    states = ct.SolutionArray(plasma, 1, extra={'t': [0.0]})

    # At the beginning, create a separate Te tracking array:
    te_values = [plasma.Te]
    time_values = [0.0]

    # Pulse stage: high energy electrons
    plasma.set_discretized_electron_energy_distribution(EEDF_grid, EEDF_f)

    step_counter = 0
    while solver.successful() and step_counter != nSteps_pulse:
        step_counter += 1
        solver.integrate(solver.t + dt_pulse)
        plasma.concentrations = solver.y
        states.append(plasma.state, t=solver.t)
        te_values.append(plasma.Te)  # Manually track Te

    # Decay stage: thermal electrons
    plasma.electron_energy_distribution_type = "isotropic"
    plasma.Te = plasma.T

    step_counter = 0
    while solver.successful() and step_counter != nSteps_decay:
        step_counter += 1
        solver.integrate(solver.t + dt_decay)
        plasma.concentrations = solver.y
        states.append(plasma.state, t=solver.t)
        te_values.append(plasma.Te)  # Manually track Te

    print(f"Completed pulse {p_index + 1}, time: {solver.t:.2e} s")
    y0 = np.hstack(plasma.concentrations)
    solver.set_initial_value(y0, 0.0)




# %%
# Plot the results
try:
    import matplotlib.pyplot as plt

    # Create two separate subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1: Electron Temperature (use manually tracked values)
    ax1.plot(states.t, te_values[:len(states.t)], color='r', label='Te', lw=2)
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Electron Temperature (K)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')  # Use log scale to better see the temperature jumps
    ax1.set_xscale('log')

    # Plot 2: Species Mole Fractions (Electron and He*)
    ax2.plot(states.t, states('e').X, color='b', label='e', lw=2)
    ax2.plot(states.t, states('He*').X, color='g', label='He*', lw=2)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Mole Fraction')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')  # Likely small values, so log scale helps
    ax2.set_xscale('log')

    plt.tight_layout()

    plt.show()

except ImportError:
    print('Matplotlib not found. Unable to plot results.')



