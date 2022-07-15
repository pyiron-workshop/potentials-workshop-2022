#!/usr/bin/env python
# coding: utf-8

# # Interatomic potentials
# 
# In general interatomic potentials can be written as a sum of functional terms depending on the positions of the atoms in a structure. Then the energy $U$ of the system is
# 
# $U = \sum_{i=1}^N U_1(\vec{r_i}) + \sum_{i,j=1}^N U_2(\vec{r_i}, \vec{r_j}) + \sum_{i,j,k=1}^N U_2(\vec{r_i}, \vec{r_j}, \vec{r_k}) +...$
# 
# The one body term only matter when the system is within an external field. Most classic interatomic potentials don't use 4th and higher order body terms, pair potentials only use the 2 body term. As a general rule potentials that include higher order body terms can be more accurate but are slower.
# 
# There are many different forms for $U_i$, which cover a wide range of different run time and accuracy necessities.
# 
# Simple pair potentials (f.e Lennard-Jones, Morse, Buckingham) only contain very few parameters, typically less than 10. In many body potentials (f.e. EAM, MEAM, Tersoff) the typical number of parameters is 10-50 and for machine learning potentials the number of parameters can reach several thousands.
# 
# # Fitting
# 
# In the fit process the parameters of the chosen functions for $U_i$ are optimized. For this purpose an objective or cost function is defined and minimized. In general the objective function is defined as
# 
# $\chi^2 = \sum_i w_i r_i$
# 
# where $w_i$ is a weight and $r_i$ is a residual that describes the difference to target values. This residual can be defined in different ways, so it is not possible to simply compare the residual for different fitting processes or codes. A more in depth explanation and some examples can be found on https://atomicrex.org/overview.html#objective-function.
# 
# The minimization can be done with local or global optimization algorithms.
# Generally local optimization algorithms should all be able to find the local minimum coming from some initial parameter set, so the "best" local algorithm is the one finding the minimum in the shortest time. Typically used local algorithms are f.e. (L)BFGS or Nelder-Mead.
# Examples for global algorithms are evolutionary algorithms or simulated annealing. For most problems it is impossible to tell a priori which global algorithm will give the best results, so using global algorithms typically involves testing many of them.
# 
# # EAM potentials
# 
# EAM potentials are pair functionals. 
# In a generalised form they are equal to Finnis-Sinclair, effective medium theory or glue potentials. Their total energy can be written as
# 
# $E = \frac{1}{2}\sum_{ij}V(r_{ij}) + \sum_i F(\rho_i)$
# 
# with
# 
# $\rho_i = \sum_j \rho(r_{ij})$
# 
# The original functions for V, $\rho$ and F were derived from different theories, but they can be chosen freely.
# 
# # Fitting code
# 
# Fitting is done using the pyiron interface to the atomicrex code https://atomicrex.org. It can be used to fit different types of classic interatomic potentials:
# - pair potentials
# - EAM
# - MEAM
# - Tersoff
# - ABOP
# - ADP (in development)
# 
# It allows to fit different properties (energies, forces, lattice constants, elastic properties, etc.) and implements the LBFGS minimizer. Additionally it offers an interface to the nlopt library which implements several global and local optimization algorithms and the ability to apply arbitrary constraints to parameters.

# In[ ]:


from pyiron import Project, ase_to_pyiron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


pr = Project(".")


# ### Get the training data
# Load a job that contains structures with energies and forces from DFT

# In[ ]:


tc = pr['../../introduction/training/Al_basic_atomicrex']


# ### Have a look at the training data

# In[ ]:


tc.plot.energy_volume(crystal_systems=True)


# In[ ]:


plt.figure(figsize=(20,7))
tc.plot.cell()


# ### Create an atomicrex job

# In[ ]:


job = pr.create.job.Atomicrex("FitAl", delete_existing_job=True)


# ### Training Data
# Set the training data. This can also be done structure by structure to set other fit properties and weights, but here we simply load the TrainingContainer

# In[ ]:


job.add_training_data(tc)


# Set the potential type. In this case an EAM potential

# In[ ]:


job.potential = job.factories.potentials.eam_potential()


# ### Functions
# Reminder: $E = \sum_{ij}V(r_{ij}) + \sum_i F(\rho_i)$ with $\rho_i = \sum_j \rho(r_{ij})$
# 
# It is necessary to define a pair potential, an electronic density function and an embedding function.
# For all of those it is possible to choose between different functional forms.
# Classic pair potentials are physically motivated and have a very limited number of paramaters that can often be derived from an experimentally measurable quantity.
# Splines or polynomials offer more flexibility, but can lead to unphysical oscillations or overfitting. Compared with the machine learning potentials shown later the number of parameters is very low no matter which functions you choose.
# 
# In this case a generalized morse function is used for the pair interaction. It has the form
# 
# $(\frac{D_0}{S-1}exp(-\beta \sqrt{2S}(r-r_0))-\frac{D_0S}{S-1}exp(-\beta\sqrt{2/S}(r-r_0)))+\delta $
# 
# The parameters in the morse potential can be derived from phyiscal quantities, here they are just educated guesses. For example $r_0$ is the equilibrium distance of a dimer. The nearest neighbor distance in fcc Cu is about 2.8 $\mathring A$ so it is taken as initial value.
# In the case of analytic functions the initial parameter choices should not matter too much, since the functional form is constrained.
# 
# The electronic density will be a cubic spline. The embedding function will be $-A*sqrt(\rho)+B*rho$, which can be defined as a user function.
# 
# The pair function and the electron denity and their first derivatives are required to smoothly approach 0 at the cutoff distance $r_{c}$
# For this purpose the pair function is screened by multiplying with the function:
# 
# $\Psi(\frac{r-rc}{h})$ where $\Psi(x) = \frac{x^4}{1+x^4}$ if $x<0$ else $\Psi(x)=0$
# 
# For the spline it is necessary to set a node point with y value 0 at the cutoff 

# In[ ]:


morseB = job.factories.functions.morse_B("V", D0=0.15, r0=3.05, beta=1.1, S=4.1, delta=-0.01)
morseB.screening = job.factories.functions.x_pow_n_cutoff(identifier="V_screen", cutoff=7.6)
morseB.parameters.D0.min_val = 0.05
morseB.parameters.D0.max_val = 1.55

morseB.parameters.r0.min_val = 2.6
morseB.parameters.r0.max_val = 3.1

morseB.parameters.S.min_val = 1.5
morseB.parameters.S.max_val = 4.5
morseB.parameters.delta.max_val = 0.005


# It is also possible to plot most of the functions. This can help to judge if the initial parameters are reasonable

# In[ ]:


morseB.plot()


# In[ ]:


rho = job.factories.functions.spline(identifier="rho_AlAl", cutoff=7.6, derivative_left=-0.2, derivative_right=0)
## the spline requires node points and initial values
init_func = lambda r: np.exp(-r)
nodes = np.array([1.0, 2.2, 2.7, 3.2, 4.0, 5.0])
init_vals = init_func(nodes)
rho.parameters.create_from_arrays(x=nodes, y=init_vals, min_vals=np.zeros(len(nodes)), max_vals=np.ones(len(nodes)))
# set node point at cutoff
rho.parameters.add_node(x=7.6, start_val=0, enabled=False)
rho.derivative_left.max_val = 0.0


# In[ ]:


# User function for embedding term
F = job.factories.functions.user_function(identifier="F", input_variable="r")
F.expression = "-A*sqrt(r)+B*r"
F.derivative = "-A/(2*sqrt(r))+B"
F.parameters.add_parameter("A", start_val=3.3, min_val=0.0)
F.parameters.add_parameter("B", start_val=1.8, min_val=0.0)


# Assign functions

# In[ ]:


job.potential.pair_interactions[morseB.identifier] = morseB
job.potential.electron_densities[rho.identifier] = rho
job.potential.embedding_energies[F.identifier] = F


# Set a fit algorithm

# In[ ]:


job.input.fit_algorithm = job.factories.algorithms.ar_lbfgs(max_iter=500)


# In[ ]:


job.run()


# Have a look at some of the outputs

# In[ ]:


plt.plot(job.output.iterations, job.output.residual)
job.output.residual[-1]


# In[ ]:


job.potential


# In[ ]:


job.plot_final_potential()


# Acces the plotting interface common to the different fitting codes usable via pyiron

# In[ ]:


plots = job.plot


# In[ ]:


plt.figure(figsize=(14,7)) # Increase the size a bit
plots.energy_scatter_histogram()


# In[ ]:


plt.figure(figsize=(14,7))
plots.force_scatter_histogram()


# In[ ]:


plt.figure(figsize=(14,7))
plots.force_scatter_histogram(axis=0)


# ### Short Test

# In[ ]:


lmp = pr.create.job.Lammps("Al", delete_existing_job=True)
lmp.structure = pr.create.structure.ase.bulk("Al", cubic=True)
lmp.potential = job.lammps_potential
lmp.calc_minimize(pressure=0)
lmp.run()


# In[ ]:


lmp.get_structure()


# In[ ]:


ref = pr.create.job.Lammps("AlRef")
ref.structure = lmp.get_structure()
ref.potential = job.lammps_potential

murn = pr.create.job.Murnaghan("AlMurn", delete_existing_job=True)
murn.ref_job = ref
murn.run()


# In[ ]:


murn.plot()


# In[ ]:


murn.fit_polynomial()


# ### Improving the potential:
# - try with different starting parameters / global optimization
# - change weights of structures and fit properties
# - change functions used

# In[ ]:




