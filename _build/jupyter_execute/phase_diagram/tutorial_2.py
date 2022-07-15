#!/usr/bin/env python
# coding: utf-8

# <table border="0">
#  <tr>
#     <td style="width:30%"><img src="img/potentials_logo.png" width="100%" align="justify"></td>
#     <td style="width:70%"> <p style="width:100%;color:#B71C1C;font-size:24px;text-align:justify"> From electrons to phase diagrams </p> <p style="width:100%,font-size:16px">Day 03 Hands-on session (Part 3)</td>
#  </tr>
# </table>

# In this notebook, we will use the EAM potential fitted in the previous day and move towards the AlLi phase diagram. We will use many of the methods and tools we discussed in the last sessions and put them together for calculation of phase diagrams. We start with the phase diagram of AlLi from Ref. [[1]](https://doi.org/10.1002/adma.19910031215). 

# <img src="img/alli_phase_diagram.jpg" width="50%" align="justify">

# In the last session, we calculated the melting temperatures of the pure phases Al and Li, thereby arriving at two points on the phase diagram. In this notebook, we will start with the left side of the phase diagram, until $X_{Li} < 0.5$.

# As always, we start by importing the necessary modules.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from pyiron_atomistics import Project
from helpers import *
from calphy.integrators import kb
from tqdm.notebook import tqdm


# Most of the interesting features of the phase diagram at composition of $X_{Li} < 0.5$ lies between the temperature range of 800-1000 K. Therefore, we will calculate the free energy in this range. Similar to the previous session, we will use reversible scaling to obtain the free energy in this temperature range in a single calculation. We will recalculate the free energy of pure Al in FCC lattice and pure Al liquid first.

# In[74]:


pr = Project("phase_diagram")


# ## Pure Al

# ### Solid

# We start by creating an FCC structure. The converged lattice constant has been provided so as to speed up the calculations. 

# In[75]:


structure = pr.create.structure.ase.bulk('Al', cubic=True, a=4.135).repeat(4)


# We can visualise the structure.

# In[76]:


structure.plot3d()


# Our FCC lattice consists of a 5x5x5 supercell consisting of 500 atoms. Now we create a Calphy job in pyiron and assign a potential. 

# In[77]:


job_sol = pr.create.job.Calphy("xp_sol")
job_sol.structure = structure


# The potential has been configured for pure Al.

# In[78]:


job_sol.potential = "Al-atomicrex"


# We have assigned the potential and structure. To speed up the calculations, we will run it on 2 cores.

# In[79]:


job_sol.server.cores = 2


# Now let's use the `calc_free_energy` method.

# In[80]:


job_sol.calc_free_energy(temperature=[800, 1000], 
                     pressure=0, 
                     reference_phase="solid",
                     n_equilibration_steps=5000,
                     n_switching_steps=5000)


# Before we actually run the calculation, let us discuss the various parameters. `temperature` keyword gives the temperature range over which the free energy is to be calculated. Since we provide `[700, 1000]`, the free energy is calculated between this range. `pressure` denotes the pressure of the calculation, we chose 0 in this case. Since we are using a solid FCC lattice, we set `reference_phase` to `"solid"`. This means that the Einstein crystal will be used as the reference system. Finally, we have `n_equilibration_steps` and `n_switching_steps`. `n_equilibration_steps` denotes the number of MD steps over which the system is equilibrated to the required temperature and pressure. `n_switching_steps` are the number of MD steps over which the system is continuously transformed between the given interatomic potential, and the reference Einstein crystal.

# Finally we run the calculation.

# In[81]:


job_sol.run()


# ### Liquid

# Before we look at the output of the previous calculation, we will also calculate the free energy of the liquid phase. For this we can use the same structure as the solid. The Calphy workflow will first superheat the structure, melt it, and then equilibrate to the required temperature and pressure. Therefore the input for the pyiron job looks fairly same.

# In[82]:


job_lqd = pr.create.job.Calphy("xp_lqd")
job_lqd.structure = structure
job_lqd.potential = "Al-atomicrex"
job_lqd.server.cores = 2
job_lqd.calc_free_energy(temperature=[800, 1000], 
                     pressure=0, 
                     reference_phase="liquid",
                     n_equilibration_steps=5000,
                     n_switching_steps=5000)


# The major change in the input is that the `reference_phase` is `"liquid"`, instead of `"solid"`. In this case, the Uhlenbeck-Ford model is used as the reference system instead of the Einstein crystal. Now run the job,

# In[83]:


job_lqd.run()


# Now we can look at the output; and plot the free energies of the two phases and calculate the melting temperature.

# In[84]:


pr


# In[85]:


plt.plot(pr['xp_sol'].output.temperature, pr['xp_sol'].output.energy_free,
        label="Al solid", color='#C62828')
plt.plot(pr['xp_lqd'].output.temperature, pr['xp_lqd'].output.energy_free,
        label="Al liquid", color='#006899')


# Great, finally, we will also calculate the free energy of the Al Li structure. Let's read in the structrue from the given file and plot it.

# In[86]:


AlLi = pr.create.structure.ase.read('LiAl_poscar', format='vasp')


# In[87]:


AlLi.plot3d()


# We are calculating the free energy at zero percent Li, there will replace all the Li atoms with Al.

# In[88]:


AlLi[:] = 'Al'


# In[89]:


AlLi.plot3d()


# Now we run the calculation to calculate the free energy at the same temperature range.

# In[90]:


job_AlLi = pr.create.job.Calphy("xp_alli")
job_AlLi.structure = AlLi
job_AlLi.potential = "Al-atomicrex"
job_AlLi.server.cores = 2
job_AlLi.calc_free_energy(temperature=[800, 1000], 
                     pressure=0, 
                     reference_phase="solid",
                     n_equilibration_steps=5000,
                     n_switching_steps=5000)
job_AlLi.run()


# ## Free energy with composition

# Now we will calculate the free energy of FCC solid, liquid and B32 phases with increasing Li compositions. We will use compositions from 0.1 to 0.5 Li. For the solid structure, we will first create an Al FCC structure, and replace randomly selected atoms with Li. Let's see how we do this.

# In[91]:


structure = pr.create.structure.ase.bulk('Al', cubic=True, a=4.135).repeat(4)


# In[92]:


structure.plot3d()


# Now we assume we need to create 0.1 composition of Li. Therefore, the number of Li atoms needed are:

# In[93]:


comp = 0.1
n_Li = int(comp*len(structure))
n_Li


# Now we randomly replace 50 Al atoms with Li.

# In[94]:


structure[np.random.permutation(len(structure))[:n_Li]] = 'Li'


# In[95]:


structure.plot3d()


# We can see that some Al atoms are now replaced with Li. We also need to create B32 structures of varying compositions. For that we start with the LiAl B32 structure, and replace randomly selected Li atoms with Al, therby reducing the amount of Li in the structure.

# In[96]:


structure = pr.create.structure.ase.read('AlLi_poscar', format='vasp')


# In[97]:


structure.plot3d()


# Once again, find the number of Li atoms that need to replaced.

# In[98]:


n_Li = int((0.5-comp)*len(structure))
n_Li


# Now replace random Li atoms with Al

# In[99]:


rinds = len(structure)//2 + np.random.choice(range(len(structure)//2), n_Li, replace=False)
structure[rinds] = 'Al'


# In[100]:


structure.plot3d()


# Now we have all the necessary components in place. We can simply create a loop over the compositions and run the calculations. We have prepared the functions which create structures that can be used directly.

# In[101]:


for count, comp in tqdm(enumerate([0.1, 0.2, 0.3, 0.4, 0.5])):
    structure_fcc, structure_b32 = create_structures(pr, comp, repetitions=4)
    job_sol = pr.create.job.Calphy("x%d_sol"%count, delete_aborted_job=True)
    job_sol.structure = structure_fcc
    job_sol.potential = "LiAl-atomicrex"
    job_sol.server.cores = 2
    job_sol.calc_free_energy(temperature=[800, 1000], 
                         pressure=0, 
                         reference_phase="solid",
                         n_equilibration_steps=5000,
                         n_switching_steps=5000)
    job_sol.run()
    job_lqd = pr.create.job.Calphy("x%d_lqd"%count, delete_aborted_job=True)
    job_lqd.structure = structure_fcc
    job_lqd.potential = "LiAl-atomicrex"
    job_lqd.server.cores = 2
    job_lqd.calc_free_energy(temperature=[800, 1000], 
                         pressure=0, 
                         reference_phase="liquid",
                         n_equilibration_steps=5000,
                         n_switching_steps=5000)
    job_lqd.run()
    job_alli = pr.create.job.Calphy("x%d_alli"%count, delete_aborted_job=True)
    job_alli.structure = structure_b32
    job_alli.potential = "LiAl-atomicrex"
    job_alli.server.cores = 2
    job_alli.calc_free_energy(temperature=[800, 1000], 
                         pressure=0, 
                         reference_phase="solid",
                         n_equilibration_steps=5000,
                         n_switching_steps=5000)
    job_alli.run()


# ### Pack the project

# In[102]:


pr.pack("LiAl")


# In[103]:


pr = Project("LiAl_analysis")
pr.unpack("LiAl")


# In[104]:


pr


# ## Analysing the results

# Now we can analyse the results of the above calculations. First we create an array of the composition values.

# In[105]:


comp = np.arange(0, 0.6, 0.1)


# For the initial set of analysis, we will choose a temperature of 800 K. 

# In[106]:


temp = 800


# The calculations we ran already has the free energy at all temperatures from 800-1000 K. We need to extract the free energy at the correct temperature. The `helpers.py` file in the folder contains some helper functions for this notebook. We provide a `fe_at` method which can extract the free energy at the required temperature. Let's take a look at the method.

# In[107]:


get_ipython().run_line_magic('pinfo', 'fe_at')


# For for pure Al calculations, and for each composition, we extract the free energy at 800 K of the FCC, liquid and B32 phases.

# In[108]:


fcc = []
b32 = []
lqd = []

fcc.append(fe_at(pr["phase_diagram/xp_sol"], temp))
lqd.append(fe_at(pr["phase_diagram/xp_lqd"], temp))
b32.append(fe_at(pr["phase_diagram/xp_alli"], temp))

for i in range(5):
    fcc.append(fe_at(pr["phase_diagram/x%d_sol"%i], temp))
    lqd.append(fe_at(pr["phase_diagram/x%d_lqd"%i], temp))
    b32.append(fe_at(pr["phase_diagram/x%d_alli"%i], temp))


# Plot the results

# In[109]:


plt.plot(comp, fcc, '-', color="#e58080")
plt.plot(comp, lqd, '-', color="#66cfff")
plt.plot(comp, b32, '-', color="#ffc766")
plt.plot(comp, fcc, 'o', label='fcc', color="#e58080", markeredgecolor="#424242")
plt.plot(comp, lqd, 'o', label='lqd', color="#66cfff", markeredgecolor="#424242")
plt.plot(comp, b32, 'o', label='b32', color="#ffc766", markeredgecolor="#424242")
plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# ### Configurational entropy

# In the above example, we had off stoichiometric compositions, but we did not include configurational entropy for the solid structures. The easiest way to do this, which we will use here, is to employ the ideal mixing assumption. In the case of ideal mixing, the configuration entropy of mixing is given by,
# 
# $$
# S_{mix} = -k_B (x \log(x) + (1-x) \log(1-x))
# $$
# 
# We can add this directly to the free energy.
# 
# For the liquid phase, addition of configurational entropy explicitely is not needed as is it included in the free energy calculations.

# In[110]:


smix = -kb*(comp*np.log(comp) + (1-comp)*np.log(1-comp))
smix[0] = 0


# In[111]:


fcc_mix = np.array(fcc)-temp*smix
b32_mix = np.array(b32)-temp*smix
lqd_mix = np.array(lqd)


# In[112]:


plt.plot(comp, fcc_mix, '-', color="#e58080")
plt.plot(comp, lqd_mix, '-', color="#66cfff")
plt.plot(comp, b32_mix, '-', color="#ffc766")
plt.plot(comp, fcc_mix, 'o', label='fcc', color="#e58080", markeredgecolor="#424242")
plt.plot(comp, lqd_mix, 'o', label='lqd', color="#66cfff", markeredgecolor="#424242")
plt.plot(comp, b32_mix, 'o', label='b32', color="#ffc766", markeredgecolor="#424242")
plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# To obtain the results in a computationally efficient way, we used composition along every 0.1 Li. We can fit a 3rd order polynomial to these points to get a finer grid. We will use `numpy.polyfit` for this purpose.

# Let's first define a finer composition grid

# In[113]:


comp_grid = np.linspace(0, 0.5, 1000)


# Now we fit the free energy values and use this fit to revaluate the free energy on the finer grid.

# In[114]:


fcc_fit = np.polyfit(comp, fcc_mix, 3)
fcc_fe = np.polyval(fcc_fit, comp_grid)

lqd_fit = np.polyfit(comp, lqd_mix, 3)
lqd_fe = np.polyval(lqd_fit, comp_grid)

b32_fit = np.polyfit(comp, b32_mix, 3)
b32_fe = np.polyval(b32_fit, comp_grid)


# Plot the fits

# In[115]:


plt.plot(comp_grid, fcc_fe, '-', color="#e58080")
plt.plot(comp_grid, lqd_fe, '-', color="#66cfff")
plt.plot(comp_grid, b32_fe, '-', color="#ffc766")
plt.plot(comp, fcc_mix, 'o', label='fcc', color="#e58080", markeredgecolor="#424242")
plt.plot(comp, lqd_mix, 'o', label='lqd', color="#66cfff", markeredgecolor="#424242")
plt.plot(comp, b32_mix, 'o', label='b32', color="#ffc766", markeredgecolor="#424242")
plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# Now in order to identify regions we need to construct common tangents to free energy curves. To make this easier, we convert to free energy of mixing relative to the liquid. For this purpose, we subtract a straight line connecting the free energy of the pure Al liquid to the 0.5 Li liquid, from the calculated free energy. This same line will then be subtracted from the fcc and b32 curves. Our `helpers.py` module previously introduced includes helper methods to do this.

# In[116]:


lqd_fe_norm, slope, intercept = normalise_fe(lqd_fe, comp_grid)
fcc_fe_norm = fcc_fe-(slope*comp_grid + intercept)
b32_fe_norm = b32_fe-(slope*comp_grid + intercept)


# In[117]:


plt.plot(comp_grid, fcc_fe_norm, '-', color="#e58080", label='fcc')
plt.plot(comp_grid, lqd_fe_norm, '-', color="#66cfff", label='lqd')
plt.plot(comp_grid, b32_fe_norm, '-', color="#ffc766", label='b32')
plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# ### Common tangent constructions
# 
# With the above curves, we can move on to common tangent constructions to identify regions where the two phases coexist. In order to calculate the common tangent for two curves $f$ and $g$, we can solve the following set of equationns:
# 
# $$
# f^\prime (x_1) = g^\prime (x_1)
# $$
# 
# $$
# \frac{f(x_1) - g(x_2)}{(x_1 - x_2)} = f^\prime (x_1)
# $$
# 
# $x_1$ and $x_2$ are the endpoints of the common tangent.

# The fitting is done using the `fsolve` method from `scipy`. Once again, `helpers.py` module offers a function to do this fitting.

# In[118]:


get_ipython().run_line_magic('pinfo', 'find_common_tangent')


# We will use this method to find the common tangent between the fcc and b32 curves

# In[119]:


ct = find_common_tangent(fcc_fit, b32_fit, [0.0, 0.25])
ct


# Note that we provided the polynomials that describe the free energy curves, obtained from fitting to the function. We have obtained $x_1$ and $x_2$. We can plot the common tangent now.

# In[120]:


plt.plot(comp_grid, fcc_fe_norm, '-', color="#e58080", label='fcc')
plt.plot(comp_grid, lqd_fe_norm, '-', color="#66cfff", label='lqd')
plt.plot(comp_grid, b32_fe_norm, '-', color="#ffc766", label='b32')
plt.plot(ct, [np.polyval(fcc_fit, ct[0])-(slope*ct[0] + intercept),
             np.polyval(b32_fit, ct[1])-(slope*ct[1] + intercept)], color="#424242")
plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# Lets take a moment to analyse this plot. We obtained 0.24 and 0.32 as the end points of the common tangent. From the figure, this means that FCC is the most stable structure upto $x_{Li} = 0.24$. Between $0.24 \leq x_{Li} \leq 0.32$, both fcc and b32 phases coexist. Finally, for $x_{Li} > 0.32$, b32 is the most stable phase. 

# Therefore we have obtained already a slice of the phase diagram at 800 K. We can now put all of this methods together and easily calculate for different temperatures.

# In[121]:


temp = 900

fcc = []
b32 = []
lqd = []
fcc.append(fe_at(pr["phase_diagram/xp_sol"], temp))
lqd.append(fe_at(pr["phase_diagram/xp_lqd"], temp))
b32.append(fe_at(pr["phase_diagram/xp_alli"], temp))
for i in range(5):
    fcc.append(fe_at(pr["phase_diagram/x%d_sol"%i], temp))
    lqd.append(fe_at(pr["phase_diagram/x%d_lqd"%i], temp))
    b32.append(fe_at(pr["phase_diagram/x%d_alli"%i], temp))
fcc_mix = np.array(fcc)-temp*smix
b32_mix = np.array(b32)-temp*smix
lqd_mix = np.array(lqd)
fcc_fit = np.polyfit(comp, fcc_mix, 3)
fcc_fe = np.polyval(fcc_fit, comp_grid)
lqd_fit = np.polyfit(comp, lqd_mix, 3)
lqd_fe = np.polyval(lqd_fit, comp_grid)
b32_fit = np.polyfit(comp, b32_mix, 3)
b32_fe = np.polyval(b32_fit, comp_grid)
lqd_fe_norm, slope, intercept = normalise_fe(lqd_fe, comp_grid)
fcc_fe_norm = fcc_fe-(slope*comp_grid + intercept)
b32_fe_norm = b32_fe-(slope*comp_grid + intercept)
ct = find_common_tangent(fcc_fit, b32_fit, [0.0, 0.25])
print(ct)
plt.plot(comp_grid, fcc_fe_norm, '-', color="#e58080", label='fcc')
plt.plot(comp_grid, lqd_fe_norm, '-', color="#66cfff", label='lqd')
plt.plot(comp_grid, b32_fe_norm, '-', color="#ffc766", label='b32')
plt.plot(ct, [np.polyval(fcc_fit, ct[0])-(slope*ct[0] + intercept),
             np.polyval(b32_fit, ct[1])-(slope*ct[1] + intercept)], color="#424242")
plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# Now we continue this process for temperatures of 950 K.

# In[127]:


temp = 950

fcc = []
b32 = []
lqd = []
fcc.append(fe_at(pr["phase_diagram/xp_sol"], temp))
lqd.append(fe_at(pr["phase_diagram/xp_lqd"], temp))
b32.append(fe_at(pr["phase_diagram/xp_alli"], temp))
for i in range(5):
    fcc.append(fe_at(pr["phase_diagram/x%d_sol"%i], temp))
    lqd.append(fe_at(pr["phase_diagram/x%d_lqd"%i], temp))
    b32.append(fe_at(pr["phase_diagram/x%d_alli"%i], temp))
fcc_mix = np.array(fcc)-temp*smix
b32_mix = np.array(b32)-temp*smix
lqd_mix = np.array(lqd)
fcc_fit = np.polyfit(comp, fcc_mix, 3)
fcc_fe = np.polyval(fcc_fit, comp_grid)
lqd_fit = np.polyfit(comp, lqd_mix, 3)
lqd_fe = np.polyval(lqd_fit, comp_grid)
b32_fit = np.polyfit(comp, b32_mix, 3)
b32_fe = np.polyval(b32_fit, comp_grid)
lqd_fe_norm, slope, intercept = normalise_fe(lqd_fe, comp_grid)
fcc_fe_norm = fcc_fe-(slope*comp_grid + intercept)
b32_fe_norm = b32_fe-(slope*comp_grid + intercept)
plt.plot(comp_grid, fcc_fe_norm, '-', color="#e58080", label='fcc')
plt.plot(comp_grid, lqd_fe_norm, '-', color="#66cfff", label='lqd')
plt.plot(comp_grid, b32_fe_norm, '-', color="#ffc766", label='b32')
ct = find_common_tangent(fcc_fit, lqd_fit, [0.1, 0.3])
print(ct)
ct1 = find_common_tangent(b32_fit, lqd_fit, [0.3, 0.5])
print(ct1)
plt.plot(comp_grid, fcc_fe_norm, '-', color="#e58080", label='fcc')
plt.plot(comp_grid, lqd_fe_norm, '-', color="#66cfff", label='lqd')
plt.plot(comp_grid, b32_fe_norm, '-', color="#ffc766", label='b32')
plt.plot(ct, [np.polyval(fcc_fit, ct[0])-(slope*ct[0] + intercept),
             np.polyval(lqd_fit, ct[1])-(slope*ct[1] + intercept)], color="#424242")
plt.plot(ct1, [np.polyval(b32_fit, ct1[0])-(slope*ct1[0] + intercept),
             np.polyval(lqd_fit, ct1[1])-(slope*ct1[1] + intercept)], color="#424242")



plt.xlabel(r"$x_{Li}$")
plt.ylabel(r"F (eV/atom)")
plt.legend()


# Now let's put together all the common tangent constructions to arrive at the phase diagram.
# 
# Since we used a very small system, and switching times, the obtained phase diagram will not be accurate. The results shown here uses a system size of about 4000 atoms for each structure, and 50 ps of equilibration and switching time.
# 
# Let's first look at the phase diagram with the calculated points.

# <img src="img/phase_diagram_dotted.png" width="50%" align="justify">

# And the phase diagram

# <table border="0">
#  <tr>
#     <td style="width:50%"><img src="img/phase_diagram_calculated.png" width="100%" align="justify"></td>
#     <td style="width:50%"><img src="img/alli_phase_diagram.jpg" width="100%" align="justify"></td>
#  </tr>
# </table>

# ### Further reading
# 
# - [Ryu, Seunghwa, and Wei Cai. “A Gold-Silicon Potential Fitted to the Binary Phase Diagram.” Journal of Physics Condensed Matter 22, no. 5 (2010).](https://doi.org/10.1088/0953-8984/22/5/055401).
# 

# In[ ]:




