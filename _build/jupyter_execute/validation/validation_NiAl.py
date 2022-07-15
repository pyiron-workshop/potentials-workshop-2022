#!/usr/bin/env python
# coding: utf-8

# # **Workshop: From electrons to phase diagrams**
# 
# # Day 2: Validation of the potentials (draft)

# ## Import the fitted potentials for Li-Al ( from prevoius excercise)

# In[1]:


import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pandas as pd
import time


# In[2]:


time_start =  time.process_time()

time_start


# In[ ]:





# In[3]:


from pyiron_atomistics import Project
import pyiron_gpl
from ase.lattice.compounds import B2
from pyiron_atomistics import ase_to_pyiron


# In[4]:


# from structdbrest import StructDBLightRester
# rest = StructDBLightRester(token="workshop2021")


# In[5]:


import matplotlib.pylab as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, PowerNorm

linewidth = 3
axis_width = linewidth - 1

# Figure parameters
mpl.rcParams["figure.titlesize"] = 38
mpl.rcParams["figure.figsize"] = 10, 8
mpl.rcParams["figure.subplot.wspace"] = 0.6
mpl.rcParams["figure.subplot.hspace"] = 0.6

# Line parameters
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['lines.markeredgewidth'] = linewidth
mpl.rcParams['lines.markersize'] = linewidth

# Font parameters
#mpl.rcParams['font.family'] = 'Times New Roman'

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 20

# Latex params
#mpl.rcParams['text.usetex'] = True

# Axes parameters
mpl.rcParams['axes.linewidth'] = axis_width
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 20

# Tick parameters
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.major.size"] = 2 * linewidth + 1
mpl.rcParams["xtick.minor.size"] = linewidth
mpl.rcParams["xtick.major.width"] = axis_width
mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.size"] = 2 * linewidth + 1
mpl.rcParams["ytick.minor.size"] = linewidth
mpl.rcParams["ytick.major.width"] = axis_width
mpl.rcParams["ytick.labelsize"] = 20
mpl.rcParams["ytick.direction"] = "in"

# Grid parameters
mpl.rcParams["grid.linewidth"] = axis_width       ## in points

# Legend parameters
mpl.rcParams["legend.framealpha"] = 1
mpl.rcParams["legend.edgecolor"] = "k"
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["legend.fontsize"] = 20

# Mathtext parameters parameters
mpl.rcParams["mathtext.fontset"] = "stix"

#del mpl.font_manager.weight_dict['roman']
#mpl.font_manager._rebuild()


# In[6]:


pr = Project("validation_test")
####pr.remove_jobs(silently=True, recursive=True)


# In[7]:


##########! rm -rf validation_test


# ## Generate list of potentials (after fitting)

# In[8]:


dummy = pr.create.job.Lammps("dummy")
dummy.structure = pr.create_ase_bulk("Al", cubic=True)
dummy.structure[0] = "Ni"
potential_list = dummy.list_potentials()[:3]


# In[9]:


potential_list


# ## Iterate over all potentials and all possible phases

# In[10]:


struct_dict = dict()
struct_dict["Al"] = dict()
struct_dict["Al"]["s_murn"] = ["fcc", "bcc", "sc"]
struct_dict["Al"]["a"] = 4.04

struct_dict["Ni"] = dict()
struct_dict["Ni"]["s_murn"] = ["fcc", "bcc", "sc"]
struct_dict["Ni"]["a"] = 3.5


struct_dict["NiAl"] = dict()
struct_dict["NiAl"]["s_murn"] = ["B2"]
struct_dict["NiAl"]["a"] = 3.7



struct_dict


# In[11]:


def clean_project_name(name):
    return name.replace("-", "_").replace(".", "_")


# ### Ground state: E-V curves

# In[12]:


for pot in potential_list:
    with pr.open(clean_project_name(pot)) as pr_pot:
        for compound, compound_dict in struct_dict.items():
            for crys_structure in compound_dict["s_murn"]:
                
                # Relax structure
                if crys_structure == "B2":
                    basis = ase_to_pyiron(B2(["Ni", "Al"], latticeconstant=compound_dict["a"]))
                else:
                    basis = pr_pot.create_ase_bulk(compound, crys_structure, a=compound_dict["a"])
                job_relax = pr_pot.create_job(pr_pot.job_type.Lammps, f"{compound}_{crys_structure}_relax", delete_existing_job=True)

                job_relax.structure = basis
                job_relax.potential = pot
                job_relax.calc_minimize(pressure=0)
                job_relax.run()
                
                # Murnaghan
                job_ref = pr_pot.create_job(pr_pot.job_type.Lammps, f"ref_job_{compound}_{crys_structure}")
                job_ref.structure = job_relax.get_structure(-1)
                job_ref.potential = pot
                job_ref.calc_minimize()
                murn_job = job_ref.create_job(pr_pot.job_type.Murnaghan, f"murn_job_{compound}_{crys_structure}")
                murn_job.input["vol_range"] = 0.1
                murn_job.run()


# In[13]:


##pr.remove_jobs(recursive=True)


# In[14]:


# Define functions to get data

# Only work with Murnaghan jobs
def get_only_murn(job_table):
    return (job_table.hamilton == "Murnaghan") & (job_table.status == "finished") 

def get_eq_vol(job_path):
    return job_path["output/equilibrium_volume"]

def get_eq_lp(job_path):
    return np.linalg.norm(job_path["output/structure/cell/cell"][0]) * np.sqrt(2)

def get_eq_bm(job_path):
    return job_path["output/equilibrium_bulk_modulus"]

def get_potential(job_path):
    return job_path.project.path.split("/")[-3]

def get_eq_energy(job_path):
    return job_path["output/equilibrium_energy"]

def get_n_atoms(job_path):
    return len(job_path["output/structure/positions"])


def get_potential(job_path):
    return job_path.project.path.split("/")[-2]

def get_crystal_structure(job_path):
    return job_path.job_name.split("_")[-1]

def get_compound(job_path):
    return job_path.job_name.split("_")[-2]


# In[15]:


# Compile data using pyiron tables
table = pr.create_table("table_murn", delete_existing_job=True)
table.convert_to_object = True
table.db_filter_function = get_only_murn
table.add["potential"] = get_potential
table.add["compound"] = get_compound
table.add["crystal_structure"] = get_crystal_structure
table.add["a"] = get_eq_lp
table.add["eq_vol"] = get_eq_vol
table.add["eq_bm"] = get_eq_bm
table.add["eq_energy"] = get_eq_energy
table.add["n_atoms"] = get_n_atoms
table.run()
data_murn = table.get_dataframe()


# In[16]:


pr.job_table(status="finished");


# ## Elastic constants and Phonons

# In[17]:


for pot in potential_list:
    group_name = clean_project_name(pot)
    pr_pot = pr.create_group(group_name)
    print(pot)
    
    for _, row in data_murn[data_murn.potential==group_name].iterrows():
        job_id = row.job_id
        
        job_ref = pr_pot.create_job(pr_pot.job_type.Lammps, f"ref_job_{row.compound}_{row.crystal_structure}")
        ref = pr_pot.load(job_id)
        job_ref.structure = ref.structure
        job_ref.potential = pot
        job_ref.calc_minimize()
        elastic_job = job_ref.create_job(pr_pot.job_type.ElasticMatrixJob, f"elastic_job_{row.compound}_{row.crystal_structure}")
        elastic_job.input["eps_range"] = 0.05
        elastic_job.run()
        
        
        phonopy_job = job_ref.create_job(pr_pot.job_type.PhonopyJob, f"phonopy_job_{row.compound}_{row.crystal_structure}")
        job_ref.calc_static()
        phonopy_job.run()


# In[18]:


def filter_elastic(job_table):
    return (job_table.hamilton == "ElasticMatrixJob") & (job_table.status == "finished")

# Get corresponding lattice constants
def get_c11(job_path):
    return job_path["output/elasticmatrix"]["C"][0, 0]

def get_c12(job_path):
    return job_path["output/elasticmatrix"]["C"][0, 1]

def get_c44(job_path):
    return job_path["output/elasticmatrix"]["C"][3, 3]


# In[19]:


table = pr.create_table("table_elastic", delete_existing_job=True)
table.db_filter_function = filter_elastic
table.add["potential"] = get_potential
table.add["C11"] = get_c11
table.add["C12"] = get_c12
table.add["C44"] = get_c44
table.add["compound"] = get_compound
table.add["crystal_structure"] = get_crystal_structure

table.run()
data_elastic = table.get_dataframe()
data_elastic


# ### Visualization of the results

# In[20]:


data_murn


# In[21]:


data_elastic


# In[22]:


df_ground_state = pd.merge(on=["potential", "compound", "crystal_structure"], left=data_murn, right=data_elastic, suffixes=('_murn', '_elastic'))
df_ground_state["phase"] = df_ground_state.compound + "_" + df_ground_state.crystal_structure
df_ground_state


# In[23]:


fig, ax_list = plt.subplots(ncols=len(potential_list), nrows=1, sharex="row", sharey="row")

fig.set_figwidth(20)
fig.set_figheight(6)

color_palette = sns.color_palette("tab10", n_colors=len(df_ground_state.phase.unique()))


for i, pot in enumerate(potential_list):
    
    ax = ax_list[i]
    data = df_ground_state[df_ground_state.potential == clean_project_name(pot)]
    
    for j, (_, row) in enumerate(data.iterrows()):
        
        ax = pr.load(row.job_id_murn).plot(plt_show=False, ax=ax, plot_kwargs={"label": row.phase, "color": color_palette[j]})
    
    ax.set_title(f"{pot}")
    #break
fig.subplots_adjust(wspace=0.1);


# In[24]:


# fig, ax_list = plt.subplots(ncols=len(potential_list), nrows=1, sharex="row", sharey="row")

# fig.set_figwidth(20)
# fig.set_figheight(6)

# color_palette = sns.color_palette("tab10", n_colors=len(df_ground_state.phase.unique()))


# for i, pot in enumerate(potential_list):
    
#     ax = ax_list[i]
#     data = df_ground_state[df_ground_state.potential == clean_project_name(pot)]
    
#     for j, (_, row) in enumerate(data.iterrows()):
        
#         ax = pr.load(row.job_id_murn).plot(plt_show=False, ax=ax, plot_kwargs={"label": f"phase_{j}", "color": color_palette[j]})
    
#     ax.set_title(f"Potential {i}")
#     #break
# fig.subplots_adjust(wspace=0.1);
# plt.savefig("example.jpeg", bbox_inches="tight");


# In[25]:


fig, ax_list = plt.subplots(ncols=len(df_ground_state.phase.unique()), nrows=1, sharex="row", sharey="row")

fig.set_figwidth(20)
fig.set_figheight(5)

color_palette = sns.color_palette("tab10", n_colors=len(df_ground_state.potential.unique()))


for i, phase in enumerate(df_ground_state.phase.unique()):
    
    ax = ax_list[i]
    data = df_ground_state[df_ground_state.phase == phase]
    
    
    
    for j, pot in enumerate(potential_list):
        
        phonopy_job = pr[clean_project_name(pot) + f"/phonopy_job_{phase}"]
    
        thermo = phonopy_job.get_thermal_properties(t_min=0, t_max=800)

        ax.plot(thermo.temperatures, thermo.free_energies, label=pot, color=color_palette[j])
        ax.set_xlabel("Temperatures [K]")
    ax.set_title(f"{phase}")
ax_list[0].set_ylabel("Free energies [eV]")

ax_list[-1].legend()
fig.subplots_adjust(wspace=0.1);


# In[26]:


fig, ax_list = plt.subplots(ncols=len(df_ground_state.phase.unique()), nrows=1, sharex="row", sharey="row")

fig.set_figwidth(20)
fig.set_figheight(5)

color_palette = sns.color_palette("tab10", n_colors=len(df_ground_state.potential.unique()))


for i, phase in enumerate(df_ground_state.phase.unique()):
    
    ax = ax_list[i]
    data = df_ground_state[df_ground_state.phase == phase]
    
    
    
    for j, pot in enumerate(potential_list):
        
        phonopy_job = pr[clean_project_name(pot) + f"/phonopy_job_{phase}"]
    
        thermo = phonopy_job.get_thermal_properties(t_min=0, t_max=800)
        
        ax.plot(phonopy_job["output/dos_energies"], phonopy_job["output/dos_total"], color=color_palette[j], label=pot)
        ax.set_xlabel("Frequency [THz]")
    ax.set_title(f"{phase}")
ax_list[0].set_ylabel("DOS")

ax_list[-1].legend()
fig.subplots_adjust(wspace=0.1);


# In[27]:


# phonopy_job.plot_band_structure()


# In[28]:


fig, ax_list = plt.subplots(ncols=len(df_ground_state.phase.unique()), nrows=len(potential_list), sharey="row")

fig.set_figwidth(25)
fig.set_figheight(12)

color_palette = sns.color_palette("tab10", n_colors=len(df_ground_state.potential.unique()))


for i, phase in enumerate(df_ground_state.phase.unique()):
    
    
    data = df_ground_state[df_ground_state.phase == phase]
    
    
    
    for j, pot in enumerate(potential_list):
        ax = ax_list[j][i]
        phonopy_job = pr[clean_project_name(pot) + f"/phonopy_job_{phase}"]
    
        phonopy_job.plot_band_structure(axis=ax)
        ax.set_ylabel("")
        ax.set_title("")
        ax_list[j][0].set_ylabel("DOS")
    ax_list[0][i].set_title(f"{phase}")
fig.subplots_adjust(wspace=0.1, hspace=0.4);


# In[29]:


time_stop = time.process_time()
print(f"Total run time for the notebook {time_stop - time_start} seconds")


# Todo:
# 
#     - SQS and intermediate ordered phases, layered phases (supply the structures)
#     - Properties of compounds
#     - Split the workflows into several notebooks
#     - Defect formation energies etc.
#     - Link to Sarath's part?? (Thermal expansion using MD/QHA)
#     - Showing that MD works with these potentials

# In[ ]:




