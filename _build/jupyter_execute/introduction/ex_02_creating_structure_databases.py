#!/usr/bin/env python
# coding: utf-8

# # Exercise 2: Creating and working with structure databases
# 
# Before the excercise, you should:
# 
# * Finish exercise 1
# 
# The aim of this exercise is to make you familiar with:
# 
# * Creating structure databases and working with them for potential fitting (day 2)

# ## Importing necessary modules and creating a project
# 
# This is done the same way as shown in the first exercise

# In[1]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import os


# In[2]:


from pyiron import Project


# In[3]:


pr = Project("creating_datasets")


# ## Creating a structure "container" from the data
# 
# We now go over the jobs generated in the first notebook to store structures, energies, and forces into a structure container which will later be used for potential fitting
# 
# **Note**: Usually these datasets are created using highly accurate DFT calculations. But for practical reasons, we only demonstrate how to do this using data from LAMMPS calculations (the workflow remain the same)

# Access the project created in exercise 1.  `..` means go up one folder in the directory tree as usual in linux.

# In[4]:


pr_fs = pr["../first_steps"]


# Create a TrainingContainer job (to store structures and databases).

# In[5]:


container = pr.create.job.TrainingContainer('dataset_example')


# ## Add structures from the E-V curves
# 
# For starters, we append structures from the energy volume curves we calculated earlier

# In[6]:


for job in pr_fs["E_V_curve"].iter_jobs(status="finished"):
    container.include_job(job)


# We can obtain this data as a `pandas` table

# In[7]:


container.to_pandas()


# ## Add structures from the MD
# 
# We also add some structures obtained from the MD simulations

# Reloading the MD job.  Indexing a project loads jobs within.

# In[8]:


job_md = pr_fs["lammps_job"]


# We can now iterate over the structures within and add each of them to the container.

# In[9]:


traj_length = job_md.number_of_structures
stride = 10


# By default include_job will fetch the last computation step from the given job
# for other steps you have to explicitly pass which step you want.

# In[10]:


for i in range(0, traj_length, stride):
    container.include_job(job_md, iteration_step=i)


# ## Add some defect structures (vacancies, surfaces, etc)
# 
# It's necessary to also include some defect structures, and surfaces to the training dataset.
# 
# Setup a MD calculation for a structure with a vacancy.

# In[11]:


job_lammps = pr.create.job.Lammps("lammps_job_vac")
job_lammps.structure = pr.create.structure.bulk("Al", cubic=True, a=3.61).repeat([3, 3, 3])


# remove the first atom of the structure to create the vacancy

# In[12]:


del job_lammps.structure[0]
job_lammps.potential = "2005--Mendelev-M-I--Al-Fe--LAMMPS--ipr1"
job_lammps.calc_md(temperature=800, pressure=0, n_ionic_steps=10000)
job_lammps.run()


# Setup a MD calculation for a surface structure

# In[13]:


job_lammps = pr.create.job.Lammps("lammps_job_surf")
job_lammps.structure = pr.create.structure.surface("Al", surface_type="fcc111", size=(4, 4, 8), vacuum=12, orthogonal=True)
job_lammps.potential = "2005--Mendelev-M-I--Al-Fe--LAMMPS--ipr1"
job_lammps.calc_md(temperature=800, pressure=0, n_ionic_steps=10000)
job_lammps.run()


# In[14]:


pr


# We now add these structures to the dataset like we did before.

# In[15]:


for job_md in pr.iter_jobs(status="finished", hamilton="Lammps"):
    stride = 10
    for i in range(0, job.number_of_structures, stride):
        container.include_job(job_md, iteration_step=i)


# We run the job to store this dataset in the pyiron database.  Without running the training container "job" the data will **not** saved!

# In[16]:


container.run()


# In[17]:


pr.job_table()


# ## Reloading the dataset
# 
# This dataset can now be reloaded anywhere to use in the potential fitting procedures

# In[18]:


dataset = pr["dataset_example"]
dataset.to_pandas()


# We can now inspect the data in this dataset quite easily

# In[19]:


struct = dataset.get_structure(10)


# In[20]:


struct.plot3d()


# In[21]:


dataset.plot.energy_volume();


# In[22]:


dataset.plot.forces()


# The datasets used in the potential fitting procedure for day 2 (obtained from accurate DFT calculations) will be accessed in the same way.

# ## Extra Credit
# 
# 1. Add more interesting structures. Ideas:
#     - Dimer, trimers
#     - Cleaving of a bulk structure, i.e. create a super cell and separate the atoms along a chosen plane
#     - high or low pressure MD
#     - Different crystal structures
#     - ...
