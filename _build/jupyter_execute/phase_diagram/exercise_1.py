#!/usr/bin/env python
# coding: utf-8

# <table border="0">
#  <tr>
#     <td style="width:30%"><img src="img/potentials_logo.png" width="100%" align="justify"></td>
#     <td style="width:70%"> <p style="width:100%;color:#B71C1C;font-size:24px;text-align:justify"> From electrons to phase diagrams </p> <p style="width:100%,font-size:16px">Day 03 Hands-on session exercise (Part 1)</td>
#  </tr>
# </table>

# In[2]:


from helpers import potential_list
from pyiron_atomistics import Project
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


pr = Project('exe2') 


# ### Task 1: Calculate free energy of FCC Al at 500 K, 10000 bar

# Use a lattice constant of 4.099 for the FCC structure

# In[ ]:





# ### Task 2: Does Al have a solid-solid phase tranformation?
# 
# Calculate the free energy of Al in BCC and FCC structures in the temperature range of 500-1000 K. See if there is a solid-solid phase transformation. Use lattice constant of 3.264 for BCC, and 4.099 for FCC.

# In[7]:





# In[5]:





# Plot the solution

# In[ ]:





# #### What about at higher pressures? Check the pressure range of 100 GPa to 300 GPa? What is the most efficient way to run these calculations? 
# 
# Use a temperature of 500 K. For FCC, use a 4x4x4 supercell with lattice constant of 3.49825. For BCC, use a 4x4x4 super cell with lattice constant of 2.7765.

# In[ ]:





# In[ ]:





# ### Task 3: Calculate melting temperature of Al at 10000 bar

# In[ ]:





# ### Task 4: How do select the temperature range to calculate the melting temperature? What happens if the range is too high/low?

# In[ ]:





# ### Task 5: How can we compare estimate the error on the calculated melting temperature?

# In[ ]:





# In[ ]:





# ### Task 6: Calculate the melting temperature of Li
# 
# Use temperature range of 350-500.  

# In[ ]:




