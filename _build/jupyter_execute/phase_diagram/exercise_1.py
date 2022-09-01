#!/usr/bin/env python
# coding: utf-8

# # <font style="color:#B71C1C" face="Helvetica" > Exercise 1 </font>

# In[2]:


from helpers import potential_list
from pyiron_atomistics import Project
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


pr = Project('exe2') 


# <font style="color:#B71C1C" face="Helvetica" > Task 1: Calculate free energy of FCC Al at 500 K, 10000 bar </font>

# Use a lattice constant of 4.099 for the FCC structure

# In[ ]:





# <font style="color:#B71C1C" face="Helvetica" > Task 2: Does Al have a solid-solid phase tranformation? </font>
# 
# Calculate the free energy of Al in BCC and FCC structures in the temperature range of 500-1000 K. See if there is a solid-solid phase transformation. Use lattice constant of 3.264 for BCC, and 4.099 for FCC.

# In[7]:





# In[5]:





# Plot the solution

# In[ ]:





# <font style="color:#B71C1C" face="Helvetica" > What about at higher pressures? Check the pressure range of 100 GPa to 300 GPa? What is the most efficient way to run these calculations? </font>
# 
# Use a temperature of 500 K. For FCC, use a 4x4x4 supercell with lattice constant of 3.49825. For BCC, use a 4x4x4 super cell with lattice constant of 2.7765.

# In[ ]:





# In[ ]:





# <font style="color:#B71C1C" face="Helvetica" > Task 3: Calculate melting temperature of Al at 10000 bar </font>

# In[ ]:





# <font style="color:#B71C1C" face="Helvetica" > Task 4: How do select the temperature range to calculate the melting temperature? What happens if the range is too high/low? </font>

# In[ ]:





# <font style="color:#B71C1C" face="Helvetica" > Task 5: How can we compare estimate the error on the calculated melting temperature? </font>

# In[ ]:





# In[ ]:





# <font style="color:#B71C1C" face="Helvetica" > Task 6: Calculate the melting temperature of Li </font>
# 
# Use temperature range of 350-500.  

# In[ ]:




