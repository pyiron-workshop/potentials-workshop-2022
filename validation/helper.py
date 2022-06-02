# This file contains some useful helper functions to make the notebooks more readable

import pandas as pd
import os



#file_location = "../potentials/AlLi.eam.fs"

#with open(file_location, "r") as f:
#    lines = f.readlines()
    
pot_eam = pd.DataFrame({
    'Name': ['LiAl_eam'],
    'Filename': [[os.path.abspath("../potentials/AlLi.eam.fs")]],
    'Model': ["EAM"],
    'Species': [['Li', 'Al']],
    'Config': [['pair_style eam/fs\n', 'pair_coeff * * AlLi.eam.fs Li Al\n']]
})

pot_ace = pd.DataFrame({
    'Name': ['LiAl_yace'],
    'Filename': [[os.path.abspath("../potentials/AlLi.yace")]],
    'Model': ["ACE"],
    'Species': [['Al', 'Li']],
    'Config': [['pair_style pace\n', 'pair_coeff * * AlLi.yace Li Al\n']]
})


potentials_list = [pot_eam,pot_ace]


def get_clean_project_name(pot):    
    if isinstance(pot, str):
        return pot.replace("-", "_").replace(".", "_")
    elif isinstance(pot, pd.DataFrame):
        return pot["Name"][0]
    else:
        raise ValueError("Invalid potential type")
