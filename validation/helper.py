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

pot_hdnnp = pd.DataFrame({
    'Name': ['RuNNer-AlLi'],
    'Filename': [['/home/jovyan/workshop_preparation/resources/lammps/potentials/hdnnp/input.nn',
                  '/home/jovyan/workshop_preparation/resources/lammps/potentials/hdnnp/scaling.data',
                  '/home/jovyan/workshop_preparation/resources/lammps/potentials/hdnnp/weights.013.data',
                  '/home/jovyan/workshop_preparation/resources/lammps/potentials/hdnnp/weights.003.data']],
    'Model': ['RuNNer'],
    'Species': [['Al', 'Li']],
    'Config': [['pair_style hdnnp 6.350126526766093 dir "./" showew yes showewsum 0 resetew no maxew 100 cflength 1.8897261328 cfenergy 0.0367493254\n',
                'pair_coeff * * Al Li\n']]
})

pot_ace = pd.DataFrame({
    'Name': ['LiAl_yace'],
    'Filename': [[os.path.abspath("../potentials/03-ACE/AlLi-6gen-18May.yace")]],
    'Model': ["ACE"],
    'Species': [['Al', 'Li']],
    'Config': [['pair_style pace\n', 'pair_coeff * * AlLi-6gen-18May.yace Al Li\n']]
})


potentials_list = [pot_eam, pot_hdnnp, pot_ace]


def get_clean_project_name(pot):    
    if isinstance(pot, str):
        return pot.replace("-", "_").replace(".", "_")
    elif isinstance(pot, pd.DataFrame):
        return pot["Name"][0]
    else:
        raise ValueError("Invalid potential type")
