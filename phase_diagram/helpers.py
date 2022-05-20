import pandas as pd
import os

file_location = "../potentials/AlLi.eam.fs"
pot_al = pd.DataFrame({
    'Name': ['Al_eam'],
    'Filename': [[os.path.abspath(file_location)]],
    'Model': ["EAM"],
    'Species': [['Al']],
    'Config': [['pair_style eam/fs\n', 'pair_coeff * * AlLi.eam.fs Al\n']]
})
pot_li = pd.DataFrame({
    'Name': ['Li_eam'],
    'Filename': [[os.path.abspath(file_location)]],
    'Model': ["EAM"],
    'Species': [['Li']],
    'Config': [['pair_style eam/fs\n', 'pair_coeff * * AlLi.eam.fs Li\n']]
})
pot_alli = pd.DataFrame({
    'Name': ['AlLi_eam'],
    'Filename': [[os.path.abspath(file_location)]],
    'Model': ["EAM"],
    'Species': [['Al', 'Li']],
    'Config': [['pair_style eam/fs\n', 'pair_coeff * * AlLi.eam.fs Al Li\n']]
})

potential_list = [pot_al, pot_li, pot_alli]