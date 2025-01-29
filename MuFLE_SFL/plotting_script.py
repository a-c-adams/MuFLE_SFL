# script to load and plot files that are cancerous and normal

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re

folder_path = '/Users/alexadams/Desktop/240322_initial_tissue_Muff/'
results_files = [f for f in os.listdir(folder_path)]

normal_colour = '#0072C6'
abnormal_colour = '#1BA881'

marker_dict = {'loc1': 's', 'loc2': 'd', 'loc3': 'o', 'loc4': 'P', 'loc5': 'p', 'loc6': 'x',
               'loc7': 's', 'loc8': 'd', 'loc9': 'o', 'loc10': 'P', 'loc11': 'p', 'loc12': 'x',
               'loc13': 's', 'loc14': 'd', 'loc15': 'o', 'loc16': 'P', 'loc17': 'p', 'loc18': 'x',
               'loc19': 's', 'loc20': 'd', 'loc21': 'o', 'loc22': 'P', 'loc23': 'p', 'loc24': 'x', 'loc25': 's'}

for file in results_files:

    if file.startswith('N'):

        loc = re.search(r'loc\d+', file).group()

        file_path = folder_path + file
        n_file = pd.read_csv(file_path)

        plt.plot(n_file['Wavelength'], n_file['Spline Int'], color=normal_colour, linewidth=3, label=f'N {loc}',
                     marker=marker_dict[loc], markersize=8, markevery=10, alpha=0.7)


    elif file.startswith('Ab'):

        loc = re.search(r'loc\d+', file).group()

        file_path = folder_path + file
        ab_file = pd.read_csv(file_path)

        plt.plot(ab_file['Wavelength'], ab_file['Spline Int'], color=abnormal_colour, linewidth=3,
                 label=f'Ab {loc}',
                 marker=marker_dict[loc], markersize=8, markevery=10, alpha=0.7)

plt.xlabel('Wavelength (nm)', fontsize=15)
plt.ylabel('Fluorescence lifetime (ns)', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
# plt.ylim([0, 2.6])
# plt.title(f'Sample number: {int(label)}')
plt.tight_layout()
# plt.savefig(f'{label}_lt_mean.png', format='png', dpi=1200)
plt.show()
