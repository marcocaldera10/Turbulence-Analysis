# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 12:14:59 2024

@author: caldera marco
"""

from menura_utils import *

from scipy import ndimage

path = f'all_pres_ion_it600'
remote_label = 'jean-zay'
path_remote = f''
it=600

md = menura_data(path, path_remote, it, ask_scp=False, remote_label=remote_label,
                 print_param=False, full_scp=False)

stress=md.load_field('pres_ion')

stress_0 = stress[0,2:402,2:402,2:402]
stress_1 = stress[1,2:402,2:402,2:402]
stress_2 = stress[2,2:402,2:402,2:402]
stress_3 = stress[3,2:402,2:402,2:402]
stress_4 = stress[4,2:402,2:402,2:402]
stress_5 = stress[5,2:402,2:402,2:402]

path = f'all_B_it600'
remote_label = 'jean-zay'
path_remote = f''
it=600

md = menura_data(path, path_remote, it, ask_scp=False, remote_label=remote_label,
                 print_param=False, full_scp=False)

x = md.edges_x[3:-2]
y = md.edges_y[3:-2]
z = md.edges_z[3:-2]

#md.plt_field('density', plane='xy', log=True, vminmax=[-1, 1], save_fig=True, idx_cut=165)

#md.plt_field('B', plane='xy', log=True, save_fig=True, vminmax=[-1, 1], B_lines=True, save_as_pdf=False)

B=md.load_field('B')
B_dip=md.load_field('B_dip')

print(stress.shape)

print(B_dip.shape)

x = x - 2.5
y = y - 2.5
z = z - 2.5

#definisco componenti del campo magnetico

Bx=B[0,2:402,2:402,2:402]+B_dip[0,2:402,2:402,2:402]
By=B[1,2:402,2:402,2:402]+B_dip[1,2:402,2:402,2:402]
Bz=B[2,2:402,2:402,2:402]+B_dip[2,2:402,2:402,2:402]

#path = f'all_dens'
#remote_label = 'jean-zay'
#path_remote = f''
#it=1000

#md = menura_data(path, path_remote, it, ask_scp=False, remote_label=remote_label,
               #  print_param=False, full_scp=False)

#d=md.load_field('density')
#dens=d[2:402,2:402,2:402]

#print(dens.shape)
