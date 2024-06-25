# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:56:57 2024

@author: caldera marco
"""

from menura_utils import *

from scipy import ndimage

path = f'all_B'
remote_label = 'jean-zay'
path_remote = f''
it=1300

md = menura_data(path, path_remote, it, ask_scp=False, remote_label=remote_label,
                 print_param=False, full_scp=False)
x = md.edges_x[2:-3]
y = md.edges_y[2:-3]
z = md.edges_z[2:-3]

#md.plt_field('density', plane='xy', log=True, vminmax=[-1, 1], save_fig=True, idx_cut=165)

#md.plt_field('B', plane='xy', log=True, save_fig=True, vminmax=[-1, 1], B_lines=True, save_as_pdf=False)

B=md.load_field('B')

Bx_mean=np.mean(B[0,2:402,2:402,2:402])
By_mean=np.mean(B[1,2:402,2:402,2:402])
Bz_mean=np.mean(B[2,2:402,2:402,2:402])

#definisco componenti del campo magnetico

Bx=B[0,2:402,2:402,2:402]
By=B[1,2:402,2:402,2:402]
Bz=B[2,2:402,2:402,2:402]


Bx_rot = ndimage.rotate(Bx[:,:,0], 45)
By_rot = ndimage.rotate(By[:,:,0], 45)

print(Bx_rot)


plt.xlabel('x')
plt.ylabel('y')
plt.title('Campo magnetico nel piano xy (z=0)')
plt.streamplot(x,y,Bx_rot[:,:],By_rot[:,:])
plt.show()




