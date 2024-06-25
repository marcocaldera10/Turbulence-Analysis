# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:49:41 2024

@author: caldera marco
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_B(Bx,By,Bz,x,y,z,density):
    
    B=np.sqrt(Bx**2+By**2+Bz**2)
    B = np.moveaxis(B,0,1)
    Bx = np.moveaxis(Bx,0,1)
    By = np.moveaxis(By,0,1)
    Bz = np.moveaxis(Bz,0,1)
#    B = np.moveaxis(B,0,1)
    plt.xticks(np.arange(0,2000,500))
#    plt.xticks(np.arange(0,2830,500))
    plt.yticks(np.arange(0,2000,500))
#    plt.yticks(np.arange(0,2830,500))
#    plt.grid()
    plt.xlabel('y [di]')
    plt.ylabel('z [di]')
#    plt.title('Temperature anisotropy in yz plane (log scale)')
#    plt.title('Obliqual Firehose Instability Threshold (log scale)')
#    plt.title('Mirror Instability Threshold (log scale)')
    plt.imshow(np.log10(B),norm='linear',vmin=-1,vmax=1,cmap='seismic',origin='lower',extent=(0,2000,0,2000))
#    plt.imshow(np.log10(temp),norm='linear',cmap='magma',origin='lower',extent=(0,2000,0,2000))
    plt.colorbar()
    
    
    plt.streamplot(z,y,By[:,:],Bz[:,:], linewidth=0.3)
    plt.show()
    
    dens = np.moveaxis(density,0,1)
    plt.xticks(np.arange(0,2000,500))
    plt.yticks(np.arange(0,2000,500))
    plt.xlabel('y [di]')
    plt.ylabel('z [di]')
    plt.imshow(np.log10(dens),norm='linear',vmin=-1,vmax=1,cmap='seismic',origin='lower',extent=(0,2000,0,2000))
    plt.colorbar()
    plt.show()
    
    
    
    
    
    