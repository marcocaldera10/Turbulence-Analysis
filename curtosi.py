# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:42:04 2024

@author: caldera marco
"""

def curtosi(sf_2_1, sf_4_1, sf_2_2, sf_4_2, x):
    
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(50,250)
    plt.xlim(1,2000)
    plt.xlabel('log(r) [di]')
    plt.ylabel('log(K(r))')
    K1=sf_4_1/(sf_2_1**2)
    K2=sf_4_2/(sf_2_2**2)
    plt.plot(x,K1,label='By')
    plt.plot(x,K2,label='Bz')
    plt.vlines(x=x[0],ymin=30,ymax=300,colors='black', linestyle='dotted')
    plt.vlines(x=x[1],ymin=30,ymax=300,colors='black', linestyle='dotted')
    plt.vlines(x=x[11],ymin=30,ymax=300,colors='black', linestyle='dotted')
    plt.legend()
    plt.show()
