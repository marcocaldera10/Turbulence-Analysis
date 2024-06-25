# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:56:25 2024

@author: caldera marco
"""

import matplotlib.pyplot as plt

def plot_SF (sf_1, sf_2, sf_3, sf_4, sf_5, sf_6, x):
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('log(r) [di]')
    plt.ylabel('log(sf)')
#    plt.title('Funzioni di struttura lungo la direzione z')
#    plt.xlim(1,4000)
    plt.ylim(0.00000001,0.1)
    plt.plot(x,sf_1,label='1')
    plt.plot(x,sf_2,label='2')
    plt.plot(x,sf_3,label='3')
    plt.plot(x,sf_4,label='4')
    plt.plot(x,sf_5,label='5')
    plt.plot(x,sf_6,label='6')
    plt.vlines(x=x[0],ymin=0.00000001,ymax=0.1,colors='black', linestyle='dotted')
    plt.vlines(x=x[1],ymin=0.00000001,ymax=0.1,colors='black', linestyle='dotted')
    plt.vlines(x=x[11],ymin=0.00000001,ymax=0.1,colors='black', linestyle='dotted')
#    plt.vlines(x=x[24],ymin=0.000001,ymax=1,colors='black', linestyle='dotted')
    plt.legend()
    plt.show()
    
    
    #      plt.xscale('log')
    #      plt.yscale('log')
    #      plt.ylim(0.000007,0.1)
    plt.ylim(np.log10(0.0000001),np.log10(0.01))
    #      plt.xlim(-5.5,-1)
          
    plt.xlabel('Log(r) [di]')
    plt.ylabel('Log(sf)')
    plt.plot(np.log10(x),np.log10(sf_1),label='1')
    plt.plot(np.log10(x),np.log10(sf_2),label='2')
    plt.plot(np.log10(x),np.log10(sf_3),label='3')
    plt.plot(np.log10(x),np.log10(sf_4),label='4')
    plt.plot(np.log10(x),np.log10(sf_5),label='5')
    plt.plot(np.log10(x),np.log10(sf_6),label='6')
    #      plt.plot(k_perp,B_spectrum)
    plt.vlines(x=np.log10(x[0]),ymin=np.log10(0.0000001),ymax=np.log10(0.1),colors='black', linestyle='dotted')
    plt.vlines(x=np.log10(x[1]),ymin=np.log10(0.0000001),ymax=np.log10(0.1),colors='black', linestyle='dotted')
    plt.vlines(x=np.log10(x[11]),ymin=np.log10(0.0000001),ymax=np.log10(0.1),colors='black', linestyle='dotted')

    x1=np.arange(np.log10(10),np.log10(100),0.001)
    y1=0.79*x1-3.8
    plt.plot(x1,y1,'k--')
    x2=np.arange(np.log10(10),np.log10(100),0.001)
    y2=1.46*x2-5.4
    plt.plot(x2,y2,'k--')
    x3=np.arange(np.log10(10),np.log10(100),0.001)
    y3=2.02*x3-6.7
    plt.plot(x3,y3,'k--')
    x4=np.arange(np.log10(10),np.log10(100),0.001)
    y4=2.5*x4-7.7
    plt.plot(x4,y4,'k--')
    x5=np.arange(np.log10(10),np.log10(100),0.001)
    y5=3.0*x5-8.6
    plt.plot(x5,y5,'k--')
    x6=np.arange(np.log10(10),np.log10(100),0.001)
    y6=3.4*x6-9.4
    plt.plot(x6,y6,'k--')
    plt.legend()      
    plt.show()
    
