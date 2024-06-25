# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 10:26:29 2024

@author: caldera marco
"""

def plot_spettro_para(B_spectrum_para1, B_spectrum_para2, k_para):
    
#      plt.xscale('log')
#      plt.yscale('log')
#      plt.ylim(0.0000028,0.1)
      plt.ylim(np.log10(0.0000028),np.log10(0.1))
#      plt.xlim(-6.1,-1)

      plt.xlabel('Log(Wave number)')
      plt.ylabel('Log(Magnetic Energy Spectrum)')
#      plt.plot(k_para,B_spectrum)
      plt.plot(np.log10(k_para),np.log10(B_spectrum_para1), color='blue',linestyle='dotted')
      plt.plot(np.log10(k_para),np.log10(B_spectrum_para2),color='blue')
      
      x=np.arange(np.log10(0.006),np.log10(0.10),0.00001)
      y=-5/2*x-6.7
      plt.plot(x,y,color='orange')
      plt.text(-1.3,-3.3,'-5/2 Log(k)',fontsize=15,color='orange')
      
      plt.show()