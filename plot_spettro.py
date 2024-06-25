# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:30:57 2024

@author: caldera marco
"""

def plot_spettro(k_perp, B_spectrum1, B_spectrum2):
    
#      plt.xscale('log')
#      plt.yscale('log')
#      plt.ylim(0.000007,0.1)
      plt.ylim(np.log10(0.000007),np.log10(0.1))
#      plt.xlim(-5.5,-1)
      
      plt.xlabel('Log(Wave number)')
      plt.ylabel('Log(Magnetic Energy Spectrum)')
      plt.plot(np.log10(k_perp),np.log10(B_spectrum1),color='blue',linestyle='dotted')
      plt.plot(np.log10(k_perp),np.log10(B_spectrum2),color='blue')
#      plt.plot(k_perp,B_spectrum)

      x=np.arange(np.log10(0.01),np.log10(0.20),0.00001)
      y=-5/3*x-5.15
      plt.plot(x,y,color='orange')
      plt.text(-1,-3.3,'-5/3 Log(k)',fontsize=15,color='orange')
      plt.show()
      
      