# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:37:17 2024

@author: caldera marco
"""

import numpy as np

def compute_B_spectrum(Bx,By,Bz,dx,dy,dz):
    
    #definisco il numero di punti griglia nelle direzioni x, y, z

#      nx = len(Bx[:,0,0])
      ny = len(By[0,:,0])
      nz = len(Bz[0,0,:])
      
      #definisco lo spazio totale di punti griglia nt
        
      nt = ny*nz
      
      ny_max = int((ny/2))
      nz_max = int((nz/2))
      
      n_max = min(ny_max,nz_max)+1
      
      #calcolo la trasformata di Fourier delle componenti del campo magnetico
      
      Bxf = np.fft.fft2(Bx[0,:,:])/nt
      Byf = np.fft.fft2(By[0,:,:])/nt
      Bzf = np.fft.fft2(Bz[0,:,:])/nt
      
      Ef = np.zeros((ny,nz))
      Ef = (abs(Bxf))**2+(abs(Byf))**2+(abs(Bzf))**2
      
#      Ef_correct=np.zeros((ny_max,nz_max))
#      Ef_correct[0,:nz_max] = Ef[0,:nz_max]
#      Ef_correct[:ny_max,0] = Ef[:ny_max,0]
#      Ef_correct [1:ny_max,1:nz_max]= 2*Ef[1:ny_max,1:nz_max]

               
      #calcolo i numeri d'onda associati alla griglia fisica
      
      ky = np.fft.fftfreq(ny,dy)*2*np.pi
      kz = np.fft.fftfreq(nz,dz)*2*np.pi
      
            
    
      
      Ef_shift=np.log10(np.fft.fftshift(Ef))
      ky_shift=np.fft.fftshift(ky)
      kz_shift=np.fft.fftshift(kz)
      
              
      plt.xlabel('kz')
      plt.ylabel('ky')
      plt.title('Energy spectrum in ky-kz plane')
      plt.contourf(kz_shift,ky_shift,Ef_shift[:,:],cmap="magma",levels=128)
      plt.colorbar()
      plt.show()

      print(n_max)
      
      Ef[:,1:-1]*=2

      B_spectrum = np.zeros((int(n_max)))
      k_perp = np.zeros((int(n_max)))
      
      print(len(B_spectrum))
      
      for j in range(len(ky)):        
          for l in range(len(kz)):
              rk = int(np.round(np.sqrt(j**2+l**2)))
              if (rk<n_max):
                  B_spectrum[rk] = B_spectrum[rk] + Ef[j,l]
                  k_perp[rk] = np.sqrt((ky[j]**2+kz[l]**2))
                  
      
#      plt.xscale('log')
#      plt.yscale('log')
#      plt.ylim(0.00000002,0.1)
      plt.ylim(np.log10(0.00000002),np.log10(0.1))
#      plt.xlim(-5.5,-1)
      
      plt.xlabel('Log [Perpendicular wave number]')
      plt.ylabel('Log [Magnetic Power Spectrum]')
      plt.plot(np.log10(k_perp),np.log10(B_spectrum))
      
      x=np.arange(np.log10(0.008),np.log10(0.22),0.00001)
      y=-5/3*x-5.2
      plt.plot(x,y)
      
      plt.show()
      
#rk = int(np.round(np.sqrt(ny*dy*ky[j]**2/(2*np.pi)+nz*dz*kz[l]**2/(2*np.pi))))
      
      
      
      
      
      
      
      
      
      