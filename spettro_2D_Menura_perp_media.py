# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 18:06:08 2024

@author: caldera marco
"""

import numpy as np

def compute_B_spectrum2(Bx,By,Bz,dx,dy,dz):
    
    #definisco il numero di punti griglia nelle direzioni x, y, z

      nx = len(Bx[:,0,0])
      ny = len(By[0,:,0])
      nz = len(Bz[0,0,:])
      
      #definisco lo spazio totale di punti griglia nt
        
      nt = ny*nz
      
      ny_max = int((ny/2))
      nz_max = int((nz/2))
      
      n_max = min(ny_max,nz_max)
      
      #calcolo i numeri d'onda associati alla griglia fisica
      
      ky = np.fft.fftfreq(ny,dy)*2*np.pi
      kz = np.fft.fftfreq(nz,dz)*2*np.pi
      
      B_spectrum = np.zeros((int(nx),int(n_max)))
      k_perp = np.zeros((int(n_max)))
      
      #calcolo la trasformata di Fourier delle componenti del campo magnetico
      
      for i in range(nx):
          
          Bxf = np.fft.fft2(Bx[i,:,:])/nt
          Byf = np.fft.fft2(By[i,:,:])/nt
          Bzf = np.fft.fft2(Bz[i,:,:])/nt
          
          Ef = np.zeros((ny,nz))
          Ef = (abs(Bxf))**2+(abs(Byf))**2+(abs(Bzf))**2
          
    #      Ef_correct=np.zeros((ny_max,nz_max))
    #      Ef_correct[0,:nz_max] = Ef[0,:nz_max]
    #      Ef_correct[:ny_max,0] = Ef[:ny_max,0]
    #      Ef_correct [1:ny_max,1:nz_max]= 2*Ef[1:ny_max,1:nz_max]            
            
          
    #      Ef_shift=np.log(np.fft.fftshift(Ef))
    #      ky_shift=np.fft.fftshift(ky)
    #      kz_shift=np.fft.fftshift(kz)
          
                  
    #      plt.xlabel('kz')
    #      plt.ylabel('ky')
    #      plt.title('Energy spectrum in ky-kz plane')
    #      plt.contourf(kz_shift[1:],ky_shift[1:],Ef_shift[:,:],cmap="magma",levels=128)
    #      plt.colorbar()
    #      plt.show()

    #      print(n_max)
          
          Ef[:,1:-1]*=2

          
    #      print(len(B_spectrum))
          
          for j in range(len(ky)):        
              for l in range(len(kz)):
                  rk = int(np.round(np.sqrt(j**2+l**2)))
                  if (rk<n_max):
                      B_spectrum[i,rk] = B_spectrum[i,rk] + Ef[j,l]
                      k_perp[rk] = np.sqrt((ky[j]**2+kz[l]**2))
      
      
                  
      
      B_spectrum=np.mean(B_spectrum,axis=0)
      
      plt.xscale('log')
      plt.yscale('log')
#      plt.ylim(0.000002,0.1)
#      plt.ylim(-13,-3.5)
#      plt.xlim(-5.5,-1)
#      x=np.arange(np.log(0.01),np.log(0.25),0.00001)
#      y=-5/3*x-11.8
#      plt.plot(x,y)
      plt.xlabel('wave_number K_perp')
      plt.ylabel('Magnetic Power Spectrum [K_perp]')
      plt.plot(k_perp,B_spectrum)
      
      plt.show()
      
      return k_perp, B_spectrum
      
#rk = int(np.round(np.sqrt(ny*dy*ky[j]**2/(2*np.pi)+nz*dz*kz[l]**2/(2*np.pi))))