# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 16:56:37 2024

@author: caldera marco
"""

import numpy as np

def compute_B_spectrum1(Bx,By,Bz,dx,dy,dz):
    
    #definisco il numero di punti griglia nelle direzioni x, y, z

      nx = len(Bx[:,0,0])
      ny = len(By[0,:,0])
      nz = len(Bz[0,0,:])
      
      #definisco lo spazio totale di punti griglia nt
        
      nt = nx
      
      nx_max = int((nx/2))
#      ny_max = int((ny/2))
#      nz_max = int((nz/2))
      
      kx = np.fft.fftfreq(nx,dx)*2*np.pi
      B_spectrum = np.zeros((int(ny*nz),int(nx_max)))
      k_para = np.zeros((int(nx_max)))
      
#      n_max = min(ny_max,nz_max)
      
      #calcolo la trasformata di Fourier delle componenti del campo magnetico
      
      l=0
      
      for j in range(int(ny*nz)):
          
          if (j%566==0 and j!=0):
              l+=1
              
          
          Bxf = np.fft.fft(Bx[:,int(j-l*ny),l])/nt
          Byf = np.fft.fft(By[:,int(j-l*ny),l])/nt
          Bzf = np.fft.fft(Bz[:,int(j-l*ny),l])/nt
          
          Ef = np.zeros((nx))
          Ef = (abs(Bxf))**2+(abs(Byf))**2+(abs(Bzf))**2
          
    #      Ef_correct=np.zeros((ny_max,nz_max))
    #      Ef_correct[0,:nz_max] = Ef[0,:nz_max]
    #      Ef_correct[:ny_max,0] = Ef[:ny_max,0]
    #      Ef_correct [1:ny_max,1:nz_max]= 2*Ef[1:ny_max,1:nz_max]

                   
          #calcolo i numeri d'onda associati alla griglia fisica
          
          
    #      ky = np.fft.fftfreq(ny,dy)*2*np.pi
    #      kz = np.fft.fftfreq(nz,dz)*2*np.pi
                   
        
          
    #      Ef_shift=np.log(np.fft.fftshift(Ef))
    #      kx_shift=np.fft.fftshift(kx)
    #      
    #      plt.xlabel('kx')
    #      plt.ylabel('Energy spectrum in kx plane')
    #      plt.plot(kx_shift[1:],Ef_shift)
    #      plt.show()
          
                  
    #      plt.xlabel('kz')
    #      plt.ylabel('ky')
    #      plt.title('Energy spectrum in ky-kz plane')
    #      plt.contourf(kz_shift[1:],ky_shift[1:],Ef_shift[:,:],cmap="magma",levels=128)
    #      plt.colorbar()
    #      plt.show()

    #      print(nx_max)
          
          Ef[1:-1]*=2
          
    #      print(len(B_spectrum))
          
          for i in range(len(kx)):
              rk=int(i)
              if (rk<nx_max):
                  B_spectrum[j,rk] = B_spectrum[j,rk] + Ef[i]
                  k_para[rk]=kx[i]
      
      
      B_spectrum=np.mean(B_spectrum,axis=0)
      
      plt.xscale('log')
      plt.yscale('log')
#      plt.ylim(0.000000003,0.1)
#      plt.ylim(-11.5,-2)
#      plt.xlim(-6.1,-1)
#      x=np.arange(np.log(0.008),np.log(0.16),0.00001)
#      y=-5/2*x-15
#      plt.plot(x,y)
      plt.xlabel('Parallel wave number [log scale]')
      plt.ylabel('Magnetic Power Spectrum [log scale]')
      plt.plot(k_para,B_spectrum)
      
      plt.show()
      
      return B_spectrum, k_para
      
#rk = int(np.round(np.sqrt(ny*dy*ky[j]**2/(2*np.pi)+nz*dz*kz[l]**2/(2*np.pi))))