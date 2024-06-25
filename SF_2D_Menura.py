# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 11:48:34 2024

@author: caldera marco
"""

import numpy as np

# calcolo le funzioni di struttura lungo x e lungo y per un campo magnetico 2D

def SF (B1,B2,Nx):
    
# definisco le dimensioni della griglia
        
    n1 = len(B1[0,:,0])
    n2 = len(B2[0,0,:])
    
    N1 = int(n1/2)
    N2 = int(n2/2)
    
    N = min(N1,N2)
    
# creo i vettori con le funzioni di struttura (fino al terzo ordine) lungo x e lungo y.
# A causa della periodicità delle condizioni al contorno
    
    sf_1_1=np.zeros((Nx,N))
    sf_2_1=np.zeros((Nx,N))
    sf_3_1=np.zeros((Nx,N))
    sf_4_1=np.zeros((Nx,N))
    sf_5_1=np.zeros((Nx,N))
    sf_6_1=np.zeros((Nx,N))
    
    sf_1_2=np.zeros((Nx,N))
    sf_2_2=np.zeros((Nx,N))
    sf_3_2=np.zeros((Nx,N))
    sf_4_2=np.zeros((Nx,N))
    sf_5_2=np.zeros((Nx,N))
    sf_6_2=np.zeros((Nx,N))
    
# creo il vettore che contiene le separazioni spaziali

    for x in range (0,Nx,50):
        
        d1=np.zeros((n1))
        d2=np.zeros((n2))
        
    # creo le matrici che contengono le differenze di campo magnetico lungo le due direzioni
        
        dB1=np.zeros((n1,n2))
        dB2=np.zeros((n1,n2))
        
    # considero tutte le separazioni possibili tra i vari punti lungo le varie direzioni
    # (in questo caso le dimensioni coincidono). 
         
        for r in range (N):
            
    # considero tutta la griglia e valuto le differenze di valore di campo magnetico lungo le 
    # due direzioni
            
            for x2 in range (n2):
                for x1 in range (n1):
                    
    # se supero i limiti della griglia, le condizioni al contorno periodiche impongono di valutare
    # le differenze con i punti dal lato opposto e quindi scalo gli indici della dimensione della griglia
    # nx e ny
                    
                    if (x1+r<n1):
                        dB1[x1,x2]=abs(B1[x,x1+r,x2]-B1[x,x1,x2])
                    else:
                        dB1[x1,x2]=abs(B1[x,x1+r-n1,x2]-B1[x,x1,x2])
                        
                    if (x2+r<n2):
                        dB2[x1,x2]=abs(B2[x,x1,x2+r]-B2[x,x1,x2])
                    else:
                        dB2[x1,x2]=abs(B2[x,x1,x2+r-n2]-B2[x,x1,x2])
                        
    # definisco le funzioni di struttura dei vari ordini per una certa distanza r
    # come media della differenze di campo magnetico registrate
                    
            sf_1_1[x,r]=np.mean(dB1)
            sf_2_1[x,r]=np.mean(dB1**2)
            sf_3_1[x,r]=np.mean(dB1**3)
            sf_4_1[x,r]=np.mean(dB1**4)
            sf_5_1[x,r]=np.mean(dB1**5)
            sf_6_1[x,r]=np.mean(dB1**6)
            
            sf_1_2[x,r]=np.mean(dB2)
            sf_2_2[x,r]=np.mean(dB2**2)
            sf_3_2[x,r]=np.mean(dB2**3)
            sf_4_2[x,r]=np.mean(dB2**4)
            sf_5_2[x,r]=np.mean(dB2**5)
            sf_6_2[x,r]=np.mean(dB2**6)
     
        
    sf_1_1=np.mean(sf_1_1,axis=0)
    sf_2_1=np.mean(sf_2_1,axis=0)
    sf_3_1=np.mean(sf_3_1,axis=0)
    sf_4_1=np.mean(sf_4_1,axis=0)
    sf_5_1=np.mean(sf_5_1,axis=0)
    sf_6_1=np.mean(sf_6_1,axis=0)
    
    sf_1_2=np.mean(sf_1_2,axis=0)
    sf_2_2=np.mean(sf_2_2,axis=0)
    sf_3_2=np.mean(sf_3_2,axis=0)
    sf_4_2=np.mean(sf_4_2,axis=0)
    sf_5_2=np.mean(sf_5_2,axis=0)
    sf_6_2=np.mean(sf_6_2,axis=0)

        
    return sf_1_1, sf_2_1, sf_3_1, sf_4_1, sf_5_1, sf_6_1, sf_1_2, sf_2_2, sf_3_2, sf_4_2, sf_5_2, sf_6_2







