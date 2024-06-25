# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:09:18 2024

@author: caldera marco
"""

import numpy as np
import matplotlib.pyplot as plt

def temperature(B_x,B_y,B_z,stress_0,stress_1,stress_2,stress_3,stress_4,stress_5):
    
    M=len(B_x[:,0])
    N=len(B_x[0,:])
    
    Beta_para_1 = []
    T_1=[]
    T_ratio = np.zeros((M,N))
    P_para_mat = np.zeros((M,N))
    P_perp_mat = np.zeros((M,N))
    B_mat = np.zeros((M,N))
    Beta_para = np.zeros((M,N))
    Beta_perp = np.zeros((M,N))
    Beta = np.zeros((M,N))
    
    for i in range(M):
        for j in range(N):
            
         #   Bx=np.mean(B_x[i-1:i+2,j-1:j+2])
         #   By=np.mean(B_y[i-1:i+2,j-1:j+2])
         #   Bz=np.mean(B_z[i-1:i+2,j-1:j+2])
         
            #for a simpler notation   
            Bx=B_x[i,j]
            By=B_y[i,j]
            Bz=B_z[i,j]
            
            #Pressure tensor definition
            P_xyz = [[stress_0[i,j],stress_3[i,j],stress_4[i,j]],[stress_3[i,j],stress_1[i,j],stress_5[i,j]],[stress_4[i,j],stress_5[i,j],stress_2[i,j]]]
            
            #Values for the R_MFA matrix
            N0 = np.sqrt(Bx**2+By**2+Bz**2)
            N1 = np.sqrt(By**2+Bz**2)
            
            #Magnetic pressure definition
            P_mag = (N0**2)/2  
            
            #R_MFA matrix and its transpose
            R_MFA = [[0,Bz/N1,-By/N1],[-(By**2+Bz**2)/(N0*N1),(Bx*By)/(N0*N1),(Bx*Bz)/(N0*N1)],[Bx/N0,By/N0,Bz/N0]]
            
            R_MFA_T = np.transpose(R_MFA)
            
            #calculation of the new pressure tensor as P_MFA*P_xyz*P_MFA_T
            P = np.dot(R_MFA,P_xyz)
            
            P_MFA = np.dot(P,R_MFA_T)
            
            #I take the parallel component of the pressure
            P_para = P_MFA[2,2]    #pressione cinetica parallela
            
            #I calculate the eigenvalues of the sub-matrix 2x2 of P_MFA
#            P_MFA_perp = P_MFA[:2,:2]
#            P_perp1,P_perp2 = np.linalg.eigvals(P_MFA_perp)  #pressione cinetica perpendicolare
            
            #Perpedicular pressure (same results of the commented lines before)
            P_perp = 0.5*(np.trace(P_xyz)-P_para)
            
            #Values for a Brazil plot
            Beta_para_1.append(P_para/P_mag)
            T_1.append(P_perp/P_para)
            
            #I add the temperature ratio, parallel pressure and perpendicular pressure to each matrix
            T_ratio[i,j] = P_perp/P_para      #calcolo rapporto temperatura
            P_para_mat[i,j] = P_para
            P_perp_mat[i,j] = P_perp
            
            #I compute parallel Beta and perppendicular Beta
            Beta_para[i,j] = P_para/P_mag
            Beta_perp[i,j] = P_perp/P_mag
            
            Beta[i,j]=np.trace(P_xyz)/P_mag
    
    return T_ratio, P_para_mat, P_perp_mat, Beta_para, Beta_perp, Beta, Beta_para_1, T_1
    
    
