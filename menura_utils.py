import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# from scipy import ndimage
# from scipy.signal import butter, lfilter

# from IPython.display import Image
import sys, os
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams['pdf.fonttype'] = 42

import menura_tools as mT
import menura_colours as oC

import numpy as np

import scipy.stats
from scipy.fftpack import fft, fft2, fftn
from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator, LinearNDInterpolator
from scipy.signal import blackman, hamming

import re

import math


e    = 1.602e-19
m_i  = 1.673e-27
m_e  = 9.109e-31
mu0  = 1.257e-6
eps0 = 8.85e-12
k_B  = 1.3806e-23
c    = 299792458.


class Spectrum():

    def __init__(self):

        resolution = 600
        k_min = .01
        k_max = 10.

        self.grid_cart = None
        self.grid_spher = None
        self.dvvv = None
        self.vdf_interp = np.zeros((resolution, resolution))#,resolution))
        self.init_grid_2D((k_min, k_max), resolution)
        self.grid_cart_t = self.grid_cart.copy()
        self.grid_spher_t = self.grid_spher.copy()
        self.nb_counts = np.zeros((resolution, resolution))#,resolution))

    def init_grid_2D(self, k_min_max, resolution):

        edges_rho = np.linspace(k_min_max[0], k_min_max[1], resolution + 1, dtype=np.float32)
        edges_theta = np.linspace(0, 2*np.pi, resolution + 1, dtype=np.float32)
        # edges_phi = np.linspace(0, 2*np.pi, resolution + 1,
        #                      dtype=np.float32)
        edges = [edges_rho, edges_theta]#, edges_phi]
        centers_rho = (edges_rho[:-1] + edges_rho[1:]) * .5
        centers_theta = (edges_theta[:-1] + edges_theta[1:]) * .5
        # centers_phi = (edges_phi[:-1] + edges_phi[1:]) * .5
        self.grid_spher = np.mgrid[centers_rho[0]:centers_rho[-1]:centers_rho.size*1j,
                           centers_theta[0]:centers_theta[-1]:centers_theta.size*1j]#,
                           # centers_phi[0]:  centers_phi[-1]  :centers_phi.size*1j]
        self.grid_spher = self.grid_spher.astype(np.float32)
        self.grid_cart = self.spher2cart(self.grid_spher)
        dRho = centers_rho[1]-centers_rho[0]
        dTheta = centers_theta[1]-centers_theta[0]
        # dPhi = centers_phi[1]-centers_phi[0]
        dv = centers_rho[1]-centers_rho[0]
        self.dvvv = np.ones((resolution, resolution))\
                    * centers_rho[:, None] * dRho * dTheta
        # self.dvvv = np.ones((resolution, resolution, resolution))\
        #             * centers_rho[:, None, None] * dRho * dTheta * dPhi

    def interpolate_cart_2D(self, grid, vdf0, interpolate='near'):

        if interpolate == 'near':
            method_str = 'nearest'
        elif interpolate == 'lin':
            method_str = 'linear'

        if interpolate in ['near', 'lin']:
            if vdf0.ndim==2:
                interpFunc = RegularGridInterpolator( (grid[0,:,0], grid[1,0,:]), vdf0,
                                                        bounds_error=False, method=method_str,
                                                        fill_value=np.nan)
                d = interpFunc(self.grid_cart.reshape(2,-1).T) ## Ugly AF.
                d = d.reshape((self.vdf_interp.shape[0],self.vdf_interp.shape[0]))
                self.vdf_interp = d[:,:]
            elif vdf0.ndim==3:
                interpFunc = RegularGridInterpolator( (grid[0,:,0,0], grid[1,0,:,0], grid[2,0,0,:]), vdf0,
                                                        bounds_error=False, method=method_str,
                                                        fill_value=np.nan)
                d = interpFunc(self.grid_cart.reshape(3,-1).T)
                self.vdf_interp = d.T.reshape(self.vdf_interp.shape)  ## (res,res,res)


    def spher2cart(self, vSpher):
        """DocString
        """

        vCart = np.zeros_like(vSpher)
        if vSpher.shape[0]==3:
            vCart[0] = vSpher[0] * np.sin(vSpher[1]) * np.cos(vSpher[2])
            vCart[1] = vSpher[0] * np.sin(vSpher[1]) * np.sin(vSpher[2])
            vCart[2] = vSpher[0] * np.cos(vSpher[1])
        elif vSpher.shape[0]==2:
            vCart[0] = vSpher[0] * np.cos(vSpher[1])
            vCart[1] = vSpher[0] * np.sin(vSpher[1])

        return vCart

    def cyl2cart(self, vCyl):
        """DocString
        """
        vCart = np.zeros_like(vCyl)
        vCart[0] = vCyl[0] * np.cos(vCyl[1])
        vCart[1] = vCyl[0] * np.sin(vCyl[1])
        vCart[2] = vCyl[2].copy()

        return vCart

    def cart2cyl(self, vCart):
        """DocString
        """
        vCyl = np.zeros_like(vCart)
        vCyl[0] = np.sqrt(vCart[0] ** 2 + vCart[1] ** 2)
        vCyl[1] = np.arctan2(vCart[1], vCart[0])
        vCyl[2] = vCart[2].copy()
        itm = (vCyl[1] < 0.)
        vCyl[1][itm] += 2 * np.pi

        return vCyl

    def cart2spher(self, vCart):
        """DocString
        """
        vSpher = np.zeros_like(vCart)
        vSpher[0] = np.sqrt(np.sum(vCart ** 2, axis=0))
        vSpher[1] = np.arccos(vCart[2] / vSpher[0])
        vSpher[2] = np.arctan2(vCart[1], vCart[0])
        itm = (vSpher[2] < 0.)
        vSpher[2][itm] += 2*np.pi

        return vSpher





class menura_param():

    def __init__(self, path, file_name='parameters.txt'):

        self.param_dict = {}

        # p = np.loadtxt(f'{path}/parameters.txt', delimiter=',', converters = {0:None}) #, dtype={'names': ('variable_name', 'value'), 'formats': ('S', 'f4')}
        p = np.recfromtxt(f'{path}/products/{file_name}')

        for t in p:
            param_name = t[0].decode('UTF-8')
            if param_name == 'inject_pla_cst':
                para_name = 'exosphere_cst'
            value = t[1]
            self.param_dict[param_name] = value
            setattr(self, param_name, value)

        try:
            self.NB_DIM = int(self.NB_DIM)
        except:
            self.NB_DIM = 2

        try:
            self.mpi_nb_proc_y = int(self.mpi_nb_proc_y)
            try:
                self.mpi_nb_proc_z = int(self.mpi_nb_proc_z)
            except:
                self.mpi_nb_proc_z = 1
            self.new_rank_string_format = True
        except:
            self.mpi_nb_proc_y = int(self.mpi_nb_proc)
            self.mpi_nb_proc_z = 1
            self.new_rank_string_format = False

        try:
            self.mpi_nb_proc_tot = int(self.mpi_nb_proc_tot)
        except:
            self.mpi_nb_proc_tot = self.mpi_nb_proc_y*self.mpi_nb_proc_z
        #print(self.mpi_nb_proc_tot, self.mpi_nb_proc_y, self.mpi_nb_proc_tot)
        # sys.exit()

        self.len_x_cst = int(self.len_x_cst)
        self.len_y_cst = int(self.len_y_cst)
        if self.NB_DIM==3:
            self.len_z_cst = int(self.len_z_cst)
        else:
            self.len_z_cst = 1
        self.len_save_t_cst = int(self.len_save_t_cst)
        self.len_save_x_cst = int(self.len_save_x_cst)
        try:
            self.nb_probes_cst = int(self.nb_probes_cst)
            self.smooth_patch_save_len_cst = int(self.smooth_patch_save_len_cst)
        except:
            pass

        self.dt_low   = self.dt*self.rate_save_t_cst
        self.dx_low   = self.dX*self.rate_save_x_cst
        self.dy_low   = self.dX*self.rate_save_y_cst

        self.e    = 1.602e-19
        self.m_i  = 1.673e-27
        self.m_e  = 9.109e-31
        self.mu0  = 1.257e-6
        self.eps0 = 8.85e-12
        self.k_B  = 1.3806e-23
        self.c    = 299792458.

    def print_physical_parameters(self):

        print('\n._____________________________________________________________________________________')
        print('| Physical run parameters:\n|    ')
        if self.mpi_nb_proc_tot==1:
            print(f'| 1 process.\n|')
        else:
            print(f'| {self.mpi_nb_proc_tot} processes.\n|')
        try:
            print(f'| Dimensionality:                  {self.NB_DIM}D \n|')
        except: pass
        print(f'| B_0:                             {self.B0_SI:9.2e} T')
        print(f'| n_0:                             {self.n0_SI:9.2e} m^-3')
        print(f'| T_eon:                           {self.Te0_SI:9.2e} K')
        print(f'| T_ion:                           {self.Ti0_SI:9.2e} K\n|')
        print(f'| Proton plasma frequency:         {self.omega_i:9.4f} rad/s')
        print(f'| Proton gyrofrequency:            {self.omega_ci:9.4f} rad/s')
        print(f'| Electron plasma frequency:       {self.omega_e:9.4e} rad/s')
        print(f'| Electron gyrofrequency:          {self.omega_ce:9.4f} rad/s')
        print(f'|')
        print(f'| Proton inertial length d_i :     {self.d_i*1e-3:9.4f} km')
        print(f'| Electron inertial length d_e :   {self.d_e*1e-3:9.4f} km')
        print(f'| Debye length lambda_D :          {self.lambda_D*1e-3:9.4f} km')
        print(f'| Gyro-radius (v_th) :             {self.v_thi_SI/self.v_A:9.4f}')
        print(f'|')
        print(f'| Alfven speed :                   {self.v_A*1e-3:9.4f} km/s')
        print(f'| Sound speed :                    {self.v_s*1e-3:9.4f} km/s')
        print(f'| Magnetosonic speed :             {self.v_ms*1e-3:9.4f} km/s')
        print(f'| Ion thermal speed :              {self.v_thi_SI*1e-3:9.4f} km/s')
        print(f'| Electron thermal speed :         {self.v_the_SI*1e-3:9.4f} km/s')
        print(f'|')
        print(f'| beta0 :                         {self.beta0:9.4f}')
        print(f'| betae0 :                         {self.betae0:9.4f}')
        print(f'|\n|')
        print(f'| Particle-per-node:   {self.nb_part_node_cst}')
        print(f'| Number of particles: {self.pool_size_cst}')
        print(f'| Box length ({self.len_x_cst} x {self.len_y_cst} nodes): {self.len_x_cst*self.dX:.4e} ({self.len_x_cst*self.dX:.4f} proton inertial lengths d_i)')
        print(f'|')
        print(f'| Node spacing: {self.dX:.3e} (d_i0).')
        print(f'| dx >> {self.d_e/self.d_i:.1e} (d_e)')
        print(f'|')
        print(f'| Time step: {self.dt:.3e} ({self.dt/(2.*np.pi):.3e} gyroperiod).')
        print(f'| dt < {1/(np.sqrt(self.NB_DIM)*np.pi) * (self.dX*self.dX):.1e} (CFL).')
        print(f'| dt < {self.dX/(self.v_thi_SI/self.v_A):.1e} (No cell jump at v_thi_SI).')
        print(f'| dt < {self.dX/(self.v_s/self.v_A):.1e} (No cell jump at v_s).')
        print(f'| dt < {self.dX:.1e} (No cell jump at v_A).')
        try:
            if (self.ORF_cst):
              print(f'| dt < {self.dX/(self.v_obs):.1e} (No cell jump at v_0).')
        except:
            pass

        print('|\n|_____________________________________________________________________________________\n\n\n')





class menura_grid():

    def __init__(self, path, lp):

        self.dx = lp.dX
        self.grid_x = np.arange(lp.len_x_cst+4)*self.dx
        self.len_x = self.grid_x.size
        self.grid_y = np.arange(lp.len_y_cst)*self.dx
        self.len_y = self.grid_y.size
        self.grid = np.ones((2, self.grid_x.size, self.grid_y.size))
        self.grid[0] *= self.grid_x[:, None]
        self.grid[1] *= self.grid_y[None, :]
        self.edges_x = np.append(self.grid_x-self.dx/2, self.grid_x[-1]+self.dx/2)
        self.edges_y = np.append(self.grid_y-self.dx/2, self.grid_y[-1]+self.dx/2)
        self.grid_x_low = np.arange(lp.len_save_x_cst)

        self.dt = lp.dt
        self.dt_low = lp.dt_low
        self.grid_t_low = np.arange(lp.len_save_t_cst)*self.dt_low
        self.len_t_low = self.grid_t_low.size




class menura_probes:

    def __init__(self, local_path, remote_path, it, remote_label='jean-zay',
                 ask_scp=False, scp_all=True, print_param=False, poi=0):

        self.local_path = local_path
        self.remote_label = remote_label
        self.remote_path = remote_path

        self.poi = poi  ## probe of interest

        if ask_scp:
            self.scp_data(scp_all)

        self.mp = menura_param(f'{local_path}')
        if print_param:
            self.mp.print_physical_parameters()
        self.nb_proc = self.mp.mpi_nb_proc_tot

        self.time = np.arange(self.mp.nb_it_max_cst)*self.mp.dt
        self.len_t = self.time.size

        if self.mp.NB_DIM==2:
            self.B    = np.zeros((3, self.mp.nb_probes_cst, self.len_t))
            self.E    = np.zeros((3, self.mp.nb_probes_cst, self.len_t))
            self.J    = np.zeros((3, self.mp.nb_probes_cst, self.len_t))
            self.Ji   = np.zeros((3, self.mp.nb_probes_cst, self.len_t))
            self.dens = np.zeros((self.mp.nb_probes_cst, self.len_t))
        # elif self.mp.NB_DIM==3:
        #     self.B    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.E    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.dens = np.zeros((self.nb_proc, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.dens_spec0 = np.zeros((self.nb_proc, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.dens_spec1 = np.zeros((self.nb_proc, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.region_ID = np.zeros((self.nb_proc, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.Ji = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.Jtot = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        #     self.poynting = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
        else:
            print('No.')
            sys.exit()

        self.rx = np.zeros((self.mp.nb_probes_cst, self.len_t))
        self.ry = np.zeros((self.mp.nb_probes_cst, self.len_t))

        self.md = menura_data(self.local_path, self.remote_path, it, remote_label='jean-zay',
                              ask_scp=False, print_param=False, full_scp=False)

        self.fig = None
        self.AX = None

        self.load_data()


    def scp_data(self, scp_all=True):

        remote_pswd = ''
        sync = input('scp from remote? (y/[n]) ')

        if sync == 'y':
            if self.remote_label=='kebnekaise':
                remote_log = 'behare@kebnekaise.hpc2n.umu.se'
                remote_pswd = input('Kebnekaise password:')
            elif self.remote_label=='jean-zay':
                remote_log = 'ued64ot@jean-zay.idris.fr'
                remote_pswd = input('Jean-Zay password:')
        if sync == 'y':
            # sys.exit('niktamer')
            if scp_all:
                print(f'scp probes...')
                os.system(f'scp {remote_log}:{self.remote_path}/products/probes/probes_*      {self.local_path}/products/probes')
                # os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/probes/probes_*      {self.local_path}/products/probes')
                # print(f'scp space_time...')
                # os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/B_time_space* {self.local_path}/products')
                print('scp parameters...')
            else:
                os.system(f'scp {remote_log}:{self.remote_path}/products/probes/probes_*{self.poi}*      {self.local_path}/products/probes')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/parameters.txt {self.local_path}/products')
            # os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/grid_*_rank* {self.local_path}/products')

    def load_data(self):

        for p in range(self.mp.nb_probes_cst):
            fn = f'{self.local_path}/products/probes/probes_B_ID_{p}.npy'
            try:
                self.B[:, p] = np.load(fn)
            except:
                print(f'Probes B ID {p} not loaded.')
                if not os.path.exists(fn):
                    print(f'B-field file not found: {fn}')
            try:
                self.E[:, p] = np.load(f'{self.local_path}/products/probes/probes_E_ID_{p}.npy')
            except:
                print(f'E ID {p} not loaded.')
            try:
                self.J[:, p] = np.load(f'{self.local_path}/products/probes/probes_Jtot_ID_{p}.npy')
            except:
                print(f'Jtot ID {p} not loaded.')
            try:
                self.Ji[:, p] = np.load(f'{self.local_path}/products/probes/probes_Ji_ID_{p}.npy')
            except:
                print(f'Ji ID {p} not loaded.')
            try:
                self.dens[p] = np.load(f'{self.local_path}/products/probes/probes_density_ID_{p}.npy')
            except:
                print(f'Density ID {p} not loaded.')

            try:
                self.rx[p] = np.load(f'{self.local_path}/products/probes/probes_positions_x_ID_{p}.npy')
                self.ry[p] = np.load(f'{self.local_path}/products/probes/probes_positions_y_ID_{p}.npy')
            except:
                print(f'Positions tot rank {p} not loaded.')


    def output_probes(self):

        for p in range(self.mp.nb_probes_cst):
            E = self.E[:, p]
            B = self.B[:, p]
            ui = self.Ji[:, p]/self.dens[p]
            ui[0] -= self.mp.v_obs

            E[1] +=  -self.mp.v_obs*B[2]
            E[2] += self.mp.v_obs*B[1]


            d = np.vstack((self.time, B, E, ui, self.J[:, p]))
            d = d[:, ::2]
            h = 'time Bx By Bz Ex Ey Ez uix uiy uiz Jtot_x Jtot_y Jtot_z'
            np.savetxt(f'/home/etienneb/Desktop/probes_pincon/probe_{p}.csv', d.T,
                       fmt='%.7e', header=h)

        pos = np.vstack((self.rx[:, 0], self.ry[:, 0]))
        h = 'rx ry'
        np.savetxt(f'/home/etienneb/Desktop/probes_pincon/probes_positions.csv', pos.T,
                   fmt='%.7e', header=h)

        dens = self.md.load_field('density')
        print(dens.shape, self.md.grid_x_box.shape)
        np.savetxt('/home/etienneb/Desktop/probes_pincon/density_tot.txt', dens[2:-2, 2:-2])
        np.savetxt('/home/etienneb/Desktop/probes_pincon/grid_x.txt', self.md.grid_x_box[2:-2])
        np.savetxt('/home/etienneb/Desktop/probes_pincon/grid_y.txt', self.md.grid_y_box[2:-2])


    def plt_probes_positions(self, show_probe_ID=False):

        B = self.md.load_field('B')
        # field = np.log10( np.sqrt(B[0]**2 + B[1]**2) )
        # vmax = .5
        # vmin = -1.
        # cmap = oC.wb
        # field = B[2]
        field = np.log10(mT.norm(B))
        vmin = -1
        vmax = 1
        ##
        dens = self.md.load_field('density_s0')
        field = np.log10(dens)
        # omega_i  = np.sqrt(e*e*self.mp.n0_SI*dens/(eps0*m_i))
        # d_i      = c/omega_i
        # field = np.log10(d_i/self.mp.d_i)
        # vmin = -.5
        # vmax = .5
        cmap = oC.bwr_2

        fig, ax = plt.subplots(figsize=(14, 14))

        ax.imshow(field.T,
                            vmin=vmin, vmax=vmax,
                            extent=(self.md.edges_x[0], self.md.edges_x[-1], self.md.edges_y[0], self.md.edges_y[-1]),
                            interpolation='none',
                            rasterized=True, cmap=cmap, origin='lower')


        if show_probe_ID:
            for p in range(self.mp.nb_probes_cst):
                ax.text(self.rx[p, 0], self.ry[p, 0], f'{p}')

        ax.plot(self.rx[:, 0], self.ry[:, 0], 'xk')
        # r = 16
        # i = 6
        # j = 2
        # r = 2
        # i = 4
        # j = 1
        p = self.poi
        ax.plot(self.rx[p, 0], self.ry[p, 0], 'ro')

        ax.plot([312.5], [250.], 'x', ms=10)
        # ax.set_xlim([290, 340])
        # ax.set_ylim([220, 270])
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()


    def plt_probes(self):

        p = self.poi
        fig, ax = plt.subplots(figsize=(18, 14))

        ax.plot(self.time, self.B[0, p], c=oC.rgb[0])
        # from scipy.signal import cosine as windowww
        # from scipy.signal import gaussian as windowww
        # from scipy.signal import general_gaussian as windowww
        # ax.plot(self.time, self.B[0, p]*windowww(self.B[0, p].size, .5, 5e2), '--', c=oC.rgb[0], label='Windowing')
        # ax.plot(self.time, self.B[1, p], c=oC.rgb[1])
        # ax.plot(self.time, self.B[2, p], c=oC.rgb[2])

        plt.title(f'Probe ID {p} - B-field')
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()

        fig, ax = plt.subplots(figsize=(18, 14))

        ax.plot(self.time, self.B[0, p]**2 + self.B[1, p]**2, c=oC.rgb[0])

        plt.title(f'Probe ID {p} - B_perp sqr')
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()
        # sys.exit()

        fig, ax = plt.subplots(figsize=(18, 14))

        ax.plot(self.time, self.E[0, p], c=oC.rgb[0])
        ax.plot(self.time, self.E[1, p], c=oC.rgb[1])
        ax.plot(self.time, self.E[2, p], c=oC.rgb[2])

        plt.title(f'Probe ID {p} - E-field')
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()


        EdotJ = np.sum(self.E*self.J, axis=0)

        fig, ax = plt.subplots(figsize=(18, 14))

        ax.plot(self.time, EdotJ[p], c=oC.rgb[0])

        plt.title(f'Probe ID {p} - EdotJ')
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()

        fig, ax = plt.subplots(figsize=(18, 14))

        ax.plot(self.time, self.dens[p], c=oC.rgb[0], label='density')
        # ax.plot(self.time, self.E[0, p], c=oC.rgb[1], label='Ex')
        # ax.plot(self.time, self.B[0, p]**2 + self.B[1, p]**2, c=oC.rgb[2], label='B_perp')

        ax.legend()
        plt.title(f'Probe ID {p} - density')
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()

        rx = self.rx[p, 0]
        ry = self.ry[p, 0]
        print(f'Probe ID {p}')
        print(f'Probe rx={rx}, ry={ry}')

        mpi_rank_y = int(ry/self.mp.dX/self.mp.len_y_cst) ## Which mpi rank along y?
        # ry -= mpi_rank_y*self.mp.len_y_cst*self.mp.dX
        idx_x = int(np.round( rx*self.mp.dX_i + .5) + 1)
        idx_y = int(np.round( ry*self.mp.dX_i + .5) + 1)
        # for l, j in enumerate(self.md.grid_x_box):
        #     print(l, j)
        # print(idx_y, ry)
        # sys.exit()
        # idx_y = 604
        print(f'Probe on proc {mpi_rank_y}, closest to indices (i, j) on the simulation grid: ({idx_x}, {idx_y})')

        idx_it_field = 0
        self.md = menura_data(self.local_path, self.remote_path, idx_it_field, remote_label='jean-zay',
                              ask_scp=False, print_param=False, full_scp=False)

        B = self.md.load_field('B')#[idx_proc, :, :, idx_y]

        fig, ax = plt.subplots(figsize=(18, 14))

        x = np.arange(0, self.mp.len_x_cst*self.mp.nb_it_per_shift_cst+8, 2)+800
        ax.plot(x, B[0, :, idx_y], c=oC.rgb[2], label='Spatial cut in B_x')
        # ax.plot(self.B[mpi_rank_y, 0, i, j, :2000], c=oC.rgb[0], label='Temporal series B_x')
        ax.plot(self.B[0, p, 0:4000], c=oC.rgb[0], label='Temporal series B_x')

        ax.legend()
        plt.title('Probe Bx comparison with spatial field')
        plt.tight_layout()
        mT.set_spines(ax)
        plt.show()


    def plt_spectrum(self, start_p):


        len_t = int(self.mp.len_x_cst*self.mp.nb_it_per_shift_cst)

        idx_sta = int(0.*len_t)
        idx_sto = int(1.*len_t)

        idx_sta_0 = 0
        idx_sto_0 = len_t

        len_t = idx_sto - idx_sta
        len_t_0 = idx_sto_0 - idx_sta_0

        # p = self.poi
        from scipy.signal import general_gaussian as windowww

        PSD_0 = np.zeros(len_t_0)
        windo = windowww(len_t_0, .5, 5e2)#
        # windo = blackman(len_t_0)
        # windo = np.ones(len_t_0)
        #
        for p in np.arange(20):
            sx_0 = self.B[0, 7*20+p, idx_sta_0:idx_sto_0]
            sx_0 -= np.mean(sx_0)
            PSDx = np.fft.fftshift(fft(sx_0*windo))
            PSDx = np.abs(PSDx)**2
            #
            sy_0 = self.B[1, 7*20+p, idx_sta_0:idx_sto_0]
            sy_0 -= np.mean(sy_0)
            PSDy = np.fft.fftshift(fft(sy_0*windo))
            PSDy = np.abs(PSDy)**2
            PSD_0 += PSDx + PSDy
        PSD_0 /= 20

        PSD_1 = np.zeros(len_t)
        windo = windowww(len_t, .5, 8e2)#*blackman(len_t)))
        # windo = np.ones(len_t)
        #
        for p in np.arange(20):
            sx_1 = self.B[0, start_p+p, idx_sta:idx_sto]
            sx_1 -= np.mean(sx_1)
            PSDx = np.fft.fftshift(fft(sx_1*windo))
            PSDx = np.abs(PSDx)**2

            if 1 and p==0:
                sx_1 -= np.mean(sx_1)
                plt.plot(sx_1)
                plt.plot(sx_1*windo)
                plt.show()

            sy_1 = self.B[1, start_p+p, idx_sta:idx_sto]
            sy_1 -= np.mean(sy_1)
            PSDy = np.fft.fftshift(fft(sy_1*windo))
            PSDy = np.abs(PSDy)**2
            PSD_1 += PSDx + PSDy
        PSD_1 /= 20*1

        # dt = self.mp.dt
        dt = self.mp.dX*self.mp.nb_cell_per_shift_cst/self.mp.nb_it_per_shift_cst
        self.time *= self.mp.v_obs
        f = np.fft.fftshift(np.fft.fftfreq(len_t, d=dt))
        f_0 = np.fft.fftshift(np.fft.fftfreq(len_t_0, d=dt))
        #
        omega = f*2.*np.pi
        omega_0 = f_0*2.*np.pi



        dic = np.load(f'/home/etienneb/Models/Menura_own/Data/run_030/spectra.npy', allow_pickle=True).item()
        bin_centres = dic['bin_centres']
        #



        fig, AX = plt.subplots(1, 1, figsize=(16, 16))


        # PSD_0 /= PSD_0[int(len_t/2.):][1]
        # PSD_1 /= PSD_1[int(len_t/2.):][1]
        dic['B_perp_sqr'] *= 8e1
        #
        # x, y = mT.logSlidingWindow(bin_centres, dic['B_perp_sqr'], halfWidth=.05)
        # AX.plot(x, y,  label='B_perp_sqr',  c='k')
        # AX.plot(bin_centres, dic['B_perp_sqr'], c='k', alpha=.2)

        x, y = mT.logSlidingWindow(omega_0[int(len_t_0/2.):], PSD_0[int(len_t_0/2.):], halfWidth=.06)
        AX.plot(x, y, oC.rgb[2], label='Upstream temporal PSD (from probe)')
        AX.plot(omega_0[int(len_t_0/2.):], PSD_0[int(len_t_0/2.):], oC.rgb[2], alpha=.2)


        x, y = mT.logSlidingWindow(omega[int(len_t/2.):], PSD_1[int(len_t/2.):], halfWidth=.06)
        AX.plot(x, y, oC.rgb[0], label='Downstream temporal PSD (from probe)')
        AX.plot(omega[int(len_t/2.):], PSD_1[int(len_t/2.):], oC.rgb[0], alpha=.2)

        if 1: ## Slope guide-lines, for upstream comparison
            #
            # x = np.array([1e-1, 5e0])
            # y = x**(-1) * 3.e2
            # AX[1].plot(x, y, 'k', lw=.5)
            x = np.array([2e-2, 1e0])
            y3 = x**(-5/3) * 1.e2
            AX.plot(x, y3, 'k', lw=.5)
            ##
            # x = np.array([2e-2, 1e0])
            # y3 = x**(-5/3) * 3.e3
            # AX.plot(x, y3, 'k', lw=.5)
            # x = np.array([1e0, 1e1])
            # y3 = x**(-3) * 3.e3
            # AX.plot(x, y3, 'k', lw=.5)
            #
            # x2 = np.array([1.e0, 1e1])
            # y2 = x2**(-5.) * 3e2
            # AX.plot(x2, y2, 'k', lw=.5)
            # x2 = np.array([1.e0, 2e0])
            # y2 = x2**(-4.5) * 9.e-3
            # AX[1].plot(x2, y2, 'k', lw=.5)
            # x = np.array([2e-2, 1e2])
            # y3 = x**(-2.8) * 5.e2
            # AX.plot(x, y3, 'k', lw=.5)
        #
        AX.set_xlabel('omega/omega_ci')
        AX.set_ylabel('PDs')
        AX.legend()
        AX.set_xscale('log')
        AX.set_yscale('log')
        AX.set_xlim([np.pi/(self.md.mp.dX*self.md.mp.len_x_cst), np.pi/self.md.mp.dX])
        AX.set_ylim([0.004, 8.e6])
        # AX.grid()
        #
        mT.set_spines(AX)
        plt.show()







class menura_data:

    def __init__(self, local_path, remote_path, iteration, remote_label='jean-zay',
                 ask_scp=False, print_param=True, full_scp=True):

        self.local_path = local_path
        self.remote_label = remote_label
        self.remote_path = remote_path
        self.idx_it = iteration

        if ask_scp:
            self.scp_data(scp_ohm=False, full_scp=full_scp)

        self.mp = menura_param(f'{local_path}')
        if print_param:
            self.mp.print_physical_parameters()

        self.nb_proc_tot = self.mp.mpi_nb_proc_tot


        self.grid_x_box = (np.arange(self.mp.len_x_cst+4)-2)*self.mp.dX
        self.grid_y_box = (np.arange(self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4)-2)*self.mp.dX
        self.edges_x = centers2edges(self.grid_x_box)
        self.edges_y = centers2edges(self.grid_y_box)
        if self.mp.NB_DIM==3:
            self.grid_z_box = (np.arange(self.mp.len_z_cst*self.mp.mpi_nb_proc_z+4)-2)*self.mp.dX
            self.edges_z = centers2edges(self.grid_z_box)

        self.fig = None
        self.AX = None


    def scp_data(self, scp_ohm=False, full_scp=True):

        remote_pswd = ''
        sync = input('scp from remote? (y/[n]) ')

        if sync == 'y':
            if self.remote_label=='kebnekaise':
                remote_log = 'behare@kebnekaise.hpc2n.umu.se'
                remote_pswd = input('Kebnekaise password:')
            elif self.remote_label=='jean-zay':
                remote_log = 'ued64ot@jean-zay.idris.fr'
                remote_pswd = input('Jean-Zay password:')

        if sync=='ohm':
            print(f'scp E_mot iteration nb. {self.idx_it}...')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/E_mot_it{self.idx_it}_rank*.npy      {self.local_path}/products')
            print(f'scp E_hal iteration nb. {self.idx_it}...')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/E_hal_it{self.idx_it}_rank*.npy      {self.local_path}/products')
            print(f'scp E_amb iteration nb. {self.idx_it}...')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/E_amb_it{self.idx_it}_rank*.npy      {self.local_path}/products')
            print(f'scp E_res iteration nb. {self.idx_it}...')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/E_res_it{self.idx_it}_rank*.npy      {self.local_path}/products')

        if sync == 'y':
            ##
            print('scp parameters...')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/parameters.txt {self.local_path}/products')
            ##
            print(f'scp dens tot  iteration nb. {self.idx_it}...')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/dens_it{self.idx_it}_*   {self.local_path}/products')
            if full_scp:
                ##
                print(f'scp B iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/B_it{self.idx_it}_*      {self.local_path}/products')
                print(f'scp B_dip iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/B_dip_it{self.idx_it}_*      {self.local_path}/products')
                ##
                print(f'scp E iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/E_it{self.idx_it}_*      {self.local_path}/products')
                ##
                #print(f'scp region_ID iteration nb. {self.idx_it}...')
                #os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/region_ID_it{self.idx_it}_*   {self.local_path}/products')
                ##

                print(f'scp dens per spec  iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/dens_spec*_it{self.idx_it}_*   {self.local_path}/products')
                ##
                print(f'scp curr  iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/curr_it{self.idx_it}_*   {self.local_path}/products')
                print(f'scp curr spec 0 iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/flux_spec_0_it{self.idx_it}_*   {self.local_path}/products')
                print(f'scp curr spec 1 iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/flux_spec_1_it{self.idx_it}_*   {self.local_path}/products')
                print(f'scp curr tot iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp -v {remote_log}:{self.remote_path}/products/curr_tot_it{self.idx_it}_*   {self.local_path}/products')
                ##
                print(f'scp stress tensor species 0 iteration nb. {self.idx_it}...')
                os.system(f'scp {remote_log}:{self.remote_path}/products/stress_s0_it{self.idx_it}_*   {self.local_path}/products')
                print(f'scp stress tensor species 1 iteration nb. {self.idx_it}...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/stress_s1_it{self.idx_it}_*   {self.local_path}/products')

                ## More optional:
                # try:
                #     print(f'scp div_B nb. {self.idx_it}...')
                #     os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/div_B_it_{self.idx_it}_*   {self.local_path}/products')
                #     print(f'scp curl_E nb. {self.idx_it}...')
                #     os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/curl_E_it_{self.idx_it}_*   {self.local_path}/products')
                # except:
                #     pass

                # print(f'scp smoothed patches iteration nb. {self.idx_it}...')
                # os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/smoothed_patches_rank*   {self.local_path}/products')
                # print(f'scp space_time...')
                # os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/B_time_space* {self.local_path}/products')

    def load_field(self, label='B'):

        scalar_labels = ['density', 'density_s0', 'density_s1', 'div_B', 'region_ID', 'vacuum_boarder']
        vector_labels = ['B', 'E', 'Ji', 'Jtot', 'curl_E', 'B_dip', 'E_hal', 'Ji_s0', 'Ji_s1']
        matrix_labels = ['stress_s0', 'stress_s1', 'pres_ion']

        if label in scalar_labels:
            dim = 1 ## i.e. an array of scalars.
            if self.mp.NB_DIM==2:
                field = np.zeros((self.mp.len_x_cst+4, self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4))
            elif self.mp.NB_DIM==3:
                field = np.zeros((self.mp.len_x_cst+4, self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4, self.mp.len_z_cst*self.mp.mpi_nb_proc_z+4))
        elif label in vector_labels:
            dim = 3 ## i.e. an array of vectors.
            if self.mp.NB_DIM==2:
                field = np.zeros((3, self.mp.len_x_cst+4, self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4))
            elif self.mp.NB_DIM==3:
                field = np.zeros((3, self.mp.len_x_cst+4, self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4, self.mp.len_z_cst*self.mp.mpi_nb_proc_z+4))
        elif label in matrix_labels:
            dim = 6 ## i.e. an array of vectors.
            if self.mp.NB_DIM==2:
                field = np.zeros((6, self.mp.len_x_cst+4, self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4))
            elif self.mp.NB_DIM==3:
                field = np.zeros((6, self.mp.len_x_cst+4, self.mp.len_y_cst*self.mp.mpi_nb_proc_y+4, self.mp.len_z_cst*self.mp.mpi_nb_proc_z+4))

        else:
            print(f'\nField loading not available for label {label}. Implemented labels are:')
            for l in vector_labels:
                print(l, end=', ')
            for l in scalar_labels:
                print(l, end=', ')
            for l in matrix_labels:
                print(l, end=', ')
            print('\n')
            sys.exit()


        for r_y in range(self.mp.mpi_nb_proc_y):
            for r_z in range(self.mp.mpi_nb_proc_z):

                if self.mp.new_rank_string_format:
                    rank_str = f'rank_{r_y}_{r_z}'
                else:
                    rank_str = f'rank{r_y}'

                if label == 'B':
                    fn = f'{self.local_path}/products/B_it{self.idx_it}_{rank_str}.npy'
                elif label == 'E':
                    fn = f'{self.local_path}/products/E_it{self.idx_it}_{rank_str}.npy'
                elif label == 'E_hal':
                    fn = f'{self.local_path}/products/E_hal_it{self.idx_it}_{rank_str}.npy'
                elif label == 'Ji':
                    fn = f'{self.local_path}/products/Ji_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/curr_it{self.idx_it}_{rank_str}.npy'
                        ## To be removed eventually!
                elif label == 'Ji_s0':
                    fn = f'{self.local_path}/products/flux_s0_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/flux_spec_0_it{self.idx_it}_{rank_str}.npy'
                elif label == 'Ji_s1':
                    fn = f'{self.local_path}/products/flux_s1_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/flux_spec_1_it{self.idx_it}_{rank_str}.npy'
                elif label == 'Jtot':
                    fn = f'{self.local_path}/products/Jtot_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/curr_tot_it{self.idx_it}_{rank_str}.npy'
                elif label == 'density':
                    fn = f'{self.local_path}/products/density_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/dens_it{self.idx_it}_{rank_str}.npy'
                elif label == 'density_s0':
                    fn = f'{self.local_path}/products/counts_s0_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/counts_spec_0_it{self.idx_it}_{rank_str}.npy'
                elif label == 'density_s1':
                    fn = f'{self.local_path}/products/counts_s1_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/counts_spec_1_it{self.idx_it}_{rank_str}.npy'
                elif label == 'div_B':
                    fn = f'{self.local_path}/products/div_B_it_{self.idx_it}_{rank_str}.npy'
                elif label == 'curl_E':
                    fn = f'{self.local_path}/products/curl_E_it{self.idx_it}_{rank_str}.npy'
                elif label == 'region_ID':
                    fn = f'{self.local_path}/products/region_ID_it{self.idx_it}_{rank_str}.npy'
                elif label == 'vacuum_boarder':
                    fn = f'{self.local_path}/products/vacuum_boarder_it{self.idx_it}_{rank_str}.npy'
                elif label == 'B_dip':
                    fn = f'{self.local_path}/products/B_dip_it{self.idx_it}_{rank_str}.npy'
                elif label == 'stress_s0':
                    fn = f'{self.local_path}/products/stress_s0_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/stress_spec_0_it{self.idx_it}_{rank_str}.npy'
                elif label == 'stress_s1':
                    fn = f'{self.local_path}/products/stress_s1_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/stress_spec_1_it{self.idx_it}_{rank_str}.npy'
                elif label == 'pres_ion':
                    fn = f'{self.local_path}/products/pres_ion_it{self.idx_it}_{rank_str}.npy'
                    if not os.path.exists(fn):
                        fn = f'{self.local_path}/products/pres_ion_spec_it{self.idx_it}_{rank_str}.npy'

                if not os.path.exists(fn):
                    print(f'Field file not found:\n{fn}')
                    #
                    filenames = next(os.walk(f'{self.local_path}/products'), (None, None, []))[2]
                    regex = re.compile(r'it\d+')
                    iterations = []
                    for fn in filenames:
                        iteration = regex.findall(fn)
                        if len(iteration) > 0:
                            iteration = int(iteration[0].split('it')[1])
                            if iteration not in iterations:
                                iterations.append(int(iteration))
                    if len(iterations) > 0:
                        print('Available iterations are:')
                        print(np.sort(iterations))
                        print('')
                    else:
                        print('No available product files.\n')
                    # sys.exit()

                if self.mp.NB_DIM==2:
                    if label in ['density_s0', 'density_s1']:
                        try:
                            field[:, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4] += np.load(fn)
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    elif label in ['Ji_s0', 'Ji_s1']:
                        try:
                            field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4] += np.load(fn)
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    elif label in ['curl_E', 'Jtot']:
                        try:
                            field[:, 2:-2, 2+r_y*self.mp.len_y_cst:-2+(r_y+1)*self.mp.len_y_cst+4] = np.load(fn)[:, 2:-2, 2:-2]
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    elif label in ['stress_s0', 'stress_s1','pres_ion']:
                        try:
                            tmp = np.load(fn)
                            tmp[np.isnan(tmp)] = 0  ## For nan values along the boundaries, when the density on that process was zero, and dividing.
                            field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4] += tmp
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    else:
                        if dim == 1:
                            try:
                                field[:, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4] = np.load(fn)
                            except:
                                print(f'Field rank {r_y} {r_z} not loaded.')
                        elif dim == 3:
                            try:
                                field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4] = np.load(fn)
                            except:
                                print(f'Field rank {r_y} {r_z} not loaded.')
                        elif dim == 6:
                            try:
                                field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4] += np.load(fn)
                            except:
                                print(f'Field rank {r_y} {r_z} not loaded.')

                elif self.mp.NB_DIM==3:
                    if label in ['density_s0', 'density_s1']:
                        try:
                            field[:, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] += np.load(fn)
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    elif label in ['Ji_s0', 'Ji_s1']:
                        try:
                            field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] += np.load(fn)
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    elif label in ['curl_E', 'Jtot']:
                        try:
                            field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] = np.load(fn)
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')
                    elif label in ['stress_s0', 'stress_s1', 'pres_ion']:
                        try:
                            tmp = np.load(fn)
                            tmp[np.isnan(tmp)] = 0  ## For nan values along the boundaries, when the density on that process was zero, and dividing.
                            field[:, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] += tmp
                        except:
                            print(f'Field rank {r_y} {r_z} not loaded.')

                    else:
                        if dim == 1:
                            try:
                                field[:, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] = np.load(fn)
                            except:
                                print(f'Field rank {r_y} {r_z} not loaded.')
                        elif dim == 3:
                            try:
                                ntm = np.load(fn)
                                field[0, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] = ntm[0]
                                field[1, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] = ntm[1]
                                field[2, :, r_y*self.mp.len_y_cst:(r_y+1)*self.mp.len_y_cst+4, r_z*self.mp.len_z_cst:(r_z+1)*self.mp.len_z_cst+4] = ntm[2]
                            except:
                                print(f'Field rank {r_y} {r_z} not loaded, array too large for three components?')

        if label == 'density_s0':
            field *= self.mp.w_sw
        elif label == 'density_s1':
            field *= self.mp.w_pla # old version <12-2022
            #field *= self.mp.w_s1
        elif label == 'Ji_s0':
            field *= self.mp.w_sw # old version <12-2022
            #field *= self.mp.w_s0
        elif label == 'Ji_s1':
            field *= self.mp.w_pla # old version <12-2022
            #field *= self.mp.w_s1
        elif label == 'stress_s0':
            field *= self.mp.w_sw # old version <12-2022
            #field *= self.mp.w_s1
        elif label == 'stress_s1':
            field *= self.mp.w_pla # old version <12-2022
            #field *= self.mp.w_s1

        return field



    def mask_body(self, field):


        rsq = (self.grid_x_box[:, None]-self.mp.centre_x*self.mp.len_y_cst*self.mp.dX)**2 + \
              (self.grid_y_box[None, :]-self.mp.centre_y*self.mp.len_y_cst*self.mp.dX)**2 #+ \
        # rsq = (self.grid_x[r, :, None, None]-self.mp.centre_x*self.max_x)**2 + \
        #       (self.grid_y[r, None, :, None]-self.mp.centre_y*self.max_y)**2 + \
        #       (self.grid_z[r, None, None, :]-self.mp.centre_z*self.max_z)**2
        print(self.mp.r_obs)
        itk = rsq<(self.mp.r_obs)**2
        plt.pcolormesh(itk)
        plt.show()
        field[itk] = np.nan
        # field*=0
        return field



    def plt_all(self, save_fig=False):

        print('\nPlot all.\n')



        if self.mp.NB_DIM==2:

            self.fig, self.AX = plt.subplots(3, 4, figsize=(24, 18))
            for ax in self.AX.flatten():
                ax.set_aspect('equal')

            self.fig.canvas.mpl_connect('button_press_event', self.select_field)

            self.fig.plane = 'xy'

            for r in range(self.mp.mpi_nb_proc_y):

                B = self.load_field('B')

                p0 = self.AX[0, 0].pcolormesh(self.edges_x, self.edges_y, B[0].T,
                                    # vmin = np.amin(B[0]), vmax = np.amax(B[0]),
                                    # vmin=-.2+self.mp.B0_x, vmax=.2+self.mp.B0_x,
                                    # vmin = -.6+self.mp.B0_x, vmax = .6+self.mp.B0_x,
                                    vmin=-4., vmax=4.,
                                    cmap=oC.bwr_2, rasterized=True)
                p1 = self.AX[1, 0].pcolormesh(self.edges_x, self.edges_y, B[1].T,
                                    # vmin = np.amin(B[1]), vmax = np.amax(B[1]),
                                    # vmin=-.2+self.mp.B0_y, vmax=.2+self.mp.B0_y,
                                    # vmin = -.6+self.mp.B0_y, vmax = .6+self.mp.B0_y,
                                    vmin=-3., vmax=5.,
                                    cmap=oC.bwr_2, rasterized=True)
                p2 = self.AX[2, 0].pcolormesh(self.edges_x, self.edges_y, B[2].T,
                                    # vmin = np.amin(B[2]), vmax = np.amax(B[2]),
                                    # vmin=-.2+self.mp.B0_z, vmax=.2+self.mp.B0_z,
                                    # vmin = -.6+self.mp.B0_z, vmax = .6+self.mp.B0_z,
                                    vmin=-4., vmax=4.,
                                    cmap=oC.bwr_2, rasterized=True)
                del(B)

                E = self.load_field('E')
                p3 = self.AX[0, 1].pcolormesh(self.edges_x, self.edges_y, E[0].T,
                                    # vmin = np.amin(E[0]), vmax = np.amax(E[0]),
                                    vmin = -10, vmax = 10,
                                    cmap=oC.bwr_2, rasterized=True)
                p4 = self.AX[1, 1].pcolormesh(self.edges_x, self.edges_y, E[1].T,
                                    # vmin = np.amin(E[1]), vmax = np.amax(E[1]),
                                    # vmin = -.6, vmax = .6,
                                    vmin = -10, vmax = 10,
                                    cmap=oC.bwr_2, rasterized=True)
                p5 = self.AX[2, 1].pcolormesh(self.edges_x, self.edges_y, E[2].T,
                                    # vmin = np.amin(E[2]), vmax = np.amax(E[2]),
                                    # vmin = -.6, vmax = .6,
                                    vmin = -10, vmax = 10,
                                    cmap=oC.bwr_2, rasterized=True)
                del(E)

                dens = self.load_field('density')
                p6 = self.AX[0, 2].pcolormesh(self.edges_x, self.edges_y, dens.T,
                                    # vmin = np.amin(dens), vmax = np.amax(dens),
                                    vmin = 0., vmax = 1.2,
                                    cmap=oC.bwr_2, rasterized=True)
                density_s0 = self.load_field('density_s0')
                p7 = self.AX[1, 2].pcolormesh(self.edges_x, self.edges_y, density_s0.T,
                                    vmin = np.amin(density_s0), vmax = np.amax(density_s0),
                                    # vmin=0., vmax=1.2,
                                    cmap=oC.bwr_2, rasterized=True)
                try:
                    if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
                        density_s1 = self.load_field('density_s1')
                        p8 = self.AX[2, 2].pcolormesh(self.edges_x, self.edges_y, np.log10(density_s1).T,
                                            vmin = np.amax(np.log10(self.dens_spec1))-5, vmax = np.amax(np.log10(self.dens_spec1)),
                                            # vmin=-4, vmax=1,
                                            cmap=oC.bwr_2, rasterized=True)
                except:
                    pass
                    # try:
                    #     R = self.mp.R_gyr_norm
                    #     t = np.linspace(0, np.pi, 400)
                    #     xx = -R*(t - np.sin(t)) + 256
                    #     yy = -R*(1 + np.cos(t)) + 512
                    #     self.AX[2, 2].plot(xx, yy, 'k')
                    #     self.AX[2, 2].plot(xx[0], yy[0], 'xk')
                    # except:
                    #     pass
                del(dens)
                del(density_s0)
                try:
                    del(density_s1)
                except:
                    pass

                Ji = self.load_field('Ji')
                p9 = self.AX[0, 3].pcolormesh(self.edges_x, self.edges_y, Ji[0].T,
                                    # vmin = np.nanmin(Ji[0]), vmax = np.nanmax(Ji[0]),
                                    # vmin=-.2, vmax=.2,
                                    # vmin=-1.-self.mp.v_obs, vmax=1.-self.mp.v_obs,
                                    vmin = -10., vmax = 10.,
                                    cmap=oC.bwr_2, rasterized=True)
                p10 = self.AX[1, 3].pcolormesh(self.edges_x, self.edges_y, Ji[1].T,
                                    # vmin = np.amin(Ji[1]), vmax = np.amax(Ji[1]),
                                    # vmin=-.2, vmax=.2,
                                    # vmin = -.6, vmax = .6,
                                    vmin = -10., vmax = 10.,
                                    cmap=oC.bwr_2, rasterized=True)
                p11 = self.AX[2, 3].pcolormesh(self.edges_x, self.edges_y, Ji[2].T,
                                    # vmin = np.amin(Ji[2]), vmax = np.amax(Ji[2]),
                                    # vmin=-.2, vmax=.2,
                                    # vmin = -.6, vmax = .6,
                                    vmin = -10., vmax = 10.,
                                    cmap=oC.bwr_2, rasterized=True)
                del(Ji)

                try:
                    t = np.linspace(0, 2*np.pi, 200)
                    r_obs = self.mp.r_obs
                    for ax in self.AX.flatten():
                        ax.axis('off')
                        if self.mp.solid_body_cst and ax!=self.AX[2, 2]:
                            ax.plot(r_obs*np.cos(t)+self.max_x*self.mp.centre_x+self.mp.dX,
                                    r_obs*np.sin(t)+self.max_y*self.mp.centre_y+self.mp.dX, 'k', lw=2)
                except: pass


                plt.colorbar(p0, ax=self.AX[0, 0])
                plt.colorbar(p1, ax=self.AX[1, 0])
                plt.colorbar(p2, ax=self.AX[2, 0])
                plt.colorbar(p3, ax=self.AX[0, 1])
                plt.colorbar(p4, ax=self.AX[1, 1])
                plt.colorbar(p5, ax=self.AX[2, 1])
                plt.colorbar(p6, ax=self.AX[0, 2])
                plt.colorbar(p7, ax=self.AX[1, 2])
                try:
                    if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
                        plt.colorbar(p8, ax=self.AX[2, 2])
                except:
                    pass
                plt.colorbar(p9, ax=self.AX[0, 3])
                plt.colorbar(p10, ax=self.AX[1, 3])
                plt.colorbar(p11, ax=self.AX[2, 3])
                self.AX[0, 0].text(10, 10, 'Bx(x, y)')
                self.AX[1, 0].text(10, 10, 'By(x, y)')
                self.AX[2, 0].text(10, 10, 'Bz(x, y)')
                self.AX[0, 1].text(10, 10, 'Ex(x, y)')
                self.AX[1, 1].text(10, 10, 'Ey(x, y)')
                self.AX[2, 1].text(10, 10, 'Ez(x, y)')
                self.AX[0, 2].text(10, 10, 'density(x, y)')
                self.AX[1, 2].text(10, 10, 'dens_sw(x, y)')
                self.AX[0, 3].text(10, 10, 'Jix(x, y)')
                self.AX[1, 3].text(10, 10, 'Jiy(x, y)')
                self.AX[2, 3].text(10, 10, 'Jiz(x, y)')



                plt.tight_layout()

                if save_fig:
                    file_name = f'{self.local_path}/products/plots/all_{self.fig.plane}_{self.idx_it:04}.png'
                    # file_name = f'/home/etienneb/Desktop/nikbientamer_SWRF/yo_all_{self.idx_it:04}.png'
                    print(f'Saving {file_name}')
                    plt.savefig(file_name)
                else:
                    plt.show()






        elif self.mp.NB_DIM==3:

            self.fig, self.AX = plt.subplots(3, 4, figsize=(24, 18))
            for ax in self.AX.flatten():
                ax.set_aspect('equal')

            self.fig.canvas.mpl_connect('button_press_event', self.select_field)

            self.fig.plane = 'xy'

            idx_mid_z = int(self.grid_z_box.size/2.)

            B = self.load_field('B')
            p0 = self.AX[0, 0].pcolormesh(self.edges_x, self.edges_y, B[0, :, :, idx_mid_z].T,
                                # vmin = np.nanmin(B[0, :, :, idx_mid_z]), vmax = np.nanmax(B[0, :, :, idx_mid_z]),
                                # vmin=-.2+self.mp.B0_x, vmax=.2+self.mp.B0_x,
                                vmin=-4., vmax=4.,
                                cmap=oC.bwr_2, rasterized=True)
            p1 = self.AX[1, 0].pcolormesh(self.edges_x, self.edges_y, B[1, :, :, idx_mid_z].T,
                                # vmin = np.nanmin(B[1, :, :, idx_mid_z]), vmax = np.nanmax(B[1, :, :, idx_mid_z]),
                                # vmin=-.2+self.mp.B0_y, vmax=.2+self.mp.B0_y,
                                # vmin=-.2, vmax=.2,
                                vmin=-3., vmax=5.,
                                cmap=oC.bwr_2, rasterized=True)
            p2 = self.AX[2, 0].pcolormesh(self.edges_x, self.edges_y, B[2, :, :, idx_mid_z].T,
                                # vmin = np.nanmin(B[2, :, :, idx_mid_z]), vmax = np.nanmax(B[2, :, :, idx_mid_z]),
                                # vmin=-.2+self.mp.B0_z, vmax=.2+self.mp.B0_z,
                                vmin=-4., vmax=4.,
                                cmap=oC.bwr_2, rasterized=True)

            E = self.load_field('E')
            # print(np.amax(E[0]))
            p3 = self.AX[0, 1].pcolormesh(self.edges_x, self.edges_y, E[0, :, :, idx_mid_z].T,
                                # vmin = np.amin(E[0, :, :, idx_mid_z]), vmax = np.amax(E[0, :, :, idx_mid_z]),
                                # vmin=-.6, vmax=.6,
                                vmin = -10, vmax = 10,
                                cmap=oC.bwr_2, rasterized=True)
            p4 = self.AX[1, 1].pcolormesh(self.edges_x, self.edges_y, E[1, :, :, idx_mid_z].T,
                                # vmin = np.amin(E[1, :, :, idx_mid_z]), vmax = np.amax(E[1, :, :, idx_mid_z]),
                                # vmin=-.6, vmax=.6,
                                vmin = -10, vmax = 10,
                                cmap=oC.bwr_2, rasterized=True)
            p5 = self.AX[2, 1].pcolormesh(self.edges_x, self.edges_y, E[2, :, :, idx_mid_z].T,#-14.5455,
                                # vmin = np.amin(E[2, :, :, idx_mid_z]), vmax = np.amax(E[2, :, :, idx_mid_z]),
                                # vmin=-2., vmax=6.,
                                vmin = -10, vmax = 10,
                                cmap=oC.bwr_2, rasterized=True)

            dens = self.load_field('density')
            p6 = self.AX[0, 2].pcolormesh(self.edges_x, self.edges_y, dens[:, :, idx_mid_z].T,
                                # vmin = np.amin(dens), vmax = np.amax(dens),
                                vmin=0., vmax=1.2,
                                cmap=oC.bwr_2, rasterized=True)
            density_s0 = self.load_field('density_s0')
            p7 = self.AX[1, 2].pcolormesh(self.edges_x, self.edges_y, density_s0[:, :, idx_mid_z].T,
                                vmin = np.amin(density_s0), vmax = np.amax(density_s0),
                                # vmin=0., vmax=1.2,
                                cmap=oC.bwr_2, rasterized=True)
            try:
                if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
                    density_s1 = self.load_field('density_s1')
                    p8 = self.AX[2, 2].pcolormesh(self.edges_x, self.edges_y, np.log10(density_s1[:, :, idx_mid_z]).T,
                                        # vmin = np.amin(self.dens_spec1), vmax = np.amax(self.dens_spec1),
                                        # vmin=-4, vmax=1,
                                        cmap=oC.bwr_2, rasterized=True)
            except:
                pass

            Ji = self.load_field('Ji')
            # print('pute ', Ji[0, 70, 0, 0])
            p9 = self.AX[0, 3].pcolormesh(self.edges_x, self.edges_y, (Ji[0, :, :, idx_mid_z]).T,#/self.dens[:, :, idx_mid_z]).T,
                                # vmin = np.amin(Ji[0]), vmax = np.amax(Ji[0]),
                                # vmin=-1.-self.mp.v_obs, vmax=1.-self.mp.v_obs,
                                # vmin=-self.mp.v_obs-1.2, vmax=-self.mp.v_obs+1.2,
                                # vmin=-.2, vmax=.2,
                                vmin = -10, vmax = 10,
                                cmap=oC.bwr_2, rasterized=True)
            p10 = self.AX[1, 3].pcolormesh(self.edges_x, self.edges_y, (Ji[1, :, :, idx_mid_z]).T,#/self.dens[:, :, idx_mid_z]).T,
                                # vmin = np.amin(Ji[1]), vmax = np.amax(Ji[1]),
                                # vmin=-.2, vmax=.2,
                                vmin = -10, vmax = 10,
                                cmap=oC.bwr_2, rasterized=True)
            p11 = self.AX[2, 3].pcolormesh(self.edges_x, self.edges_y, (Ji[2, :, :, idx_mid_z]).T,#/self.dens[:, :, idx_mid_z]).T,
                                # vmin = np.amin(Ji[2]), vmax = np.amax(Ji[2]),
                                # vmin=-.2, vmax=.2,
                                vmin = -10, vmax = 10,
                                cmap=oC.bwr_2, rasterized=True)

            try:
                t = np.linspace(0, 2*np.pi, 200)
                r_obs = self.mp.r_obs
                for ax in self.AX.flatten():
                    ax.axis('off')
                    if self.mp.solid_body_cst and ax!=self.AX[2, 2]:
                        ax.plot(r_obs*np.cos(t)+self.max_x*self.mp.centre_x+self.mp.dX,
                                r_obs*np.sin(t)+self.max_y*self.mp.centre_y+self.mp.dX, 'k', lw=2)
            except: pass


            plt.colorbar(p0, ax=self.AX[0, 0])
            plt.colorbar(p1, ax=self.AX[1, 0])
            plt.colorbar(p2, ax=self.AX[2, 0])
            plt.colorbar(p3, ax=self.AX[0, 1])
            plt.colorbar(p4, ax=self.AX[1, 1])
            plt.colorbar(p5, ax=self.AX[2, 1])
            plt.colorbar(p6, ax=self.AX[0, 2])
            plt.colorbar(p7, ax=self.AX[1, 2])
            try:
                if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
                    plt.colorbar(p8, ax=self.AX[2, 2])
            except:
                pass
            plt.colorbar(p9, ax=self.AX[0, 3])
            plt.colorbar(p10, ax=self.AX[1, 3])
            plt.colorbar(p11, ax=self.AX[2, 3])
            self.AX[0, 0].text(10, 10, 'Bx(x, y)')
            self.AX[1, 0].text(10, 10, 'By(x, y)')
            self.AX[2, 0].text(10, 10, 'Bz(x, y)')
            self.AX[0, 1].text(10, 10, 'Ex(x, y)')
            self.AX[1, 1].text(10, 10, 'Ey(x, y)')
            self.AX[2, 1].text(10, 10, 'Ez(x, y)')
            self.AX[0, 2].text(10, 10, 'density(x, y)')
            self.AX[1, 2].text(10, 10, 'dens_sw(x, y)')
            self.AX[0, 3].text(10, 10, 'Jix(x, y)')
            self.AX[1, 3].text(10, 10, 'Jiy(x, y)')
            self.AX[2, 3].text(10, 10, 'Jiz(x, y)')



            plt.tight_layout()

            if save_fig:
                file_name = f'{self.local_path}/products/plots/all_{self.fig.plane}_{self.idx_it:04}.png'
                # file_name = f'/home/etienneb/Desktop/nikbientamer_SWRF/yo_all_{self.idx_it:04}.png'
                print(f'Saving {file_name}')
                plt.savefig(file_name)
            else:
                plt.show()


            #___________________________________________________________________
            ## XZ-plane
            self.fig, self.AX = plt.subplots(3, 4, figsize=(24, 18))
            for ax in self.AX.flatten():
                ax.set_aspect('equal')

            self.fig.canvas.mpl_connect('button_press_event', self.select_field)
            self.fig.plane = 'xz'

            idx_mid_y = int(self.grid_y_box.size/2.)

            p0 = self.AX[0, 0].pcolormesh(self.edges_x, self.edges_z, B[0, :, idx_mid_y].T,
                                vmin = np.nanmin(B[0, :, :, idx_mid_z]), vmax = np.nanmax(B[0, :, :, idx_mid_z]),
                                # vmin=-.2+self.mp.B0_x, vmax=.2+self.mp.B0_x,
                                cmap=oC.bwr_2, rasterized=True)
            p1 = self.AX[1, 0].pcolormesh(self.edges_x, self.edges_z, B[1, :, idx_mid_y].T,
                                vmin = np.nanmin(B[1, :, :, idx_mid_z]), vmax = np.nanmax(B[1, :, :, idx_mid_z]),
                                # vmin=-.2+self.mp.B0_y, vmax=.2+self.mp.B0_y,
                                # vmin=-.2, vmax=.2,
                                cmap=oC.bwr_2, rasterized=True)
            p2 = self.AX[2, 0].pcolormesh(self.edges_x, self.edges_z, B[2, :, idx_mid_y].T,
                                vmin = np.nanmin(B[2, :, :, idx_mid_z]), vmax = np.nanmax(B[2, :, :, idx_mid_z]),
                                # vmin=-.2+self.mp.B0_z, vmax=.2+self.mp.B0_z,
                                cmap=oC.bwr_2, rasterized=True)

            p3 = self.AX[0, 1].pcolormesh(self.edges_x, self.edges_z, E[0, :, idx_mid_y].T,
                                vmin = np.amin(E[0, :, :, idx_mid_z]), vmax = np.amax(E[0, :, :, idx_mid_z]),
                                # vmin=-.6, vmax=.6,
                                cmap=oC.bwr_2, rasterized=True)
            p4 = self.AX[1, 1].pcolormesh(self.edges_x, self.edges_z, E[1, :, idx_mid_y].T,
                                vmin = np.amin(E[1, :, :, idx_mid_z]), vmax = np.amax(E[1, :, :, idx_mid_z]),
                                # vmin=-.6, vmax=.6,
                                cmap=oC.bwr_2, rasterized=True)
            p5 = self.AX[2, 1].pcolormesh(self.edges_x, self.edges_z, E[2, :, idx_mid_y].T,#-14.5455,
                                vmin = np.amin(E[2, :, :, idx_mid_z]), vmax = np.amax(E[2, :, :, idx_mid_z]),
                                # vmin=-2., vmax=6.,
                                cmap=oC.bwr_2, rasterized=True)

            p6 = self.AX[0, 2].pcolormesh(self.edges_x, self.edges_z, dens[:, idx_mid_y].T,
                                vmin = np.amin(dens), vmax = np.amax(dens),
                                # vmin=0., vmax=1.2,
                                cmap=oC.bwr_2, rasterized=True)
            p7 = self.AX[1, 2].pcolormesh(self.edges_x, self.edges_z, density_s0[:, idx_mid_y].T,
                                vmin = np.amin(density_s0), vmax = np.amax(density_s0),
                                # vmin=0., vmax=1.2,
                                cmap=oC.bwr_2, rasterized=True)
            try:
                if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
                    p8 = self.AX[2, 2].pcolormesh(self.edges_x, self.edges_z, np.log10(density_s1[:, idx_mid_y]).T,
                                        vmin = np.amin(density_s1), vmax = np.amax(density_s1),
                                        # vmin=-4, vmax=1,
                                        cmap=oC.bwr_2, rasterized=True)
            except:
                pass
            # self.AX[0, 2].pcolormesh(self.edges_x[r], self.edges_y[r], np.sum(self.dens[r], axis=2).T, #self.dens[r, :, :, idx_mid_z].T,
            #                     # vmin = np.amin(self.dens), vmax = np.amax(self.dens),
            #                     cmap=oC.bwr_2, rasterized=True)
            # self.AX[1, 2].pcolormesh(self.edges_x[r], self.edges_z[r], np.sum(self.dens[r], axis=1).T, #self.dens_spec0[r, :, :, idx_mid_z].T,
            #                     # vmin = np.amin(self.dens_spec0), vmax = np.amax(self.dens_spec0),
            #                     cmap=oC.bwr_2, rasterized=True)
            # self.AX[2, 2].pcolormesh(self.edges_y[r], self.edges_z[r], np.sum(self.dens[r], axis=0).T, #self.dens_spec0[r, :, :, idx_mid_z].T,
            #                     # vmin = np.amin(self.dens_spec0), vmax = np.amax(self.dens_spec0),
            #                     cmap=oC.bwr_2, rasterized=True)
            p9 = self.AX[0, 3].pcolormesh(self.edges_x, self.edges_z, (Ji[0, :, idx_mid_y]/dens[:, idx_mid_y]).T+self.mp.v_obs,
                                vmin = np.amin(Ji[:, 0]), vmax = np.amax(Ji[:, 0]),
                                # vmin=-1.-self.mp.v_obs, vmax=1.-self.mp.v_obs,
                                # vmin=-self.mp.v_obs-1.2, vmax=-self.mp.v_obs+1.2,
                                # vmin=-.2, vmax=.2,
                                cmap=oC.bwr_2, rasterized=True)
            p10 = self.AX[1, 3].pcolormesh(self.edges_x, self.edges_z, (Ji[1, :, idx_mid_y]/dens[:, idx_mid_y]).T,
                                vmin = np.amin(Ji[:, 1]), vmax = np.amax(Ji[:, 1]),
                                # vmin=-.2, vmax=.2,
                                cmap=oC.bwr_2, rasterized=True)
            p11 = self.AX[2, 3].pcolormesh(self.edges_x, self.edges_z, (Ji[2, :, idx_mid_y]/dens[:, idx_mid_y]).T,
                                vmin = np.amin(Ji[:, 2]), vmax = np.amax(Ji[:, 2]),
                                # vmin=-.2, vmax=.2,
                                cmap=oC.bwr_2, rasterized=True)

            del(B); del(E); del(dens); del(density_s0); del(Ji)
            try:
                del(density_s1)
            except:
                pass

            try:
                t = np.linspace(0, 2*np.pi, 200)
                r_obs = self.mp.r_obs
                for ax in self.AX.flatten():
                    ax.axis('off')
                    if self.mp.solid_body_cst and ax!=self.AX[2, 2]:
                        ax.plot(r_obs*np.cos(t)+self.max_x*self.mp.centre_x+self.mp.dX,
                                r_obs*np.sin(t)+self.max_y*self.mp.centre_y+self.mp.dX, 'k', lw=2)
            except: pass


            plt.colorbar(p0, ax=self.AX[0, 0])
            plt.colorbar(p1, ax=self.AX[1, 0])
            plt.colorbar(p2, ax=self.AX[2, 0])
            plt.colorbar(p3, ax=self.AX[0, 1])
            plt.colorbar(p4, ax=self.AX[1, 1])
            plt.colorbar(p5, ax=self.AX[2, 1])
            plt.colorbar(p6, ax=self.AX[0, 2])
            plt.colorbar(p7, ax=self.AX[1, 2])
            try:
                if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
                    plt.colorbar(p8, ax=self.AX[2, 2])
            except:
                pass
            plt.colorbar(p9, ax=self.AX[0, 3])
            plt.colorbar(p10, ax=self.AX[1, 3])
            plt.colorbar(p11, ax=self.AX[2, 3])
            self.AX[0, 0].text(10, 10, 'Bx(x, z)')
            self.AX[1, 0].text(10, 10, 'By(x, z)')
            self.AX[2, 0].text(10, 10, 'Bz(x, z)')
            self.AX[0, 1].text(10, 10, 'Ex(x, z)')
            self.AX[1, 1].text(10, 10, 'Ey(x, z)')
            self.AX[2, 1].text(10, 10, 'Ez(x, z)')
            self.AX[0, 2].text(10, 10, 'density(x, z)')
            self.AX[1, 2].text(10, 10, 'dens_sw(x, z)')
            self.AX[0, 3].text(10, 10, 'v or Jix(x, z)')
            self.AX[1, 3].text(10, 10, 'v or Jiy(x, z)')
            self.AX[2, 3].text(10, 10, 'v or Jiz(x, z)')



            plt.tight_layout()

            if save_fig:
                file_name = f'{self.local_path}/products/plots/all_{self.fig.plane}_{self.idx_it:04}.png'
                # file_name = f'/home/etienneb/Desktop/nikbientamer_SWRF/yo_all_{self.idx_it:04}.png'
                print(f'Saving {file_name}')
                plt.savefig(file_name)
            else:
                plt.show()





    def select_field(self, event, vminmax=None):

        plt_it = True
        if   event.inaxes==self.AX[0, 0]:
            label = 'Bx'
        elif event.inaxes==self.AX[1, 0]:
            label = 'By'
        elif event.inaxes==self.AX[2, 0]:
            label = 'Bz'
        elif event.inaxes==self.AX[0, 1]:
            label = 'Ex'
        elif event.inaxes==self.AX[1, 1]:
            label = 'Ey'
        elif event.inaxes==self.AX[2, 1]:
            label = 'Ez'
        elif event.inaxes==self.AX[0, 2]:
            label = 'density'
        elif event.inaxes==self.AX[1, 2]:
            label = 'density_s0'
        elif event.inaxes==self.AX[2, 2]:
            label = 'density_s1'
        elif event.inaxes==self.AX[0, 3]:
            label = 'Jix'
        elif event.inaxes==self.AX[1, 3]:
            label = 'Jiy'
        elif event.inaxes==self.AX[2, 3]:
            label = 'Jiz'
        else:
            plt_it = False

        if plt_it:
            self.plt_field(label, vminmax=None, filled_contour=False)



    def plt_field(self, label, vminmax=None, save_fig=False, filled_contour=False,
                  log=False, show_patches=False, plane=None, xylim=None, interpolate=False,
                  mask_body=False, B_lines=False, show_colorbar=True, show_spines=True,
                  save_as_pdf=False, idx_cut=None, density_lines=None):

        set_vminmax = True

        if vminmax is not None:
            set_vminmax = False

        cmap = oC.bwr_2

        keep_guard_cells = False

        if label == 'Bx':
            field = self.load_field('B')[0]
        elif label == 'By':
            field = self.load_field('B')[1]
        elif label == 'Bz':
            field = self.load_field('B')[2]
        elif label == 'B':
            field = self.load_field('B')
            field = mT.norm(field)
        elif label == 'B+B_dip':
            field = self.load_field('B')
            field_dip = self.load_field('B_dip')
            field = mT.norm(field_dip)
        elif label == 'B_dipx':
            field = self.load_field('B_dip')[0]
        elif label == 'B_dipy':
            field = self.load_field('B_dip')[1]
        elif label == 'B_dipz':
            field = self.load_field('B_dip')[2]
        elif label == 'B_dip':
            field = mT.norm(self.load_field('B_dip'))
        elif label == 'B_perp_sqr':
            field = self.load_field('B')
            # field = self.stag_vector(field)
            field = (field[0]**2 + field[1]**2)
            cmap = oC.bwr_2
            # cmap = oC.wb
        elif label == 'B_perp':
            label='B_perp'
            B = self.load_field('B')
            # B = self.stag_vector(B)
            field = np.sqrt(B[0]**2 + B[1]**2)
            cmap = oC.wb
            # cmap = 'twilight'
        elif label=='div_B':
            B = self.load_field('B')
            dx = self.mp.dX
            if self.mp.NB_DIM==2:
                dxBx = 1./(12.*dx) * ( B[0, 4:  , 2:-2]-8*B[0, 3:-1, 2:-2]+8*B[0, 1:-3, 2:-2]-B[0,  :-4, 2:-2] )
                dyBy = 1./(12.*dx) * ( B[1, 2:-2, 4:  ]-8*B[1, 2:-2, 3:-1]+8*B[1, 2:-2, 1:-3]-B[1, 2:-2,  :-4] )
                field = dxBx + dyBy
            elif self.mp.NB_DIM==3:
                dxBx = 1./(12.*dx) * ( B[0, 4:  , 2:-2, 2:-2]-8*B[0, 3:-1, 2:-2, 2:-2]+8*B[0, 1:-3, 2:-2, 2:-2]-B[0,  :-4, 2:-2, 2:-2] )
                dyBy = 1./(12.*dx) * ( B[1, 2:-2, 4:  , 2:-2]-8*B[1, 2:-2, 3:-1, 2:-2]+8*B[1, 2:-2, 1:-3, 2:-2]-B[1, 2:-2,  :-4, 2:-2] )
                dzBz = 1./(12.*dx) * ( B[2, 2:-2, 2:-2, 4:  ]-8*B[2, 2:-2, 2:-2, 3:-1]+8*B[2, 2:-2, 2:-2, 1:-3]-B[2, 2:-2, 2:-2,  :-4] )
                field = dxBx + dyBy + dzBz

        elif label == 'Ex':
            field = self.load_field('E')[0]
        elif label == 'Ey':
            field = self.load_field('E')[1]
        elif label == 'Ez':
            field = self.load_field('E')[2]
        elif label == 'E':
            field = self.load_field('E')
            field = mT.norm(field)

        elif label == 'curl_E':
            field = self.load_field('curl_E')
            field = mT.norm(field)
        elif label == 'curl_E_x':
            field = self.load_field('curl_E')
            field = field[0]
        elif label == 'curl_E_y':
            field = self.load_field('curl_E')
            field = field[1]
        elif label == 'curl_E_z':
            field = self.load_field('curl_E')
            field = field[2]

        elif label=='EdotJ':
            E = self.load_field('E')
            J = self.load_field('Jtot')
            field = np.sum(E*J, axis=0)

        elif label == 'density':
            field = self.load_field('density')
        elif label == 'region_ID':
            field = self.load_field('region_ID')
            cmap = 'Pastel2'
        elif label == 'vacuum_boarder':
            field = self.load_field('vacuum_boarder')
            cmap = 'Pastel2'
        elif label == 'density_s0':
            field = self.load_field('density_s0')
        elif label == 'density_s1':
            field = self.load_field('density_s1')
            cmap = oC.wbr_1

        elif label == 'Jix':
            field = self.load_field('Ji')[0]
        elif label == 'Jiy':
            field = self.load_field('Ji')[1]
        elif label == 'Jiz':
            field = self.load_field('Ji')[2]

        elif label == 'Ji_s0_x':
            field = self.load_field('Ji_s0')[0]
        elif label == 'Ji_s0_y':
            field = self.load_field('Ji_s0')[1]
        elif label == 'Ji_s0_z':
            field = self.load_field('Ji_s0')[2]

        elif label == 'Ji_s1_x':
            field = self.load_field('Ji_s1')[0]
        elif label == 'Ji_s1_y':
            field = self.load_field('Ji_s1')[1]
        elif label == 'Ji_s1_z':
            field = self.load_field('Ji_s1')[2]

        elif label == 'uix':
            d = self.load_field('density')
            field = self.load_field('Ji')[0]/d
        elif label == 'uiy':
            d = self.load_field('density')
            field = self.load_field('Ji')[1]/d
        elif label == 'uiz':
            d = self.load_field('density')
            field = self.load_field('Ji')[2]/d
        elif label == 'ui_perp_sqr':
            Ji = self.load_field('Ji')
            dens = self.load_field('density')
            field = (Ji[0]/dens)**2 + (Ji[1]/dens)**2
            # cmap = oC.wr

        elif label == 'J':
            field = self.load_field('Jtot')
            field = mT.norm(field)
        elif label == 'Jx':
            field = self.load_field('Jtot')[0]
        elif label == 'Jy':
            field = self.load_field('Jtot')[1]
        elif label == 'Jz':
            field = self.load_field('Jtot')[2]

        elif label == 'd_i':
            dens = self.load_field('density')
            omega_i  = np.sqrt(e*e*self.mp.n0_SI*dens/(eps0*m_i))
            d_i      = c/omega_i
            field = d_i/self.mp.d_i


        elif label == 'pres_ion_xx':
            pres_ion = self.load_field('pres_ion')
            #dens = self.load_field('density')
            field = pres_ion[0]
        elif label == 'pres_ion_yy':
            pres_ion = self.load_field('pres_ion')
            field = pres_ion[1]
        elif label == 'pres_ion_zz':
            pres_ion = self.load_field('pres_ion')
            field = pres_ion[2]
        elif label == 'pres_ion_xy':
            pres_ion = self.load_field('pres_ion')
            field = pres_ion[3]
        elif label == 'pres_ion_xz':
            pres_ion = self.load_field('pres_ion')
            field = pres_ion[4]
        elif label == 'pres_ion_yz':
            pres_ion = self.load_field('pres_ion')
            field = pres_ion[5]

        elif label == 'pres_spec_0':
            pres_ion = self.load_field('pres_spec_0')
            field = 1./3.*(pres_ion[0] + pres_ion[1] + pres_ion[2])
        elif label == 'pres_spec_0_xx':
            pres_ion = self.load_field('pres_spec_0')
            field = pres_ion[0]
        elif label == 'pres_spec_0_yy':
            pres_ion = self.load_field('pres_spec_0')
            field = pres_ion[1]
        elif label == 'pres_spec_0_zz':
            pres_ion = self.load_field('pres_spec_0')
            field = pres_ion[2]
        elif label == 'pres_spec_0_xy':
            pres_ion = self.load_field('pres_spec_0')
            field = pres_ion[3]
        elif label == 'pres_spec_0_xz':
            pres_ion = self.load_field('pres_spec_0')
            field = pres_ion[4]
        elif label == 'pres_spec_0_yz':
            pres_ion = self.load_field('pres_spec_0')
            field = pres_ion[5]

        elif label == 'pres_spec_1':
            pres_ion = self.load_field('pres_spec_1')
            field = 1./3.*(pres_ion[0] + pres_ion[1] + pres_ion[2])
        elif label == 'pres_spec_1_xx':
            pres_ion = self.load_field('pres_spec_1')
            field = pres_ion[0]
        elif label == 'pres_spec_1_yy':
            pres_ion = self.load_field('pres_spec_1')
            field = pres_ion[1]
        elif label == 'pres_spec_1_zz':
            pres_ion = self.load_field('pres_spec_1')
            field = pres_ion[2]
        elif label == 'pres_spec_1_xy':
            pres_ion = self.load_field('pres_spec_1')
            field = pres_ion[3]
        elif label == 'pres_spec_1_xz':
            pres_ion = self.load_field('pres_spec_1')
            field = pres_ion[4]
        elif label == 'pres_spec_1_yz':
            pres_ion = self.load_field('pres_spec_1')
            field = pres_ion[5]

        elif label == 'temp_spec_1':
            pres_ion = self.load_field('pres_spec_1')
            field = 1./3.*(pres_ion[0] + pres_ion[1] + pres_ion[2])
            field /= self.load_field('density_s1')

        elif label == 'T_ion_xx':
            pres_ion = self.load_field('pres_ion')
            dens = self.load_field('density')
            field = pres_ion[0]/dens

        else:
            sys.exit(f'Label {label} not implemented in plt_field().')


        if log:
            field = np.log10(np.abs(field))


        if mask_body:
            region_ID = self.load_field('region_ID')
            if self.mp.NB_DIM==3:
                if plane == 'xy':
                    region_ID = region_ID[:, :, int(self.mp.centre_z*self.grid_z_box.size)]
                elif plane == 'xz':
                    region_ID = region_ID[:, int(self.mp.centre_y*self.grid_y_box.size)]
                elif plane == 'yz':
                    region_ID = region_ID[int(self.mp.centre_x*self.grid_x_box.size)]
            field[region_ID==2] = np.nan

        if set_vminmax:
            vmin =  np.nanmin(field)
            vmax = np.nanmax(field)
        else:
            vmin = vminmax[0]
            vmax = vminmax[1]

        if self.mp.NB_DIM==3:
            if plane==None:
                try:
                    plane = self.fig.plane
                except:
                    plane = 'xy'
            if plane == 'xy':
                if idx_cut is None:
                    field = field[:, :, int(self.mp.centre_z*self.grid_z_box.size)]
                    # field = np.sum(field, axis=2)/200
                else:
                    field = field[:, :, idx_cut]
                x = self.edges_x
                y = self.edges_y
            elif plane == 'xz':
                if idx_cut is None:
                    field = field[:, int(self.mp.centre_y*self.grid_y_box.size)]
                    # field = np.sum(field, axis=1)/200
                else:
                    field = field[:, idx_cut]
                x = self.edges_x
                y = self.edges_z
            elif plane == 'yz':
                if idx_cut is None:
                    field = field[int(self.mp.centre_x*self.grid_x_box.size)]
                    # field = np.sum(field, axis=0)/200
                else:
                    field = field[idx_cut]
                x = self.edges_y
                y = self.edges_z
        else:
            plane = 'xy'
            x = self.edges_x
            y = self.edges_y






        self.fig2, ax = plt.subplots(figsize=(14, 14))

        ax.set_aspect('equal')


        if B_lines:

            if self.mp.dipole_cst:
                B = self.load_field('B') + self.load_field('B_dip')#
            else:
                B =  self.load_field('B')

            if self.mp.NB_DIM==2:
                fieldx = B[0]
                fieldy = B[1]
                x = self.grid_x_box
                y = self.grid_y_box
            elif self.mp.NB_DIM==3:
                if plane == 'xy':
                    fieldx = B[0, :, :, int(self.mp.centre_z*self.grid_z_box.size)]
                    fieldy = B[1, :, :, int(self.mp.centre_z*self.grid_z_box.size)]
                    x = self.grid_x_box
                    y = self.grid_y_box
                    if xylim is not None:
                        fieldx[x<xylim[0][0]] = np.nan
                        fieldx[x>xylim[0][1]] = np.nan
                        fieldx[:, y<xylim[1][0]] = np.nan
                        fieldx[:, y>xylim[1][1]] = np.nan
                elif plane == 'xz':
                    fieldx = B[0, :, int(self.mp.centre_y*self.grid_y_box.size)]
                    fieldy = B[2, :, int(self.mp.centre_y*self.grid_y_box.size)]
                    x = self.grid_x_box
                    y = self.grid_z_box
                elif plane == 'yz':
                    fieldx = B[1, int(self.mp.centre_x*self.grid_x_box.size)]
                    fieldy = B[2, int(self.mp.centre_x*self.grid_x_box.size)]
                    x = self.grid_y_box
                    y = self.grid_z_box


            if density_lines is None:
                density_lines = 5
            ax.streamplot(x, y, fieldx.T, fieldy.T,
                                  density=density_lines,
                                  # start_points=start_points,
                                  color='k', linewidth=.3)#, linewidth=np.log10(B_perp).T)

        # if label=='density_s1':
        #     ui_spec_1 = -self.load_field('Ji_s1')/field
        #     if self.mp.NB_DIM==2:
        #         fieldx = Ji_s1[0, :, :]
        #         fieldy = Ji_s1[1, :, :]
        #         x = self.grid_x_box
        #         y = self.grid_y_box
        #     elif self.mp.NB_DIM==3:
        #         if plane == 'xy':
        #             fieldx = Ji_s1[0, :, :, int(self.mp.centre_z*self.grid_z_box.size)]
        #             fieldy = Ji_s1[1, :, :, int(self.mp.centre_z*self.grid_z_box.size)]
        #             x = self.grid_x_box
        #             y = self.grid_y_box
        #         elif plane == 'xz':
        #             fieldx = Ji_s1[0, :, int(self.mp.centre_y*self.grid_y_box.size)]
        #             fieldy = Ji_s1[2, :, int(self.mp.centre_y*self.grid_y_box.size)]
        #             x = self.grid_x_box
        #             y = self.grid_z_box
        #         elif plane == 'yz':
        #             centre_x = int(self.mp.centre_x*self.grid_x_box.size)
        #             # centre_x = int(.5*(1+self.mp.centre_x)*self.grid_x_box.size)
        #             # print(centre_x)
        #             fieldx = ui_spec_1[1, centre_x]
        #             fieldy = ui_spec_1[2, centre_x]
        #             x = self.grid_y_box
        #             y = self.grid_z_box
        #
        #     ax.streamplot(x, y, fieldx.T, fieldy.T,
        #                           density=5,
        #                           # start_points=start_points,
        #                           color='k', linewidth=.5)#np.sqrt((fieldx**2+fieldy**2)))#, linewidth=np.log10(B_perp).T)

        if interpolate:
            interp = 'bilinear'
        else:
            interp = 'none'
        pp0 = ax.imshow(field.T,
                          vmin=vmin, vmax=vmax,
                          extent=(x[0], x[-1], y[0], y[-1]),
                          interpolation=interp,
                          rasterized=True, cmap=cmap, origin='lower')



        if show_patches and self.idx_it>100:
            for r in range(self.nb_proc):
                for p in range(self.mp.smooth_patch_save_len_cst):
                    i = int(self.smooth_patches[r, 0, p])
                    j = int(self.smooth_patches[r, 1, p])
                    t = int(self.smooth_patches[r, 2, p])
                    if (i!=0 and j!=0 and t>self.idx_it-100 and t<self.idx_it):
                        # print(i, j, t)
                        try:
                            ax.plot(self.grid_x[r, i], self.grid_y[r, j], 'x', ms=12)
                        except:
                            pass


        if label=='density_s1' and 0:
            R = 180#146#116#self.mp.R_gyr_norm
            print('gyro-radius: ', R)
            print('v0_com: ', self.mp.v_obs, self.mp.v_obs/self.mp.v_A)
            t = np.linspace(0, np.pi, 400)
            ## For run 31
            xx = -R*(t - np.sin(t)) + (5./8.*500) + 8
            yy = -R*(1 + np.cos(t)) + 540#460#480
            ## For run 33
            xx = -R*(t - np.sin(t)) + (5./8.*500) + 12
            yy = -R*(1 + np.cos(t)) + 610#480
            ax.plot(xx, yy, 'k')
            ax.plot(xx[0], yy[0], 'xk')

            try:
                path = '/home/etienneb/Models/Menura/menura_test_particle'
                traj = np.load(f'{path}/products/HK/traj.npy').T
                t = np.linspace(0, .05*1000, 1000)
                v0_com = self.mp.v_A*self.mp.dX*self.mp.nb_cell_per_shift_cst/(self.mp.nb_it_per_shift_cst*self.mp.dt)
                v0_com /= self.mp.v_A
                traj[0, :, :] -= v0_com*t[:, None]
                for t in range(100):
                    ax.plot((traj[0, :, t]-traj[0, 0, t]+5./8.*500+5)%500, traj[1, :, t], c=oC.rgb2[0], lw=.5, alpha=.6)
            except:
                pass

        try:
            if self.mp.solid_body_cst:
                t = np.linspace(0, 2*np.pi, 200)
                r_obs = self.mp.r_obs
                if plane=='xy':
                    centre_x = self.mp.centre_x*self.mp.len_x_cst*self.mp.dX - .5*self.mp.dX
                    centre_y = self.mp.centre_y*self.mp.len_y_cst*self.mp.mpi_nb_proc_y*self.mp.dX - .5*self.mp.dX
                if plane=='xz':
                    centre_x = self.mp.centre_x*self.mp.len_x_cst*self.mp.dX
                    centre_y = self.mp.centre_z*self.mp.len_z_cst*self.mp.mpi_nb_proc_z*self.mp.dX
                if plane=='yz':
                    centre_x = self.mp.centre_y*self.mp.len_y_cst*self.mp.mpi_nb_proc_y*self.mp.dX
                    centre_y = self.mp.centre_z*self.mp.len_z_cst*self.mp.mpi_nb_proc_z*self.mp.dX
                ax.plot(r_obs*np.cos(t)+centre_x,
                             r_obs*np.sin(t)+centre_y, 'k', lw=2)
                ax.plot(self.mp.r_smooth_shell*np.cos(t)+centre_x,
                             self.mp.r_smooth_shell*np.sin(t)+centre_y, '--k', lw=2)
        except: pass

        idx_y = int((.999-1.)*self.mp.mpi_nb_proc_y*self.mp.len_y_cst+self.mp.len_y_cst)

        # ax.text(self.edges_x[-1, int(.15*self.mp.len_x_cst)], .75*self.edges_y[-1, idx_y], label, ha='center', va='center', fontsize=20)
        # ax.text(self.edges_x[-1, int(.1*self.mp.len_x_cst)], self.edges_y[-1, idx_y], np.amax(field), ha='center', va='center', fontsize=20)
        if log:
            plt.title(f'log10 {label}')
        else:
            plt.title(label)

        if xylim is not None:
            ax.set_xlim([xylim[0][0], xylim[0][1]])
            ax.set_ylim([xylim[1][0], xylim[1][1]])

        if not show_spines:
            ax.axis('off')

        if show_colorbar:
            posAx = ax.get_position()
            cax = self.fig2.add_axes([posAx.x1*1., posAx.y0, 0.02, .3])
            cb = self.fig2.colorbar(pp0, cax=cax, orientation='vertical')
            # cb.set_label(label, rotation=0, ha='left', fontsize=24)
        #
        mT.set_spines(ax)
        plt.tight_layout()
        #
        if save_fig:
            if save_as_pdf:
                fn = f'{self.local_path}/products/plots/{label}_{plane}_{self.idx_it:04}.pdf'
                print(f'Saving {fn}...')
                plt.savefig(fn)

            fn = f'{self.local_path}/products/plots/{label}_{plane}_{self.idx_it:04}.png'
            print(f'Saving {fn}...')
            plt.savefig(fn)
            # plt.savefig(f'/home/etienneb/Desktop/plots_tmp/{label}_{self.idx_it:04}.png')
            # plt.savefig(f'../run_024/products/plots/{label}_{self.idx_it:04}.png')
        else:
            plt.show()




    def plt_ohm(self, save_fig=False):

        print('\nPlot Ohm.\n')
        if self.mp.NB_DIM==2:
        #     self.E_mot    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4))#, self.mp.len_z_cst+4))
        #     self.E_hal    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4))#, self.mp.len_z_cst+4))
        #     self.E_amb    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4))#, self.mp.len_z_cst+4))
        #
        #     for r in range(self.nb_proc_tot):
        #         try:
        #             self.E_mot[r] = np.load(f'{self.local_path}/products/E_mot_it{self.idx_it}_rank{r}.npy')
        #             self.E_hal[r] = np.load(f'{self.local_path}/products/E_hal_it{self.idx_it}_rank{r}.npy')
        #             self.E_amb[r] = np.load(f'{self.local_path}/products/E_amb_it{self.idx_it}_rank{r}.npy')
        #             plt_proc = True
        #             print('Ohm from run.')
        #         except:
        #             self.produce_ohm()
        #             plt_proc = False
        #             print('Ohm re-calculated.')

            plt_proc = False
            fig, AX = plt.subplots(3, 3, figsize=(18, 14), sharex=True, sharey=True)
            for ax in AX.flatten():
                ax.set_aspect('equal')

            if plt_proc:
                for r in range(self.nb_proc):

                    p0 = AX[0, 0].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_mot[r, 0].T,
                                        vmin = np.amin(self.E_mot[:, 0]), vmax = np.amax(self.E_mot[:, 0]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
                    p1 = AX[1, 0].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_mot[r, 1].T,
                                        vmin = np.amin(self.E_mot[:, 1]), vmax = np.amax(self.E_mot[:, 1]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
                    p2 = AX[2, 0].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_mot[r, 2].T,
                                        vmin = np.amin(self.E_mot[:, 2]), vmax = np.amax(self.E_mot[:, 2]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)


                    p3 = AX[0, 1].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_hal[r, 0].T,
                                        vmin = np.amin(self.E_hal[:, 0]), vmax = np.amax(self.E_hal[:, 0]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
                    p4 = AX[1, 1].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_hal[r, 1].T,
                                        vmin = np.amin(self.E_hal[:, 1]), vmax = np.amax(self.E_hal[:, 1]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
                    p5 = AX[2, 1].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_hal[r, 2].T,
                                        vmin = np.amin(self.E_hal[:, 2]), vmax = np.amax(self.E_hal[:, 2]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)

                    p6 = AX[0, 2].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_amb[r, 0].T,
                                        vmin = np.amin(self.E_amb[:, 0]), vmax = np.amax(self.E_amb[:, 0]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
                    p7 = AX[1, 2].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_amb[r, 1].T,
                                        vmin = np.amin(self.E_amb[:, 1]), vmax = np.amax(self.E_amb[:, 1]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
                    p8 = AX[2, 2].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_amb[r, 2].T,
                                        vmin = np.amin(self.E_amb[:, 2]), vmax = np.amax(self.E_amb[:, 2]),
                                        # vmin = -.6, vmax = .6,
                                        cmap=oC.bwr_2, rasterized=True)
            else:
                p0 = AX[0, 0].pcolormesh(self.edges_x, self.edges_y, self.E_mot[0].T,
                                    # vmin = np.amin(self.E_mot[r, 0, :, :, 50]), vmax = np.amax(self.E_mot[r, 0, :, :, 50]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)
                p1 = AX[1, 0].pcolormesh(self.edges_x, self.edges_y, self.E_mot[1].T,
                                    # vmin = np.amin(self.E_mot[:, 1]), vmax = np.amax(self.E_mot[:, 1]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)
                p2 = AX[2, 0].pcolormesh(self.edges_x, self.edges_y, self.E_mot[2].T,
                                    # vmin = np.amin(self.E_mot[:, 2]), vmax = np.amax(self.E_mot[:, 2]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)

                p3 = AX[0, 1].pcolormesh(self.edges_x, self.edges_y, self.E_hal[0].T,
                                    # vmin = np.amin(self.E_hal[:, 0]), vmax = np.amax(self.E_hal[:, 0]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)
                p4 = AX[1, 1].pcolormesh(self.edges_x, self.edges_y, self.E_hal[1].T,
                                    # vmin = np.amin(self.E_hal[:, 1]), vmax = np.amax(self.E_hal[:, 1]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)
                p5 = AX[2, 1].pcolormesh(self.edges_x, self.edges_y, self.E_hal[2].T,
                                    # vmin = np.amin(self.E_hal[:, 2]), vmax = np.amax(self.E_hal[:, 2]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)

                p6 = AX[0, 2].pcolormesh(self.edges_x, self.edges_y, self.E_amb[0].T,
                                    # vmin = np.amin(self.E_amb[:, 0]), vmax = np.amax(self.E_amb[:, 0]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)
                p7 = AX[1, 2].pcolormesh(self.edges_x, self.edges_y, self.E_amb[1].T,
                                    # vmin = np.amin(self.E_amb[:, 1]), vmax = np.amax(self.E_amb[:, 1]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)
                p8 = AX[2, 2].pcolormesh(self.edges_x, self.edges_y, self.E_amb[2].T,
                                    # vmin = np.amin(self.E_amb[:, 2]), vmax = np.amax(self.E_amb[:, 2]),
                                    # vmin = -.6, vmax = .6,
                                    cmap=oC.bwr_2, rasterized=True)

            # for ax in AX.flatten():
            #     ax.set_xlim([220, 260])
            #     ax.set_ylim([240, 280])
            plt.colorbar(p0, ax=AX[0, 0])
            plt.colorbar(p1, ax=AX[1, 0])
            plt.colorbar(p2, ax=AX[2, 0])
            plt.colorbar(p3, ax=AX[0, 1])
            plt.colorbar(p4, ax=AX[1, 1])
            plt.colorbar(p5, ax=AX[2, 1])
            plt.colorbar(p6, ax=AX[0, 2])
            plt.colorbar(p7, ax=AX[1, 2])
            plt.colorbar(p8, ax=AX[2, 2])

            mT.set_spines(AX)
            plt.tight_layout()

            if save_fig:
                file_name = f'{self.local_path}/products/plots/ohm_{self.idx_it:04}.pdf'
                print(f'Saving {file_name}')
                plt.savefig(file_name)
            else:
                plt.show()


        elif self.mp.NB_DIM==3:

            self.E_mot    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
            self.E_hal    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))
            self.E_amb    = np.zeros((self.nb_proc, 3, self.mp.len_x_cst+4, self.mp.len_y_cst+4, self.mp.len_z_cst+4))

            for r in range(self.nb_proc):
                try:
                    self.E_mot[r] = np.load(f'{self.local_path}/products/E_mot_it{self.idx_it}_rank{r}.npy')
                    self.E_hal[r] = np.load(f'{self.local_path}/products/E_hal_it{self.idx_it}_rank{r}.npy')
                    self.E_amb[r] = np.load(f'{self.local_path}/products/E_amb_it{self.idx_it}_rank{r}.npy')
                except:
                    pass

            self.produce_ohm()


            fig, AX = plt.subplots(3, 3, figsize=(18, 14))
            for ax in AX.flatten():
                ax.set_aspect('equal')

            idx_mid_z = int((self.mp.len_z_cst+4)/2.)

            # for r in range(self.nb_proc):
            #     # print(self.edges_x[r].shape, self.edges_y[r].shape, self.edges_z[k].shape,)
            #     p0 = AX[0, 0].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_mot[r, 0, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_mot[r, 0, :, :, 50]), vmax = np.amax(self.E_mot[r, 0, :, :, 50]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #     p1 = AX[1, 0].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_mot[r, 1, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_mot[:, 1]), vmax = np.amax(self.E_mot[:, 1]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #     p2 = AX[2, 0].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_mot[r, 2, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_mot[:, 2]), vmax = np.amax(self.E_mot[:, 2]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #
            #
            #     p3 = AX[0, 1].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_hal[r, 0, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_hal[:, 0]), vmax = np.amax(self.E_hal[:, 0]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #     p4 = AX[1, 1].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_hal[r, 1, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_hal[:, 1]), vmax = np.amax(self.E_hal[:, 1]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #     p5 = AX[2, 1].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_hal[r, 2, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_hal[:, 2]), vmax = np.amax(self.E_hal[:, 2]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #
            #     p6 = AX[0, 2].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_amb[r, 0, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_amb[:, 0]), vmax = np.amax(self.E_amb[:, 0]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #     p7 = AX[1, 2].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_amb[r, 1, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_amb[:, 1]), vmax = np.amax(self.E_amb[:, 1]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            #     p8 = AX[2, 2].pcolormesh(self.edges_x[r], self.edges_y[r], self.E_amb[r, 2, :, :, idx_mid_z].T,
            #                         # vmin = np.amin(self.E_amb[:, 2]), vmax = np.amax(self.E_amb[:, 2]),
            #                         # vmin = -.6, vmax = .6,
            #                         cmap=oC.bwr_2, rasterized=True)
            p0 = AX[0, 0].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_mot[0, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_mot[r, 0, :, :, 50]), vmax = np.amax(self.E_mot[r, 0, :, :, 50]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)
            p1 = AX[1, 0].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_mot[1, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_mot[:, 1]), vmax = np.amax(self.E_mot[:, 1]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)
            p2 = AX[2, 0].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_mot[2, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_mot[:, 2]), vmax = np.amax(self.E_mot[:, 2]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)

            p3 = AX[0, 1].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_hal[0, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_hal[:, 0]), vmax = np.amax(self.E_hal[:, 0]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)
            p4 = AX[1, 1].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_hal[1, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_hal[:, 1]), vmax = np.amax(self.E_hal[:, 1]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)
            p5 = AX[2, 1].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_hal[2, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_hal[:, 2]), vmax = np.amax(self.E_hal[:, 2]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)

            p6 = AX[0, 2].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_amb[0, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_amb[:, 0]), vmax = np.amax(self.E_amb[:, 0]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)
            p7 = AX[1, 2].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_amb[1, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_amb[:, 1]), vmax = np.amax(self.E_amb[:, 1]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)
            p8 = AX[2, 2].pcolormesh(self.edges_x_box, self.edges_y_box, self.E_amb[2, :, :, idx_mid_z].T,
                                # vmin = np.amin(self.E_amb[:, 2]), vmax = np.amax(self.E_amb[:, 2]),
                                # vmin = -.6, vmax = .6,
                                cmap=oC.bwr_2, rasterized=True)

            # for ax in AX.flatten():
            #     ax.set_xlim([220, 260])
            #     ax.set_ylim([240, 280])

            plt.colorbar(p0, ax=AX[0, 0])
            plt.colorbar(p1, ax=AX[1, 0])
            plt.colorbar(p2, ax=AX[2, 0])
            plt.colorbar(p3, ax=AX[0, 1])
            plt.colorbar(p4, ax=AX[1, 1])
            plt.colorbar(p5, ax=AX[2, 1])
            plt.colorbar(p6, ax=AX[0, 2])
            plt.colorbar(p7, ax=AX[1, 2])
            plt.colorbar(p8, ax=AX[2, 2])

            try:
                t = np.linspace(0, 2*np.pi, 200)
                r_obs = self.mp.r_obs/self.mp.d_i
                for ax in AX.flatten():
                    ax.axis('off')
                    if self.mp.solid_body_cst and ax!=AX[2, 2]:
                        ax.plot(r_obs*np.cos(t)+self.max_x*self.mp.centre_x+self.mp.dX,
                                r_obs*np.sin(t)+self.max_y*self.mp.centre_y+self.mp.dX, 'k', lw=2)
            except: pass

            mT.set_spines(AX)
            plt.tight_layout()

            if save_fig:
                file_name = f'{self.local_path}/products/plots/ohm_{self.idx_it:04}.pdf'
                print(f'Saving {file_name}')
                plt.savefig(file_name)
            else:
                plt.show()



    def plt_poynting(self, save_fig=False):

        fig, AX = plt.subplots(1, 3, figsize=(18, 14))
        for ax in AX.flatten():
            ax.set_aspect('equal')

        for r in range(self.nb_proc):
            AX[0].pcolormesh(self.edges_x[r], self.edges_y[r], self.poynting[r, 0].T,
                                # vmin = np.nanmin(self.poynting[:, 0]), vmax = np.nanmax(self.poynting[:, 0]),
                                vmin=-2., vmax=2.,
                                cmap=oC.bwr_2, rasterized=True)
            AX[1].pcolormesh(self.edges_x[r], self.edges_y[r], self.poynting[r, 1].T,
                                vmin = np.nanmin(self.poynting[:, 1]), vmax = np.nanmax(self.poynting[:, 1]),
                                cmap=oC.bwr_2, rasterized=True)
            AX[2].pcolormesh(self.edges_x[r], self.edges_y[r], self.poynting[r, 2].T,
                                vmin = np.nanmin(self.poynting[:, 2]), vmax = np.nanmax(self.poynting[:, 2]),
                                cmap=oC.bwr_2, rasterized=True)


        # for ax in AX.flatten():
        #     ax.set_xlim([220, 260])
        #     ax.set_ylim([240, 280])

        mT.set_spines(AX)
        plt.tight_layout()

        if save_fig:
            file_name = f'{self.local_path}/products/plots/ohm_{self.idx_it:04}.pdf'
            print(f'Saving {file_name}')
            plt.savefig(file_name)
        else:
            plt.show()




    def plt_tomo(self, label='B', save_fig=False, axis='x', save_as_pdf=False):

        from mpl_toolkits.mplot3d import Axes3D

        plotContour = False

        grid_x = self.grid_x_box.copy()
        grid_y = self.grid_y_box.copy()
        grid_z = self.grid_z_box.copy()
        dX = grid_x[1] - grid_x[0]

        if label=='B':
            B = self.load_field('B')
            field = np.log10( mT.norm(B) )#
            vmin = -1; vmax = 1
            cmap = oC.bwr_2
            #cmap = oC.wb

        elif label=='density':
            field = np.log10(self.load_field('density'))
            vmin = -1. ; vmax = 1.#np.amax(field)
            cmap = oC.bwr_2

        elif label=='density_s0':
            field = np.log10(self.load_field('density_s0'))
            vmin = -1. ; vmax = 1.#np.amax(field)
            cmap = oC.bwr_2

        elif label=='density_s1':
            field = np.log10(self.load_field('density_s1'))
            vmin = -3 ; vmax = .5#np.amax(field)
            cmap = oC.wbr_1

        if axis=='y':
            field = np.swapaxes(field, 0, 1)
        elif axis=='z':
            field = np.swapaxes(field, 0, 2)



        len_x = grid_x.size-4
        len_y = grid_y.size-4
        len_z = grid_z.size-4

        ## Centre the grids around 0:
        grid_x -= int(len_x/2.)*dX
        grid_y -= int(len_y/2.)*dX
        grid_z -= int(len_z/2.)*dX

        indXX = []

        # # xOI = np.array([-95, -45, 0, 45, 95])
        # xOI = np.array([-100, -75, -50, -25, 0])
        #
        # for xxoi in xOI:
        #     indXX.append((np.abs(grid_x-xxoi)).argmin())
        # indXX = np.array(indXX)
        how_many_slices = 8
        indXX = np.linspace(.1, .53, how_many_slices)*len_x
        indXX = indXX.astype(int)

        winDimY = grid_y[-2]
        winDimYMin = grid_y[30]
        winDimYMax = grid_y[-30]
        winDimZ = grid_z[-2]
        winDimZMin = grid_z[2]
        winDimZMax = grid_z[-2]
        winDimZTot = grid_z[-2]

        # indY =   (grid_y > -winDimY) * (grid_y < winDimY)
        indY =   (grid_y > winDimYMin) * (grid_y < winDimYMax)
        indZ =   (grid_z > winDimZMin) * (grid_z < winDimZMax)
        y, z = np.meshgrid(grid_y[indY], grid_z[indZ])


        # Y = [-winDimY, -winDimY, winDimY, winDimY, -winDimY]
        Y = [winDimYMin, winDimYMin, winDimYMax, winDimYMax, winDimYMin]
        # Z = [-winDimZ, winDimZ, winDimZ, -winDimZ, -winDimZ]
        Z = [winDimZMin, winDimZMax, winDimZMax, winDimZMin, winDimZMin]
        # Z2 = [winDimZMin, 8e6, 8e6, winDimZMin, winDimZMin]
        # YFrame = [-winDimZTot, -winDimZTot, winDimZTot, winDimZTot, -winDimZTot]
        # ZFrame = [winDimZMin, winDimZMax, winDimZMax, winDimZMin, winDimZMin]
        # XFrame = [.8*grid_x[0], 0.*grid_x[-1]]

        stride = 1

        fig = plt.figure(figsize=(24, 16))
        ax = plt.axes(projection=Axes3D.name)
        ax.view_init(elev=20., azim=285.)
        # ax.set_facecolor('#d1d1d2')
        ax.set_facecolor('w')
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.5, 1.1, 1., 1])) ## Stretching the x-axis!

        for indX in indXX:
            # print(indX)

            #___
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            indInd = indX*np.ones_like(grid_x, dtype=bool)[:,None,None] * indY[None,:,None] * indZ[None,None,:]
            cut = field[indX]
            cut = cut[indY]
            cut = cut[:,indZ]

            # eyzy = B[1,25]
            # eyzy = eyzy[indY]
            # eyzy = eyzy[:,indZ]
            # eyzy = eyzy[::10]
            # eyzy = eyzy[:,::10]
            # eyzz = B[2,25]
            # eyzz = eyzz[indY]
            # eyzz = eyzz[:,indZ]
            # eyzz = eyzz[::10]
            # eyzz = eyzz[:,::10]

            if plotContour:
                cut[cut<vmin] = vmin
                cut[cut>vmax] = vmax
                levels = mpl.ticker.MaxNLocator(nbins=21).tick_values(vmin, vmax)
                cs = ax.contourf(cut, y, z, levels=levels, zdir='x', offset=1.5*grid_x[int(indX)], vmin=vmin, vmax=vmax, cmap=cmap, extent='both')
            else:

                x = np.ones_like(y)*grid_x[int(indX)]
                colors = cmap(norm(cut))

                ax.plot_surface(x.T, z.T, y.T, cstride=stride, rstride=stride, facecolors=colors, shade=False, rasterized=True)
            if 0:#indX== indXX[1]:
                # xx, yy, zz = np.meshgrid(grid_x[indX], grid_y[indY][::10], grid_z[indZ][::10])
                # print(indX)
                ax.quiver(grid_x[indX], np.ones_like(eyzy)*yNew[None,:,None], np.ones_like(eyzy)*zNew[None,None,:], np.zeros_like(grid_x[indX]),
                    eyzy*5e8, eyzz*5e8,
                    color='k')


            #___
            X = np.ones(5)*grid_x[indX]
            ax.plot(1.*X, Z, Y, 'k', lw=.2, zorder=30)
            # ax.plot(X, Y, Z2, 'k', lw=.2)
            #ax.plot(X, YFrame, ZFrame, 'r', zorder=30)# 'none', lw=.2)
            X = np.ones(2)*grid_x[indX]
            # ax.plot(1.*X, [-winDimY,winDimY], [0,0], 'k', lw=.2, zorder=30)
            ax.plot(1.*X, [winDimZMin,winDimZMax], [0,0], 'k', lw=.2, zorder=30)
            # ax.plot(X, [0,0], [-winDimZ,winDimZ], 'k', lw=.2)
            ax.plot(1.*X, [0,0], [winDimYMin, winDimYMax], 'k', lw=.2, zorder=30)
            #ax.plot(XFrame, [0,0], [0,0], 'r', zorder=30)# c='none')

        # plt.colorbar(cs)

        plt.axis('off')
        plt.tight_layout()
        #
        if save_fig:

            fn = f'{self.local_path}/products/plots/tomo_{label}_{axis}-axis_{self.idx_it:04}.png'
            print(f'Saving {fn}...')
            plt.savefig(fn)

            if save_as_pdf:
                fn = f'{self.local_path}/products/plots/tomo_{label}_{axis}-axis_{self.idx_it:04}.pdf'
                print(f'Saving {fn}...')
                plt.savefig(fn)


            # plt.savefig(f'/home/etienneb/Desktop/plots_tmp/{label}_{self.idx_it:04}.png')
            # plt.savefig(f'../run_024/products/plots/{label}_{self.idx_it:04}.png')
        else:
            plt.show()




    def plt_time_space(self):

        fig, ax = plt.subplots(figsize=(12, 12))

        ax.pcolormesh(self.B_time_space[0, 0],
                      cmap=oC.bwr_2, rasterized=True)

        plt.tight_layout()
        plt.show()



    def plt_field_spectrum_1d(self, field, label=''):

        PSD = np.fft.fftshift(fft(field))
        PSD = np.abs(PSD) #2./xLen *
        #
        len_x = self.grid_x.size
        dx = grid[1] - grid[0]
        f = np.fft.fftshift(np.fft.fftfreq(len_x, d=dx))
        #
        k_vec = f*2.*np.pi

        fig, AX = plt.subplots(2, 1, figsize=(16, 18))

        AX[0].plot(self.grid_x,

                   field, oC.rgb[0], label=label)

        AX[1].plot(k_vec[int(len_x/2.):], PSD[int(len_x/2.):], oC.rgb[0])

        AX[0].set_xlabel('time')
        AX[0].set_ylabel('field')
        AX[1].set_xlabel('k.di_0')
        AX[1].set_ylabel('PDs')
        AX[0].legend()
        AX[1].set_xscale('log')
        AX[1].set_yscale('log')

        mT.set_spines(AX)
        plt.show()



    def plt_field_2d_spectrum(self, field_label='Bx', save_fig=False):
        '''(kx, ky)-plane'''

        print('\nPlot omni_directional spectrum.\n')

        if field_label=='Bx':
            # field = self.recompose_field(self.B[:, 0])
            field = self.load_field('Bx')
        if field_label=='Bz':
            # field = self.recompose_field(self.B[:, 2])
            field = self.load_field('By')
        elif field_label=='B_perp_sqr':
            # B = self.recompose_field(self.B)
            B = self.load_field('B')
            if self.mp.NB_DIM==3:
                B = B[:, :, :, int(self.mp.len_z_cst*self.mp.mpi_nb_proc_z/2.)]
            # B = self.stag_vector(B)
            field_x = B[0]
            field_y = B[1]
            field = field_x**2 + field_y**2
            print(np.amax(np.sqrt(field)))
            # field = np.roll(field, 200, axis=0)
        elif field_label=='E_perp_sqr':
            field_x = self.recompose_field(self.E[:, 0])
            field_y = self.recompose_field(self.E[:, 1])
            field = field_x**2 + field_y**2
        elif field_label=='ui_perp_sqr':
            Ji = self.load_field('Ji')
            dens = self.load_field('density')
            field_x = Ji[0]/dens
            field_y = Ji[1]/dens
            field = field_x**2 + field_y**2
        elif field_label=='B_perp_sqr_x4':
            field_x = self.recompose_field(self.B[:, 0])
            field_y = self.recompose_field(self.B[:, 1])
            bsqr = field_x**2 + field_y**2
            field = np.zeros((2000, 2000))
            field[:1000, :1000] = bsqr
            field[:1000, 1000:] = bsqr
            field[1000:, :1000] = bsqr
            field[1000:, 1000:] = bsqr
        elif field_label=='Jtot_z':
            Jtot = self.recompose_field(self.Jtot)
            Jtot = self.stag_vector(Jtot)
            # Jtot = self.smooth_vector(Jtot)
            field = Jtot[2]

        try:
            dic = np.load(f'{self.local_path}/spectra.npy', allow_pickle=True).item()
        except FileNotFoundError:
            self.produce_spectra()
            dic = np.load(f'{self.local_path}/spectra.npy', allow_pickle=True).item()

        bin_centres = dic['bin_centres']




        fig, AX = plt.subplots(1, 2, figsize=(20,10))
        AX[0].set_aspect('equal')
        # AX[1].set_aspect('equal')

        # AX[0].pcolormesh(self.grid_x_box, self.grid_y_box, field.T,
        #                  vmin=0., vmax=.8,
        #                  cmap=oC.bwr_2, rasterized=True)
        AX[0].imshow((field).T,
                    vmin = 0., vmax = 1.,
                    extent=(0., self.grid_x_box[-3], 0., self.grid_y_box[-3]),
                    interpolation='bilinear',
                    rasterized=True, cmap=oC.bwr_2, origin='lower')
        # AX[1].pcolormesh(f, f, np.log10(PSDB).T, cmap='RdBu_r')#, vmin = 0, vmax = 10000)

        x, y = mT.logSlidingWindow(bin_centres, dic['B_perp_sqr'], halfWidth=.05)
        AX[1].plot(x, y,  label='B_perp_sqr',  c=oC.rgb[0])
        AX[1].plot(bin_centres, dic['B_perp_sqr'], c=oC.rgb[0], alpha=.2)
        # AX[1].plot(dic['bin_centres_perp1'], dic['B_perp_sqr_perp1'], '--',  label='B_perp_sqr perp1',  c=oC.rgb[2])
        # AX[1].plot(dic['bin_centres_perp2'], dic['B_perp_sqr_perp2'], '.-',  label='B_perp_sqr perp2',  c=oC.rgb[2])
        x, y = mT.logSlidingWindow(bin_centres, dic['ui_perp_sqr'], halfWidth=.05)
        AX[1].plot(x, y,  label='ui_perp_sqr',  c=oC.rgb[2])
        AX[1].plot(bin_centres, dic['ui_perp_sqr'], c=oC.rgb[2], alpha=.2)

        # x, y = mT.logSlidingWindow(bin_centres, dic['Jtot_z'], halfWidth=.05)
        # AX[1].plot(x, y,  label='Jtot_z',  c=oC.rgb[0])
        # AX[1].plot(bin_centres, dic['Jtot_z'], c=oC.rgb[0], alpha=.2)

        max_k = np.pi/self.mp.dX
        min_k = np.pi/(self.grid_x_box[-3])
        # AX[1].axvline(1., color='k', linewidth=1)
        # AX[1].axvline(max_k, color='k', linewidth=1)
        # AX[1].axvline(min_k, color='k', linewidth=1)


        lamb = 4.*self.mp.dX
        k = 2*np.pi/lamb
        AX[1].axvline(k, c='k', lw=.5)


        try:
            d = np.load(f'{self.local_path}/ppc_noise.npy', allow_pickle=True).item()
            bc = d['bin_centres']
            s = d['spectrum']
            x, y = mT.logSlidingWindow(bc, s, halfWidth=.05)
            AX[1].plot(x[x>1.], y[x>1.], label='Particle noise', c='k', lw=1)
            #d = np.load(f'{self.local_path}/spec_B_init.npy', allow_pickle=True).item()
            #bc = d['bin_centres']
            #s = d['spectrum']
            ## AX[1].plot(bc[bc<.2], s[bc<.2], c='k', lw=1)
            #AX[1].plot(bc, s, '--r', lw=1)
            #print('yea')
        except:
            pass

        # AX[1].plot(bin_centres, psd, c='k')

        if 0: ## Slope guide-lines, run_020
            #
            x = np.array([5e-2, 4e0])
            y = x**(-5/3) * 7.e-2
            AX[1].plot(x, y, 'k', lw=.5)
            # x = np.array([5e-2, 2e0])
            # y3 = x**(-3/2) * 6.e-2
            # AX[1].plot(x, y3, 'k', lw=.5)
            x2 = np.array([1.e0, 3e0])
            y2 = x2**(-3.6) * 3.6e-2
            AX[1].plot(x2, y2, 'k', lw=.5)
            # x2 = np.array([1.e0, 2e0])
            # y2 = x2**(-4.5) * 9.e-3
            # AX[1].plot(x2, y2, 'k', lw=.5)

        if 0: ## Slope guide-lines, run_030
            #
            x = np.array([5e-2, 4e1])
            y = x**(-5/3) * 2.2e0
            AX[1].plot(x, y, 'k', lw=.5)
            ##
            # x = np.array([5e-2, 2e0])
            # y3 = x**(-3/2) * 2.e0
            # AX[1].plot(x, y3, 'k', lw=.5)
            ##
            x2 = np.array([2.e0, 8e0])
            y2 = x2**(-4.2) * 1.e0
            AX[1].plot(x2, y2, 'k', lw=.5)
            ##
            # x2 = np.array([1.e0, 2e0])
            # y2 = x2**(-4.5) * 1.e0
            # AX[1].plot(x2, y2, 'k', lw=.5)
            ##
            AX[1].set_ylim([1.e-3, 6.e2])

        if 0: ## Slope guide-lines, run_025
            #
            x = np.array([5e-2, 4e0])
            y = x**(-5/3) * 3.e2
            AX[1].plot(x, y, 'k', lw=.5)
            x = np.array([5e-2, 2e0])
            y3 = x**(-3/2) * 2.e0
            AX[1].plot(x, y3, 'k', lw=.5)
            x2 = np.array([1.e0, 3e0])
            y2 = x2**(-3.2) * 1.e0
            AX[1].plot(x2, y2, 'k', lw=.5)
            x2 = np.array([1.e0, 2e0])
            y2 = x2**(-4.5) * 1.e0
            AX[1].plot(x2, y2, 'k', lw=.5)
            # AX[1].set_ylim([1.e-3, 6.e2])

        if 1: ## Slope guide-lines, run_050
            #
            x = np.array([5e-2, 4e0])
            y = x**(-5/3) * 8.e5
            AX[1].plot(x, y, 'k', lw=.5)
            #
            #x = np.array([5e-2, 2e0])
            #y3 = x**(-3/2) * 2.e3
            #AX[1].plot(x, y3, 'k', lw=.5)
            #
            x2 = np.array([1.e0, 3e0])
            y2 = x2**(-4) * 1.e6
            AX[1].plot(x2, y2, 'k', lw=.5)
            #
            #x2 = np.array([1.e0, 2e0])
            #y2 = x2**(-4.5) * 1.e3
            #AX[1].plot(x2, y2, 'k', lw=.5)
            # AX[1].set_ylim([1.e-3, 6.e2])

        AX[1].set_xlabel('k d_i')


        # for j in [0,400,1200,2400,4800,9600,12000]:
        #     B = np.load('{}/B_{}.npy'.format(path,j))
        #     bPerp = B/B0
        #     bPerp[2] = 0
        #     bPerp = mT.norm(bPerp)
        #     r = np.linspace(-1,1,PSD.shape[0])
        #     r = np.sqrt(r[:,None]**2+r[None,:]**2)
        #     PSD = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(bPerp)))**2 )
        #     spectrum, binEdges, zob = scipy.stats.binned_statistic(r.flatten(), PSD.flatten(), bins=np.logspace(-3,1,50), statistic='mean')
        #     binCenters = .5*(binEdges[1:]+binEdges[:-1])
        #
        #     # AX[2].plot(r.flatten(), PSD.flatten(), '+', alpha=.2)
            # AX[2].plot(binCenters, spectrum, label=j)


        ## AX[1].set_ylim([3.2e-6, 1e0])
        ## AX[1].set_ylim([3.2e-6, 1.])
        ## AX[1].set_ylim([5.e-5, 40.])
        # AX[1].set_ylim([2.e-4, 1.e3])
        AX[1].set_xlim([min_k, max_k])
        AX[1].set_xscale('log')
        AX[1].set_yscale('log')

        plt.legend()
        mT.set_spines(AX)
        plt.tight_layout()
        #
        if save_fig:
            fn = f'{self.local_path}/products/plots/omni_dir_spec_{self.idx_it:04}.png'
            print(f'Saving {fn}...')
            plt.savefig(fn)
            # plt.savefig(f'/home/etienneb/Desktop/plots_tmp/{label}_{self.idx_it:04}.png')
            # plt.savefig(f'../run_024/products/plots/{label}_{self.idx_it:04}.png')
        else:
            plt.show()



    def plt_field_spectrum_para_perp(self, save_fig=False, save_as_pdf=False):

        print('\nPlot para perp spectrum.\n')

        B = self.load_field('B')

        if self.mp.NB_DIM==3:
            B = B[:, :, :, int(self.mp.len_z_cst*self.mp.mpi_nb_proc_z/2.)]
        B = self.stag_vector(B)
        # field_x = B[0]
        # field_y = B[1]
        # field = field_x**2 + field_y**2
        field = mT.norm(B)

        try:
            dic = np.load(f'{self.local_path}/spectra.npy', allow_pickle=True).item()
        except FileNotFoundError:
            self.produce_spectra()
            dic = np.load(f'{self.local_path}/spectra.npy', allow_pickle=True).item()

        bin_centres = dic['bin_centres']


        fig, AX = plt.subplots(1, 2, figsize=(20,10))
        AX[0].set_aspect('equal')

        AX[0].imshow(np.log10(field).T,
                    vmin = -.4, vmax = .4,
                    extent=(0., self.grid_x_box[-3], 0., self.grid_y_box[-3]),
                    interpolation='none',#'bilinear',
                    rasterized=True, cmap=oC.bwr_2, origin='lower')

        x, y = mT.logSlidingWindow(bin_centres, dic['B']['spec_para'], halfWidth=.05)
        AX[1].plot(x, y, '--',  label='B_para',  c=oC.rgb[0])
        AX[1].plot(bin_centres, dic['B']['spec_para'], '--', c=oC.rgb[0], alpha=.2)

        x, y = mT.logSlidingWindow(bin_centres, dic['B']['spec_perp'], halfWidth=.05)
        AX[1].plot(x, y,  label='B_perp',  c=oC.rgb[0])
        AX[1].plot(bin_centres, dic['B']['spec_perp'], c=oC.rgb[0], alpha=.2)

        x, y = mT.logSlidingWindow(bin_centres, dic['ui']['spec_para'], halfWidth=.05)
        AX[1].plot(x, y, '--',  label='ui_para',  c=oC.rgb[2])
        AX[1].plot(bin_centres, dic['ui']['spec_para'], '--', c=oC.rgb[2], alpha=.2)

        x, y = mT.logSlidingWindow(bin_centres, dic['ui']['spec_perp'], halfWidth=.05)
        AX[1].plot(x, y,  label='ui_perp',  c=oC.rgb[2])
        AX[1].plot(bin_centres, dic['ui']['spec_perp'], c=oC.rgb[2], alpha=.2)

        max_k = np.pi/self.mp.dX
        min_k = np.pi/(self.grid_x_box[-3])
        # AX[1].axvline(1., color='k', linewidth=1)
        AX[1].axvline(max_k, color='k', linewidth=1)
        AX[1].axvline(min_k, color='k', linewidth=1)


        try:
            d = np.load(f'{self.local_path}/ppc_noise.npy', allow_pickle=True).item()
            bc = d['bin_centres']
            s = d['spectrum']
            x, y = mT.logSlidingWindow(bc, s, halfWidth=.05)
            AX[1].plot(x[x>.05], y[x>.05], label='Particle noise', c='k', lw=1)

            d = np.load(f'{self.local_path}/spec_B_init.npy', allow_pickle=True).item()
            bc = d['bin_centres']
            s = d['spectrum']
            x, y = mT.logSlidingWindow(bc, s, halfWidth=.05)
            AX[1].plot(x[x>.001], y[x>.001], '--', label='Initial B', c='k', lw=1)

        except:
            pass


        if 0: ## Slope guide-lines, run_050
            #
            x = np.array([3e-3, 1e0])
            y = x**(-5/3) * 8.e5
            AX[1].plot(x, y, 'k', lw=.5)
            #
            x = np.array([3e-3, 1e0])
            y = x**(2) * 8.e5
            AX[1].plot(x, y, 'k', lw=.5)
            #
        if 1: ## Slope guide-lines, run_090
            #
            x = np.array([3e-3, 1e0])
            y = x**(-5/3) * 1.e8
            AX[1].plot(x, y, 'k', lw=.5)

        AX[1].set_xlabel('k d_i')


        # for j in [0,400,1200,2400,4800,9600,12000]:
        #     B = np.load('{}/B_{}.npy'.format(path,j))
        #     bPerp = B/B0
        #     bPerp[2] = 0
        #     bPerp = mT.norm(bPerp)
        #     r = np.linspace(-1,1,PSD.shape[0])
        #     r = np.sqrt(r[:,None]**2+r[None,:]**2)
        #     PSD = np.log10(np.abs(np.fft.fftshift(np.fft.fft2(bPerp)))**2 )
        #     spectrum, binEdges, zob = scipy.stats.binned_statistic(r.flatten(), PSD.flatten(), bins=np.logspace(-3,1,50), statistic='mean')
        #     binCenters = .5*(binEdges[1:]+binEdges[:-1])
        #
        #     # AX[2].plot(r.flatten(), PSD.flatten(), '+', alpha=.2)
            # AX[2].plot(binCenters, spectrum, label=j)


        AX[1].set_xlim([min_k, max_k])
        AX[1].set_xscale('log')
        AX[1].set_yscale('log')

        plt.title('Power Spectral Densities')

        plt.legend()
        mT.set_spines(AX)
        plt.tight_layout()
        #
        if save_fig:
            fn = f'{self.local_path}/products/plots/omni_dir_spec_{self.idx_it:04}.png'
            print(f'Saving {fn}...')
            plt.savefig(fn)

            if save_as_pdf:
                fn = f'{self.local_path}/products/plots/omni_dir_spec_{self.idx_it:04}.pdf'
                print(f'Saving {fn}...')
                plt.savefig(fn)
            # plt.savefig(f'/home/etienneb/Desktop/plots_tmp/{label}_{self.idx_it:04}.png')
            # plt.savefig(f'../run_024/products/plots/{label}_{self.idx_it:04}.png')
        else:
            plt.show()





    def produce_spectra(self):

        len_x = self.mp.len_x_cst
        len_y = self.mp.len_y_cst*self.mp.mpi_nb_proc_y
        dx = self.mp.dX
        grid_x = self.grid_x_box[2:-2]# self.grid_x[0, 2:-2]
        grid_y = self.grid_y_box[2:-2]#np.zeros(self.mp.len_y_cst*self.nb_proc)
        #
        f_x = np.fft.fftshift(np.fft.fftfreq(self.mp.len_x_cst, d=dx))
        f_y = np.fft.fftshift(np.fft.fftfreq(self.mp.mpi_nb_proc_y*self.mp.len_y_cst, d=dx))
        f2d = np.ones((self.mp.len_x_cst, self.mp.mpi_nb_proc_y*self.mp.len_y_cst))
        f2d = np.sqrt(f_x[:,None]**2+f_y[None,:]**2)
        ##
        max_k = 1/(2*self.mp.dX)
        min_k = 1/(2*grid_x[-3])

        bin_edges = np.linspace(f_x[0], f_x[-1], 1000)
        ##
        bin_centres = .5*(bin_edges[1:]+bin_edges[:-1])
        bin_centres *= 2*np.pi  ## From spatial frequency to k

        dic = {'bin_centres': bin_centres}

        for field_label in ['B_perp_sqr', 'ui_perp_sqr']:#, 'Jtot_z']:#, 'B_perp_sqr_x4']:

            if self.mp.NB_DIM==2:

                if field_label=='B_perp_sqr':
                    B = self.load_field('B')[:, 2:-2, 2:-2]
                    # B = self.stag_vector(B)
                    field_x = B[0]
                    field_y = B[1]
                    field = field_x**2 + field_y**2
                    # field = np.roll(field, 200, axis=0)
                elif field_label=='E_perp_sqr':
                    E = self.load_field('E')[:, 2:-2, 2:-2]
                    field_x = E[0]#self.recompose_field(self.E[:, 0])
                    field_y = E[1]#self.recompose_field(self.E[:, 1])
                    field = field_x**2 + field_y**2
                elif field_label=='ui_perp_sqr':
                    Ji = self.load_field('Ji')[:, 2:-2, 2:-2]
                    dens = self.load_field('density')[2:-2, 2:-2]
                    field_x = Ji[0]/dens
                    field_y = Ji[1]/dens
                    field = field_x**2 + field_y**2
                elif field_label=='Jtot_z':
                    Jtot = self.recompose_field(self.Jtot)
                    Jtot = self.stag_vector(Jtot)
                    # Jtot = self.smooth_vector(Jtot)
                    field = Jtot[2]


                if 0: ## PSD of B_perp
                    PSD = np.fft.fftshift(fft2(field))
                    PSD = np.abs(PSD)**2
                    spectrum, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)
                    # spectrum /= spectrum[int(spectrum.size/2)+6]
                    dic[field_label] = spectrum
                elif field_label=='Jtot_z':
                    PSD = np.fft.fftshift(fft2(field))
                    PSD =  np.abs(PSD)**2
                    spectrum, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)
                    dic[field_label] = spectrum

                else: ## PSD B_x + PSD B_y

                    PSD_x = np.fft.fftshift(fft2(field_x))
                    PSD_x =  np.abs(PSD_x)**2#dx**2/(len_x*len_y) *
                    PSD_y = np.fft.fftshift(fft2(field_y))
                    PSD_y =  np.abs(PSD_y)**2#dx**2/(len_x*len_y) *
                    # PSD = np.sqrt(PSD_x**2+PSD_y**2)
                    PSD = PSD_x + PSD_y
                    PSD /= 1e9
                    # PSD *= dx**2/(len_x*len_y)
                    # grid_k = np.ones((2, f_x.size, f_y.size))
                    # grid_k[0] = f_x[:, None]*2*np.pi
                    # grid_k[1] = f_y[None, :]*2*np.pi
                    # # print(grid_k)
                    # spectre_obj = Spectrum()
                    # spectre_obj.interpolate_cart_2D(grid_k, PSD, 'lin')
                    # spectrum = np.sum(spectre_obj.vdf_interp, axis=1)
                    # spectrum /= spectrum[1]
                    # dic[field_label] = spectrum
                    # dic['bin_centres'] = spectre_obj.grid_spher[0,:,0]
                    # print(f2d.flatten().shape, PSD.flatten().shape, bin_edges.shape)
                    spectrum, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)
                    # spectrum /= spectrum[int(spectrum.size/2)+1]
                    dic[field_label] = spectrum

                # PSD = np.fft.fftshift(fft2(field))
                # PSD = 2.*dx/len_x * np.abs(PSD)**2
                # dic['bin_centres'] = f_x*2*np.pi
                # dic[field_label] = PSD[500]


                PSD = np.fft.fftshift(fft(field[0]))
                spectrum = 2.*dx/len_x * np.abs(PSD)**2
                for i in range(1, self.mp.len_x_cst):
                    PSD = np.fft.fftshift(fft(field[i]))
                    spectrum += 2.*dx/len_x * np.abs(PSD)**2
                spectrum /= self.mp.len_x_cst
                dic[f'{field_label}_perp1'] = spectrum
                dic['bin_centres_perp1'] = 2*np.pi*f_x

                PSD = np.fft.fftshift(fft(field[:, 0]))
                spectrum = 2.*dx/len_x * np.abs(PSD)**2
                for i in range(1, self.mp.len_x_cst):
                    PSD = np.fft.fftshift(fft(field[:, i]))
                    spectrum += 2.*dx/len_x * np.abs(PSD)**2
                spectrum /= self.mp.len_x_cst
                dic[f'{field_label}_perp2'] = spectrum
                dic['bin_centres_perp2'] = 2*np.pi*f_x




            if self.mp.NB_DIM==3:

                if field_label=='B_perp_sqr':
                    B = self.load_field('B')[:, 2:-2, 2:-2, 2:-2]
                    # B = self.stag_vector(B)
                    field_xk = B[0]
                    field_yk = B[1]
                    #field = field_xk**2 + field_yk**2
                    # field = np.roll(field, 200, axis=0)
                elif field_label=='E_perp_sqr':
                    E = self.load_field('E')[:, 2:-2, 2:-2, 2:-2]
                    field_xk = E[0]#self.recompose_field(self.E[:, 0])
                    field_yk = E[1]#self.recompose_field(self.E[:, 1])
                    #field = field_xk**2 + field_yk**2
                elif field_label=='ui_perp_sqr':
                    Ji = self.load_field('Ji')[:, 2:-2, 2:-2, 2:-2]
                    dens = self.load_field('density')[2:-2, 2:-2, 2:-2]
                    field_xk = Ji[0]/dens
                    field_yk = Ji[1]/dens
                    #field = field_xk**2 + field_yk**2
                elif field_label=='Jtot_z':
                    Jtot = self.recompose_field(self.Jtot)
                    Jtot = self.stag_vector(Jtot)
                    # Jtot = self.smooth_vector(Jtot)
                    field = Jtot[2]

                spectrum = np.zeros(bin_edges.size-1)

                #nb_z = self.mp.len_z_cst*self.mp.mpi_nb_proc_z
                nb_z = 20
                pad = int(self.mp.len_z_cst*self.mp.mpi_nb_proc_z/nb_z)
                for k in range(nb_z):
                    print(k, end='\r')
                    field_x = field_xk[:, :, k*pad]
                    field_y = field_yk[:, :, k*pad]

                    PSD_x = np.fft.fftshift(fft2(field_x))
                    PSD_x =  np.abs(PSD_x)**2
                    PSD_y = np.fft.fftshift(fft2(field_y))
                    PSD_y =  np.abs(PSD_y)**2

                    PSD = PSD_x + PSD_y

                    specspec, bin_edges, fff = scipy.stats.binned_statistic(f2d.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)

                    spectrum += specspec

                spectrum /= nb_z
                dic[field_label] = spectrum


        np.save(f'{self.local_path}/spectra.npy', dic)

        if self.idx_it==0:
            np.save(f'{self.local_path}/spec_B_init.npy', {'bin_centres': bin_centres, 'spectrum': dic['B_perp_sqr']})
            np.save(f'{self.local_path}/ppc_noise.npy',   {'bin_centres': bin_centres, 'spectrum': dic['ui_perp_sqr']})



    def produce_spectra_para_perp(self):

        ## The difference from the original produce_spectra() method here above
        ## is that here we consider the guiding field to be within the plane of
        ## the simulation.

        len_x = self.mp.len_x_cst
        len_y = self.mp.len_y_cst*self.mp.mpi_nb_proc_y
        dx = self.mp.dX
        grid_x = self.grid_x_box[2:-2]# self.grid_x[0, 2:-2]
        grid_y = self.grid_y_box[2:-2]#np.zeros(self.mp.len_y_cst*self.nb_proc)
        #
        f_x = np.fft.fftshift(np.fft.fftfreq(self.mp.len_x_cst, d=dx))
        f_y = np.fft.fftshift(np.fft.fftfreq(self.mp.mpi_nb_proc_y*self.mp.len_y_cst, d=dx))
        f2d = np.ones((self.mp.len_x_cst, self.mp.mpi_nb_proc_y*self.mp.len_y_cst))
        f2d = np.sqrt(f_x[:,None]**2+f_y[None,:]**2)

        alpha = np.arctan2(f_y[None, :], f_x[:, None])
        theta = np.arccos(self.mp.B0_x)

        f_para = np.abs(f2d*np.cos(alpha-theta))
        f_perp = np.abs(f2d*np.sin(alpha-theta))

        # plt.pcolormesh(f_perp)
        # plt.show()

        bin_edges = np.linspace(f_x[0], f_x[-1], 1000)
        ##
        bin_centres = .5*(bin_edges[1:]+bin_edges[:-1])
        bin_centres *= 2*np.pi  ## From spatial frequency to k

        dic = {'bin_centres': bin_centres}



        for field_label in ['B', 'ui']:#, 'Jtot_z']:#, 'B_perp_sqr_x4']:

            if self.mp.NB_DIM==2:

                if field_label=='B':

                    B = self.load_field('B')[:, 2:-2, 2:-2]
                    field_x = B[0]
                    field_y = B[1]
                    field = field_x**2 + field_y**2

                elif field_label=='ui':

                    Ji = self.load_field('Ji')[:, 2:-2, 2:-2]
                    dens = self.load_field('density')[2:-2, 2:-2]
                    field_x = Ji[0]/dens
                    field_y = Ji[1]/dens
                    field = field_x**2 + field_y**2

                PSD_x = np.fft.fftshift(fft2(field_x))
                PSD_x =  np.abs(PSD_x)**2
                PSD_y = np.fft.fftshift(fft2(field_y))
                PSD_y =  np.abs(PSD_y)**2

                PSD = PSD_x + PSD_y

                spec_para, bin_edges, fff = scipy.stats.binned_statistic(f_para.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)

                spec_perp, bin_edges, fff = scipy.stats.binned_statistic(f_perp.flatten(), PSD.flatten(), statistic='sum', bins=bin_edges)

                dic[field_label] = {'spec_para': spec_para, 'spec_perp': spec_perp}



            if self.mp.NB_DIM==3:
                sys.exit('not implemented yet.')


        np.save(f'{self.local_path}/spectra.npy', dic)

        if self.idx_it==0:
            np.save(f'{self.local_path}/spec_B_init.npy', {'bin_centres': bin_centres, 'spectrum': dic['B']['spec_perp']})
            np.save(f'{self.local_path}/ppc_noise.npy',   {'bin_centres': bin_centres, 'spectrum': dic['ui']['spec_perp']})



    def produce_spectra_para_perp_interp(self):

        """ The goal: using interpolation instead of binning for PSDs
        """

        from scipy.interpolate import RegularGridInterpolator, NearestNDInterpolator, LinearNDInterpolator


        if self.mp.NB_DIM==2:


            k_max = 1/(2*self.mp.dX)
            resolution= 101
            edgesX = np.linspace(-k_max, k_max, resolution + 1, dtype=np.float32)
            centersX = (edgesX[:-1] + edgesX[1:]) * .5

            ## grid is the original, non-transformed interpolation grid coordinates
            ## grid_t will be transformed into a B-dield aligned frame, with B0 along x

            grid = np.mgrid[-k_max:k_max:resolution*1j,
                            -k_max:k_max:resolution*1j]
            grid_t = grid.copy()

            vector_orig = np.array([1., 0., 0.])
            vector_fin  = np.array([self.mp.B0_x, self.mp.B0_y, self.mp.B0_z])

            R = R_2vect(vector_orig, vector_fin)

            gc = grid_t.copy()
            grid_t = np.dot(R[:2, :2], gc.reshape(2, -1)).reshape(grid.shape)
            ## Now grid_t has its x dimension along B.

            bin_centres = 2*np.pi*centersX ## From spatial frequency to k
            dic = {'bin_centres': bin_centres}

            f_x = np.fft.fftshift(np.fft.fftfreq(self.mp.len_x_cst, d=self.mp.dX))
            f_y = np.fft.fftshift(np.fft.fftfreq(self.mp.mpi_nb_proc_y*self.mp.len_y_cst, d=self.mp.dX))

            for field_label in ['B', 'ui']:


                if field_label=='B':

                    B = self.load_field('B')[:, 2:-2, 2:-2]
                    field_x = B[0]
                    field_y = B[1]
                    field_z = B[2]
                    field_x -= np.mean(field_x)
                    field_y -= np.mean(field_y)
                    field_z -= np.mean(field_z)
                    #field = field_x**2 + field_y**2 + field_z**2

                elif field_label=='ui':

                    Ji = self.load_field('Ji')[:, 2:-2, 2:-2]
                    dens = self.load_field('density')[2:-2, 2:-2]
                    field_x = Ji[0]/dens
                    field_y = Ji[1]/dens
                    field_z = Ji[2]/dens
                    field_x -= np.mean(field_x)
                    field_y -= np.mean(field_y)
                    field_z -= np.mean(field_z)
                    #field = field_x**2 + field_y**2 + field_z**2

                PSD_x = np.fft.fftshift(fft2(field_x))
                PSD_x =  np.abs(PSD_x)**2
                PSD_y = np.fft.fftshift(fft2(field_y))
                PSD_y =  np.abs(PSD_y)**2
                PSD_z = np.fft.fftshift(fft2(field_z))
                PSD_z =  np.abs(PSD_z)**2


                PSD = PSD_x + PSD_y + PSD_z

                field_interp = np.zeros((resolution, resolution))
                interpFunc = RegularGridInterpolator( (f_x, f_y), PSD,
                                                        bounds_error=False,
                                                        method='linear', #'nearest'
                                                        fill_value=0.) ## 0. and not nan, summed afterward!

                field_interp = interpFunc(grid_t.reshape(2, -1).T)
                field_interp = field_interp.reshape((resolution, resolution))


                dic[field_label] = {'spec_para': np.sum(field_interp, axis=1),
                                    'spec_perp': np.sum(field_interp, axis=0)}


        if self.mp.NB_DIM==3:


            k_max = 1/(2*self.mp.dX)
            resolution= 101
            edgesX = np.linspace(-k_max, k_max, resolution + 1, dtype=np.float32)
            centersX = (edgesX[:-1] + edgesX[1:]) * .5

            ## grid is the original, non-transformed interpolation grid coordinates
            ## grid_t will be transformed into a B-dield aligned frame, with B0 along x

            # edges_rho = np.linspace(0, k_max, resolution+1, dtype=np.float32)
            # edges_phi = np.linspace(0, 2*np.pi, resolution+1, dtype=np.float32)
            # edges_z = np.linspace(-k_max, k_max, resolution+1, dtype=np.float32)
            # centers_rho = (edges_rho[:-1]+edges_rho[1:])*.5
            # centers_phi = (edges_phi[:-1]+edges_phi[1:])*.5
            # centers_z = (edges_z[:-1]+edges_z[1:])*.5
            # grid_cyl = np.mgrid[centers_rho[0]:centers_rho[-1]:centers_rho.size*1j,
            #                     centers_phi[0]:centers_phi[-1]:centers_phi.size*1j,
            #                     centers_z[0]:centers_z[-1]:centers_z.size*1j]
            # grid_cyl = grid_cyl.astype(np.float32)
            # grid_cart = cyl2cart(grid_cyl)
            # grid_spher = cart2spher(grid_cart)
            # dRho = centers_rho[1]-centers_rho[0]
            # dPhi = centers_phi[1]-centers_phi[0]
            # dZ = centers_z[1]-centers_z[0]
            # dvvv = np.ones((resolution, resolution, resolution)) \
            #         * centers_rho[:, None, None]*dRho*dPhi*dZ

            grid = np.mgrid[-k_max:k_max:resolution*1j,
                            -k_max:k_max:resolution*1j,
                            -k_max:k_max:resolution*1j]
            grid_t = grid.copy()

            vector_orig = np.array([1., 0., 0.])
            vector_fin  = np.array([self.mp.B0_x, self.mp.B0_y, self.mp.B0_z])

            R = R_2vect(vector_orig, vector_fin)

            gc = grid_t.copy()
            grid_t = np.dot(R, gc.reshape(3, -1)).reshape(grid.shape)
            ## Now grid_t has its z dimension along B.

            bin_centres = 2*np.pi*centersX ## From spatial frequency to k
            dic = {'bin_centres': bin_centres}

            f_x = np.fft.fftshift(np.fft.fftfreq(self.mp.len_x_cst, d=self.mp.dX))
            f_y = np.fft.fftshift(np.fft.fftfreq(self.mp.mpi_nb_proc_y*self.mp.len_y_cst, d=self.mp.dX))
            f_z = np.fft.fftshift(np.fft.fftfreq(self.mp.mpi_nb_proc_z*self.mp.len_z_cst, d=self.mp.dX))

            for field_label in ['ui', 'B']:

                print(field_label)
                if field_label=='B':
                    field = self.load_field('B')[:, 2:-2, 2:-2, 2:-2].astype(np.float32)
                    field_x = field[0]
                    field_y = field[1]
                    field_z = field[2]
                    field_x -= np.mean(field_x)
                    field_y -= np.mean(field_y)
                    field_z -= np.mean(field_z)

                elif field_label=='ui':

                    field = self.load_field('Ji')[:, 2:-2, 2:-2, 2:-2].astype(np.float32)
                    field /= self.load_field('density')[2:-2, 2:-2, 2:-2].astype(np.float32)
                    field_x = field[0]
                    field_y = field[1]
                    field_z = field[2]
                    field_x -= np.mean(field_x)
                    field_y -= np.mean(field_y)
                    field_z -= np.mean(field_z)

                print('yo1')
                PSD_x = np.fft.fftshift(fftn(field_x)).astype(np.float32)
                PSD_x =  np.abs(PSD_x)**2
                print('yo1.2')
                PSD_y = np.fft.fftshift(fftn(field_y)).astype(np.float32)
                PSD_y =  np.abs(PSD_y)**2
                print('yo1.3')
                PSD_z = np.fft.fftshift(fftn(field_z)).astype(np.float32)
                PSD_z =  np.abs(PSD_z)**2

                PSD = PSD_x + PSD_y + PSD_z
                print('yo2')
                field_interp = np.zeros((resolution, resolution, resolution), dtype=np.float32)
                interpFunc = RegularGridInterpolator( (f_x, f_y, f_z), PSD,
                                                        bounds_error=False,
                                                        method='linear', #'nearest'
                                                        fill_value=0.) ## 0. and not nan, summed afterward!

                field_interp = interpFunc(grid_t.reshape(3, -1).T).astype(np.float32)
                field_interp = field_interp.reshape((resolution, resolution, resolution))

                # plt.plot(np.sum(field_interp, axis=(1, 2)), 'x')
                # plt.plot(np.sum(field_interp, axis=(0, 2)), 'x')
                # plt.plot(np.sum(field_interp, axis=(0, 1)), 'x')
                # plt.show()

                dic[field_label] = {'spec_para': np.sum(field_interp, axis=(1, 2)),
                                    'spec_perp': np.sum(field_interp, axis=(0, 2))}
                print('yo3')


        np.save(f'{self.local_path}/spectra.npy', dic)

        if self.idx_it==0:
            np.save(f'{self.local_path}/spec_B_init.npy', {'bin_centres': bin_centres, 'spectrum': dic['B']['spec_perp']})
            np.save(f'{self.local_path}/ppc_noise.npy',   {'bin_centres': bin_centres, 'spectrum': dic['ui']['spec_perp']})











    def produce_ohm(self):

        B    = self.load_field('B')
        Ji   = self.load_field('Ji')
        dens = self.load_field('density')
        dx = self.mp.dX

        # B = self.stag_vector(B)

        dyBz = -1./(12.*dx) * ( B[2, 2:-2, 4:  ]-8*B[2, 2:-2, 3:-1]+8*B[2, 2:-2, 1:-3]-B[2, 2:-2,  :-4] )
        dxBz = -1./(12.*dx) * ( B[2, 4:  , 2:-2]-8*B[2, 3:-1, 2:-2]+8*B[2, 1:-3, 2:-2]-B[2,  :-4, 2:-2] )
        dxBy = -1./(12.*dx) * ( B[1, 4:  , 2:-2]-8*B[1, 3:-1, 2:-2]+8*B[1, 1:-3, 2:-2]-B[1,  :-4, 2:-2] )
        dyBx = -1./(12.*dx) * ( B[0, 2:-2, 4:  ]-8*B[0, 2:-2, 3:-1]+8*B[0, 2:-2, 1:-3]-B[0, 2:-2,  :-4] )

        # self.Jtot = np.zeros_like(B)
        # self.Jtot[0, 2:-2, 2:-2] =  dyBz
        # self.Jtot[1, 2:-2, 2:-2] = -dxBz
        # self.Jtot[2, 2:-2, 2:-2] =  dxBy - dyBx
        self.Jtot = self.load_field('Jtot')


        self.E_mot = np.zeros_like(B)
        self.E_mot[0] = Ji[1]*B[2] - Ji[2]*B[1]
        self.E_mot[1] = Ji[2]*B[0] - Ji[0]*B[2]
        self.E_mot[2] = Ji[0]*B[1] - Ji[1]*B[0]
        self.E_mot /= -dens
        # self.E_mot[1] = -6.666*B[2]
        # self.E_mot[2] =  6.666*B[1]

        self.E_hal = np.zeros_like(B)
        self.E_hal[0] = self.Jtot[1]*B[2] - self.Jtot[2]*B[1]
        self.E_hal[1] = self.Jtot[2]*B[0] - self.Jtot[0]*B[2]
        self.E_hal[2] = self.Jtot[0]*B[1] - self.Jtot[1]*B[0]
        # self.E_hal /= dens

        self.E_amb = np.zeros_like(B)
        pres = self.mp.betae0*dens**self.mp.poly_ind
        self.E_amb[0, 2:-2, 2:-2] = - 1./(2.*dens[2:-2, 2:-2]) * 1./(12.*dx) * ( -pres[4:  , 2:-2]+8*pres[3:-1, 2:-2]-8*pres[1:-3, 2:-2]+pres[ :-4, 2:-2] )
        self.E_amb[1, 2:-2, 2:-2] = - 1./(2.*dens[2:-2, 2:-2]) * 1./(12.*dx) * ( -pres[2:-2, 4:  ]+8*pres[2:-2, 3:-1]-8*pres[2:-2, 1:-3]+pres[2:-2,  :-4] )
        self.E_amb[2, 2:-2, 2:-2] = 0.

        self.E_hyp_res = np.zeros_like(B)
        self.E_hyp_res[0, 2:-2, 2:-2] = -1. * 2.e-3 * 1./(12.*dx**2) * (-self.Jtot[0, 4: , 2:-2]+16*self.Jtot[0, 3:-1, 2:-2]-30*self.Jtot[0, 2:-2, 2:-2]-16*self.Jtot[0, 1:-3, 2:-2]+self.Jtot[0,  :-4, 2:-2])
        self.E_hyp_res[1, 2:-2, 2:-2] = -1. * 2.e-3 * 1./(12.*dx**2) * (-self.Jtot[1, 2:-2, 4: ]+16*self.Jtot[1, 2:-2, 3:-1]-30*self.Jtot[1, 2:-2, 2:-2]-16*self.Jtot[1, 2:-2, 1:-3]+self.Jtot[1, 2:-2,  :-4])
        self.E_hyp_res[2, 2:-2, 2:-2] = 0.




    def stag_scalar(self, a):

        b = a.copy()
        c = a.copy()

        b[1:-1, 1:-1] = .25*a[1:-1, 1:-1] + \
                        .25*a[2:  , 1:-1] + \
                        .25*a[1:-1, 2:  ] + \
                        .25*a[2:  , 2:  ]

        c[1:-1, 1:-1] = .25*b[1:-1, 1:-1] + \
                        .25*b[ :-2, 1:-1] + \
                        .25*b[1:-1,  :-2] + \
                        .25*b[ :-2,  :-2]

        return c

    def stag_vector(self, a):

        b = a.copy()
        c = a.copy()

        b[:, 1:-1, 1:-1] = .25*a[:, 1:-1, 1:-1] + \
                           .25*a[:, 2:  , 1:-1] + \
                           .25*a[:, 1:-1, 2:  ] + \
                           .25*a[:, 2:  , 2:  ]

        c[:, 1:-1, 1:-1] = .25*b[:, 1:-1, 1:-1] + \
                           .25*b[:,  :-2, 1:-1] + \
                           .25*b[:, 1:-1,  :-2] + \
                           .25*b[:,  :-2,  :-2]

        return c

    def smooth_vector(self, a):

        b = a.copy()

        b[:, 1:-1, 1:-1] = 4.*a[:, 1:-1, 1:-1] + \
                           2.*a[:, 2:  , 1:-1] + \
                           2.*a[:,  :-2, 1:-1] + \
                           2.*a[:, 1:-1, 2:  ] + \
                           2.*a[:, 1:-1,  :-2] + \
                           1.*a[:, 2:  , 2:  ] + \
                           1.*a[:, 2:  ,  :-2] + \
                           1.*a[:,  :-2, 2:  ] + \
                           1.*a[:,  :-2,  :-2]
        b /= 16.

        return b





    def close_fig(self, event):
        plt.close(self.fig2)

    def draw_streamline(self, event):

        start_point = np.array([[event.xdata, event.ydata]])

        event.inaxes.streamplot(self.grid_x_box, self.grid_y_box, self.B_r[0].T, self.B_r[1].T,
                        start_points=start_point,
                        color='k')#, linewidth=np.log10(B_perp).T)

        event.canvas.draw()

    def low_high_pass_2d(self, a, sigma=10):
        from scipy import ndimage
        lowpass = ndimage.gaussian_filter(a, sigma=sigma)
        highpass = a - lowpass
        return lowpass, highpass

    def interpolate_cut(self, f, line):

        interpFunc = RegularGridInterpolator((self.grid_x_box, self.grid_y_box), f,
                                             bounds_error=False, method='linear',
                                             fill_value=np.nan)
        cut = interpFunc(line.T).T

        return cut







class menura_HK:

    def __init__(self, local_path, remote_path, ask_scp=False, print_param=True,
                 remote_label='jean-zay', full_scp=True):

        self.local_path = local_path
        self.remote_path = remote_path
        self.remote_label = remote_label

        if ask_scp:
            self.scp_data(full_scp=full_scp)

        self.mp = menura_param(f'{local_path}')#, file_name='parameters.txt')
        if print_param:
            self.mp.print_physical_parameters()
        self.nb_proc = self.mp.mpi_nb_proc_tot
        ##
        self.active_part_0 = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.active_part_1 = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        ##
        self.nb_part_comm_up = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.nb_part_comm_down = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.nb_pla_proba = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        ##
        self.energy_E = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.energy_B = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.energy_K = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        ##
        self.rms_sum_B = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.rms_sum_ui = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.rms_sum_Jtot = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.rms_sum_sqr_B = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.rms_sum_sqr_ui = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.rms_sum_sqr_Jtot = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        ##
        self.B_mean = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.B_var = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.div_B_mean = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.div_B_var = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.div_B_max = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.Ji_mean = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.density = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.vel_part = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        self.vel_th_part = np.zeros((self.nb_proc, 3, self.mp.len_save_t_cst))
        ##
        self.simu_time = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        self.run_time = np.zeros((self.nb_proc, self.mp.len_save_t_cst))
        ##
        self.slice_width = None ;
        if self.mp.inject_turb_cst:
            self.slice_width = np.zeros((self.nb_proc, self.mp.len_x_cst))

        self.time = np.arange(self.mp.len_save_t_cst)*self.mp.dt*self.mp.rate_save_t_cst

        self.iterations = np.arange(self.mp.len_save_t_cst)*self.mp.rate_save_t_cst

        self.run_time = np.zeros((self.mp.len_save_t_cst))

        self.load_data()

    def scp_data(self, full_scp):

        remote_pswd = ''
        sync = input('scp from remote? (y/[n]) ')

        if sync == 'y':
            if self.remote_label=='kebnekaise':
                remote_log = 'behare@kebnekaise.hpc2n.umu.se'
                remote_pswd = input('Kebnekaise password:')
            elif self.remote_label=='jean-zay':
                remote_log = 'ued64ot@jean-zay.idris.fr'
                remote_pswd = input('Jean-Zay password:')

        if sync == 'y':
            print('scp parameters...')
            print(f'{remote_log}:{self.remote_path}/products/parameters.txt {self.local_path}/products')
            os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/parameters.txt {self.local_path}/products')
            ##
            if full_scp:
                print('scp all HK...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/HK/* {self.local_path}/products/HK')
            else:
                print('scp all HK particles...')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/HK/*part* {self.local_path}/products/HK')
                os.system(f'sshpass -p "{remote_pswd}" scp {remote_log}:{self.remote_path}/products/HK/*pla_proba* {self.local_path}/products/HK')

    def load_data(self):

        for r_y in range(self.mp.mpi_nb_proc_y):
            for r_z in range(self.mp.mpi_nb_proc_z):

                if self.mp.new_rank_string_format:
                    rank_str = f'rank_{r_y}_{r_z}'
                else:
                    rank_str = f'rank{r_y}'

                r = r_y*self.mp.mpi_nb_proc_z + r_z

                self.active_part_0[r] = np.load(f'{self.local_path}/products/HK/active_part_0_{rank_str}.npy')
                self.active_part_1[r] = np.load(f'{self.local_path}/products/HK/active_part_1_{rank_str}.npy')
                ##
                self.nb_part_comm_up[r]   = np.load(f'{self.local_path}/products/HK/nb_part_comm_up_{rank_str}.npy')
                self.nb_part_comm_down[r] = np.load(f'{self.local_path}/products/HK/nb_part_comm_down_{rank_str}.npy')
                if self.mp.exosphere_cst :
                    self.nb_pla_proba[r] = np.load(f'{self.local_path}/products/HK/nb_pla_proba_{rank_str}.npy')
                ##
                try:
                    self.energy_E[r] = np.load(f'{self.local_path}/products/HK/energy_elec_{rank_str}.npy')
                    self.energy_B[r] = np.load(f'{self.local_path}/products/HK/energy_mag_{rank_str}.npy')
                    self.energy_K[r] = np.load(f'{self.local_path}/products/HK/energy_kin_{rank_str}.npy')
                    ##
                    self.rms_sum_B[r] = np.load(f'{self.local_path}/products/HK/rms_sum_B_{rank_str}.npy')
                    self.rms_sum_sqr_B[r] = np.load(f'{self.local_path}/products/HK/rms_sum_sqr_B_{rank_str}.npy')
                    self.rms_sum_ui[r] = np.load(f'{self.local_path}/products/HK/rms_sum_ui_{rank_str}.npy')
                    self.rms_sum_sqr_ui[r] = np.load(f'{self.local_path}/products/HK/rms_sum_sqr_ui_{rank_str}.npy')
                    self.rms_sum_Jtot[r] = np.load(f'{self.local_path}/products/HK/rms_sum_Jtot_{rank_str}.npy')
                    self.rms_sum_sqr_Jtot[r] = np.load(f'{self.local_path}/products/HK/rms_sum_sqr_Jtot_{rank_str}.npy')
                    ##
                    self.B_mean[r] = np.load(f'{self.local_path}/products/HK/B_mean_{rank_str}.npy')
                    self.B_var[r] = np.load(f'{self.local_path}/products/HK/B_var_{rank_str}.npy')
                    try:
                        self.div_B_mean[r] = np.load(f'{self.local_path}/products/HK/div_B_mean_{rank_str}.npy')
                        self.div_B_var[r] = np.load(f'{self.local_path}/products/HK/div_B_var_{rank_str}.npy')
                        self.div_B_max[r] = np.load(f'{self.local_path}/products/HK/div_B_max_{rank_str}.npy')
                    except:
                        pass
                    self.Ji_mean[r] = np.load(f'{self.local_path}/products/HK/Ji_mean_{rank_str}.npy')
                    try:
                        self.density[r] = np.load(f'{self.local_path}/products/HK/density_{rank_str}.npy')
                    except:
                        print('density was not produced?')
                    self.vel_part[r] = np.load(f'{self.local_path}/products/HK/vel_part_{rank_str}.npy')
                    self.vel_th_part[r] = np.load(f'{self.local_path}/products/HK/vel_th_part_{rank_str}.npy')
                    ##
                    if self.mp.inject_turb_cst:
                        self.slice_width[r] = np.load(f'{self.local_path}/products/HK/slice_width_rank{r_y}_{r_z}.npy')
                    self.run_time = np.load(f'{self.local_path}/products/HK/run_time.npy')
                except:
                    pass


    def plt_active_part(self, save_fig=False):

        itk = self.active_part_0[0]!=0#np.ones_like(self.time, dtype=bool)

        fig, ax = plt.subplots(figsize=(24, 20))

        ap = np.sum(self.active_part_0, axis=0)

        ax.text(self.time[0], .8*self.active_part_0[0, 0], 'nb active part SW', color=oC.rgb2[2], fontsize=14)
        ax.plot(self.time[0], self.active_part_0[0, 0], '*', c=oC.rgb2[2])
        for rank in range(self.nb_proc):
            ax.plot(self.time[itk], self.active_part_0[rank, itk], c=oC.rgb2[2])

        if self.mp.exosphere_cst  or self.mp.ionosphere_cst:
            for rank in range(self.nb_proc):
                ax.plot(self.time[itk], self.active_part_1[rank, itk], c=oC.rgb2[0])
                ax.plot(self.time[itk], self.active_part_1[rank, itk]+self.active_part_0[rank, itk], 'k', lw=1)
            ax.text(self.time[0], .1*self.active_part_1[0, 0], 'nb active part pla', color=oC.rgb2[0], fontsize=14)
            ax.text(self.time[0], 1.1*(self.active_part_1[0, 0]+self.active_part_0[0, 0]), 'Total', color='k', fontsize=14)
            ax.plot(self.time[0], self.active_part_1[0, 0], '*', c=oC.rgb2[0])

        ax.axhline(self.mp.pool_size_cst, c='k')
        ax.text(self.time[-1], .96*self.mp.pool_size_cst, 'Particle pool size', fontsize=14)

        ax.set_xlabel('Time')
        ax.set_xlim([self.time[0], self.time[-1]])


        ax2 = ax.twiny()
        ax2.plot(self.iterations[itk], self.active_part_0[0, itk], alpha=0)
        ax2.set_xlim([0, self.iterations[-1]])

        ax2.set_xlabel('Iterations')

        mT.set_spines([ax, ax2], two_xaxis=True)

        if save_fig:
            fn = f'{self.local_path}/products/plots/active_particles.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_part_comm(self, save_fig=False):

        itk = self.nb_part_comm_up[0]!=0 #np.ones_like(self.time, dtype=bool)
        fig, ax = plt.subplots(figsize=(24, 20))

        ax.text(self.time[0], 1.2*self.nb_part_comm_up[0, 0], 'nb_part_comm_up', color=oC.rgb2[2], fontsize=14)
        ax.text(self.time[0], 1.6*self.nb_part_comm_down[0, 0], 'nb_part_comm_down', color=oC.rgb2[0], fontsize=14)
        if self.mp.exosphere_cst :
            ax.text(self.time[0], 1.6*self.nb_pla_proba[0, 0], 'sum proba pla', color=oC.rgb2[1], fontsize=14)
        if self.mp.inject_turb_cst:
            ax.text(self.time[0], .9*self.slice_width[0, 0], 'slices widths', color=oC.rgb[0], fontsize=14)

        for rank in range(self.nb_proc):
            ax.plot(self.time[itk], self.nb_part_comm_up[rank, itk], c=oC.rgb2[2])
            ax.plot(self.time[0], self.nb_part_comm_up[rank, 0], '*', c=oC.rgb2[2])
            ax.plot(self.time[itk], self.nb_part_comm_down[rank, itk], c=oC.rgb2[0])
            ax.plot(self.time[0], self.nb_part_comm_down[rank, 0], '*', c=oC.rgb2[0])
            if self.mp.exosphere_cst :
                ax.plot(self.time[itk], self.nb_pla_proba[rank, itk], c=oC.rgb2[1])
                ax.plot(self.time[0], self.nb_pla_proba[rank, 0], '*', c=oC.rgb2[1])

            if self.mp.inject_turb_cst:
                try:
                    ax.plot(self.time, self.slice_width[rank, :int(self.mp.len_save_t_cst)], c=oC.rgb[0])
                except:
                    ax.plot(self.time[:int(self.mp.len_x_cst)], self.slice_width[rank], c=oC.rgb[0])
                    ax.plot(self.time[0], self.slice_width[rank, 0], '*', c=oC.rgb[0])

        ax.axhline(self.mp.buff_size_cst, c='k')
        ax.text(self.time[-10], .96*self.mp.buff_size_cst, 'Buffer size', fontsize=14)

        ax.axhline(self.mp.len_y_cst*self.mp.nb_part_node_cst, linestyle='--', c='k')
        ax.text(self.time[-10], .96*self.mp.len_y_cst*self.mp.nb_part_node_cst, 'Mean injection', fontsize=14)

        if self.mp.exosphere_cst :
            ax.axhline(self.mp.nb_part_add_pla, linestyle='--', c='k')
            ax.text(self.time[-10], .96*self.mp.nb_part_add_pla, 'N part add pla', fontsize=14)

        ax2 = ax.twiny()
        ax2.plot(self.iterations[itk], self.nb_part_comm_up[0][itk], alpha=0)

        ax.set_xlabel('Time')
        ax.set_xlim([self.time[0], self.time[-1]])

        ax2.set_xlabel('Iterations')
        ax2.set_xlim([0, self.iterations[-1]])

        mT.set_spines([ax, ax2], two_xaxis=True)
        if save_fig:
            fn = f'{self.local_path}/products/plots/communicated_particles.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_rms(self, save_fig=False): ## RMS

        #itk = np.ones_like(self.rms_B_perp, dtype=bool)
        try:
            L_correl = self.mp.len_x_cst*self.mp.dX/self.mp.nb_harmonics
        except:
            L_correl = self.mp.len_x_cst*self.mp.dX/4
        non_linear_time = L_correl/self.mp.fluctu_rms

        fig, AX = plt.subplots(2, 1, figsize=(18, 14), sharex=True)

        N = self.mp.nb_nodes_cst*self.mp.mpi_nb_proc_tot
        rms_Bx_sqr = np.sum(self.rms_sum_sqr_B[:, 0], axis=0)/N \
                           - np.sum(self.rms_sum_B[:, 0], axis=0)**2/N**2
        rms_By_sqr = np.sum(self.rms_sum_sqr_B[:, 1], axis=0)/N \
                           - np.sum(self.rms_sum_B[:, 1], axis=0)**2/N**2
        rms_Bz_sqr = np.sum(self.rms_sum_sqr_B[:, 2], axis=0)/N \
                           - np.sum(self.rms_sum_B[:, 2], axis=0)**2/N**2
        rms_B_perp = np.sqrt(rms_Bx_sqr + rms_By_sqr)
        rms_B      = np.sqrt(rms_Bx_sqr + rms_By_sqr + rms_Bz_sqr)

        rms_vx_sqr = np.sum(self.rms_sum_sqr_ui[:, 0], axis=0)/N \
                           - np.sum(self.rms_sum_ui[:, 0], axis=0)**2/N**2
        rms_vy_sqr = np.sum(self.rms_sum_sqr_ui[:, 1], axis=0)/N \
                           - np.sum(self.rms_sum_ui[:, 1], axis=0)**2/N**2
        rms_vz_sqr = np.sum(self.rms_sum_sqr_ui[:, 2], axis=0)/N \
                           - np.sum(self.rms_sum_ui[:, 2], axis=0)**2/N**2
        rms_v_perp = np.sqrt(rms_vx_sqr + rms_vy_sqr)
        rms_v      = np.sqrt(rms_vx_sqr + rms_vy_sqr + rms_vz_sqr)

        rms_Jx_sqr = np.sum(self.rms_sum_sqr_Jtot[:, 0], axis=0)/N \
                           - np.sum(self.rms_sum_Jtot[:, 0], axis=0)**2/N**2
        rms_Jy_sqr = np.sum(self.rms_sum_sqr_Jtot[:, 1], axis=0)/N \
                           - np.sum(self.rms_sum_Jtot[:, 1], axis=0)**2/N**2
        rms_Jz_sqr = np.sum(self.rms_sum_sqr_Jtot[:, 2], axis=0)/N \
                           - np.sum(self.rms_sum_Jtot[:, 2], axis=0)**2/N**2
        rms_J      = np.sqrt(rms_Jx_sqr + rms_Jy_sqr + rms_Jz_sqr)


        AX[0].plot(self.time, rms_B_perp, c=oC.rgb2[0], label='B_rms perp')
        AX[0].plot(self.time, rms_B, '--', c=oC.rgb2[0], label='B_rms tot')
        AX[0].plot(self.time, rms_v_perp, c=oC.rgb2[2], label='ui_rms perp')
        AX[0].plot(self.time, rms_v, '--', c=oC.rgb2[2], label='ui_rms tot')

        AX[1].plot(self.time, rms_J, c=oC.rgb2[0], label='J_rms')
        AX[1].axvline(non_linear_time, c='k', lw=.5)



        ax2 = AX[0].twiny()
        ax2.plot(self.iterations, rms_B_perp, 'r', alpha=0)

        ax2.set_xlabel('Iterations')
        AX[-1].set_xlabel('Time')
        for ax in AX:
            ax.legend()
            ax.set_xlim([self.time[0], self.time[-1]])
        ax2.set_xlim([self.iterations[0], self.iterations[-1]])

        mT.set_spines(AX[1:])
        mT.set_spines([AX[0], ax2], two_xaxis=True)
        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/rms.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_energies(self, save_fig=False):

        # enLim = 10000
        itk = self.energy_E[0]!=0

        # enKin /= 92627.**2
        # enTot = enKin + enMag + enElec
        enElec = np.mean(self.energy_E, axis=0) #np.mean(enElec)
        enMag  = np.sum(self.energy_B, axis=0) #np.mean(enMag)
        enKin  = np.mean(self.energy_K, axis=0) #np.mean(enKin)
        # eTMean = enTot[0] #np.mean(enTot)


        if 0:   # Checking B energy, HK against output field.
            it = 10000
            md = menura_data(self.local_path, '', it, remote_label='jean-zay',
                             ask_scp=False, print_param=False, full_scp=False )
            B = md.load_field('B')
            E_mag_field = B[0]**2 + B[1]**2 + B[2]**2
            E_mag = np.sum(E_mag_field[2:-2, 2:-2])

            print(E_mag, enMag[int(it/md.mp.rate_save_t_cst)])

            sys.exit()


        fig, AX = plt.subplots(3, 1, figsize=(20, 16), sharex=True)

        # enElec = (enElec - eEMean)/eEMean*100
        # enMag = (enMag - eMMean)/eMMean*100
        # enKin = (enKin - eKMean)/eKMean*100
        # enTot = (enTot - eTMean)/eTMean*100
        AX[0].plot(self.time[itk], enElec[itk], c=oC.rgb2[0], label='Electric energy (relative)')
        AX[1].plot(self.time[itk], enMag[itk], c=oC.rgb2[2], label='Magnetic energy (relative)')
        AX[2].plot(self.time[itk], enKin[itk], c=oC.rgb2[1], label='Kinetik energy (relative)')

        # axx = AX[2].twinx()
        # axx.plot(self.time, -enMag, '--', c=oC.rgb2[2], label='Magnetic energy')
        # AX.plot(time, enTot, '--k', label='total')


        # AX[2].plot(time, E_y_mean, label='Ex')
        # AX[2].plot(time, E_z_mean, label='Ex')

        # AX.plot(timeHR, enMagHR, c='b', label='Magnetic')
        # AX.plot(timeHR, enKinHR, c='g', label='Kinetik')

        # AX.plot(time[(velTot<enLim) * (velTot>-enLim)], velTot[(velTot<enLim) * (velTot>-enLim)], c='#DDDDDD', label='kineticGrid')
        # AX.plot(time[(enTot<enLim) * (enTot>-enLim)], enTot[(enTot<enLim) * (enTot>-enLim)], c='#25417f', label='Total')
        # AX[1].plot(time, nbPart, label='nbPart')
        # for ax in AX:
        #     ax.plot([time[0],time[-1]] ,[0,0], 'k', lw=.5)

        axouille = AX[0].twiny()

        axouille.plot(self.iterations[itk], enElec[itk], alpha=0)

        axouille.set_xlabel('Iterations')
        AX[-1].set_xlabel('Time (s)')
        # AX[0].set_ylabel('Relative difference (%)')
        for i, ax in enumerate(AX):
            ax.legend(loc=2)
            # ax.set_xlim([self.time[0], self.time[-1]])

        # axouille.set_xlim([self.iterations[0], self.iterations[-1]])
        # axx.legend(loc=1)

        mT.set_spines(AX[1:])
        mT.set_spines([AX[0], axouille], two_xaxis=True)
        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/energies.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_energy_tot(self, save_fig=False):

        W_E = np.sum(self.energy_E, axis=0)
        W_B = np.sum(self.energy_B, axis=0)
        W_k = np.sum(self.energy_K, axis=0)

        W_E *= self.mp.v_A**2 * self.mp.B0_SI**2
        W_B *= self.mp.B0_SI**2
        W_k *= self.mp.v_A**2
        #
        # W_E *= eps0
        # W_B *= 1/mu0
        # W_k *= m_i*self.mp.n0_SI/self.mp.nb_part_node_cst
        W_E *= eps0/(2.*self.mp.nb_nodes_cst)
        W_B *= 1./(2.*mu0*self.mp.nb_nodes_cst)
        W_k *= m_i*self.mp.n0_SI/(2*self.mp.nb_nodes_cst*self.mp.nb_part_node_cst)
        # #
        W_tot =  W_k + W_E + W_B
        #
        print((W_tot[-1]-W_tot[0])/(self.time[-1]-self.time[0]))
        print(scipy.stats.linregress(self.time[0:], y=W_tot[0:]))
        # sys.exit()


        fig, ax = plt.subplots(figsize=(20, 16))

        # ax.plot(self.time, (W_tot-W_tot[0])/W_tot[0], c='k', label='W_tot prim')
        ax.plot(self.time, W_tot, c='k', label='W_tot prim')
        ax.plot(self.time, W_k, c=oC.rgb2[0], label='W_k', lw=2)
        ax.plot(self.time, W_E, c=oC.rgb2[1], label='W_E')
        ax.plot(self.time, W_B, c=oC.rgb2[2], label='W_B')

        # ax.plot(self.time, enElec, c=oC.rgb2[1], label='W_E')
        # ax.plot(self.time, enMag, c=oC.rgb2[2], label='W_B')
        # ax.plot(self.time, enKin, c=oC.rgb2[0], label='W_kin')

        ax.legend(fontsize=20)

        mT.set_spines(ax)
        plt.tight_layout()
        plt.show()


        fig, ax = plt.subplots(figsize=(20, 16))

        ax.plot(self.time, W_tot/W_tot[0], c='k', label='W_tot prim')
        # ax.plot(self.time, W_tot, c='k', label='W_tot prim')
        # ax.plot(self.time, W_k, c=oC.rgb2[0], label='W_k', lw=2)
        # ax.plot(self.time, W_E, c=oC.rgb2[1], label='W_E')
        # ax.plot(self.time, W_B, c=oC.rgb2[2], label='W_B')

        # ax.plot(self.time, enElec, c=oC.rgb2[1], label='W_E')
        # ax.plot(self.time, enMag, c=oC.rgb2[2], label='W_B')
        # ax.plot(self.time, enKin, c=oC.rgb2[0], label='W_kin')

        ax.legend(fontsize=20)

        mT.set_spines(ax)
        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/energy_tot.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_moment_1(self, save_fig=False):

        itk = np.ones_like(self.B_mean[0, 0], dtype=bool)#self.B_mean[0, 0]!=0.

        B_mean = np.mean(self.B_mean, axis=0)   ## Mean over ranks.
        # B_mean = mT.norm(B_mean)
        Ji_mean = np.mean(self.Ji_mean, axis=0)   ## Mean over ranks.
        Ji_mean = mT.norm(Ji_mean)
        vel_part_vec = np.mean(self.vel_part, axis=0)   ## Mean over ranks.
        vel_part = mT.norm(vel_part_vec)
        vel_th_part = np.mean(self.vel_th_part, axis=0)   ## Mean over ranks.
        vel_th_part = mT.norm(vel_th_part)
        B_var = np.mean(self.B_var, axis=0)   ## Mean over ranks.
        enMag  = np.mean(self.energy_B, axis=0)

        fig, AX = plt.subplots(4, 1, figsize=(20, 16), sharex=True)

        AX[0].plot(self.time[itk], mT.norm(B_mean)[itk], c='k', label='Mean B')
        AX[0].plot(self.time[itk], B_mean[0, itk], c=oC.rgb2[0], label='Mean Bx')
        AX[0].plot(self.time[itk], B_mean[1, itk], c=oC.rgb2[1], label='Mean By')
        AX[0].plot(self.time[itk], B_mean[2, itk], c=oC.rgb2[2], label='Mean Bz')

        AX[1].plot(self.time[itk], Ji_mean[itk], c='#F442CA', label='Mean Ji')
        # AX[1].plot(d[0, 1:], Ji_mean[1:], 'x', c='#F442CA', label='Mean Ji')
        axxxx = AX[1].twinx()
        AX[1].plot(self.time[itk], vel_part[itk], '--', c='#42F496', label='particles overall mean velocity')

        AX[2].plot(self.time[itk], vel_part_vec[0, itk], c='#F442CA', label='particles v_x mean')
        AX[2].plot(self.time[itk], vel_part_vec[1, itk], c='#42F496', label='particles v_y mean')
        AX[2].plot(self.time[itk], vel_part_vec[2, itk], c='#427AF4', label='particles v_z mean')

        AX[3].plot(self.time[itk], B_var[2, itk], c='#427AF4', label='B_z variance')
        axx = AX[3].twinx()
        axx.plot(self.time[itk], enMag[itk], '--', c='#F442CA', label='Magnetic energy')


        axouille = AX[0].twiny()
        axouille.plot(self.iterations[itk], mT.norm(B_mean)[itk], alpha=0)
        axouille.set_xlabel('Iterations')

        AX[-1].set_xlabel('Time (s)')
        AX[0].set_ylabel('')


        for i, ax in enumerate(AX):
            ax.legend(loc=2)
            ax.set_xlim([self.time[0], self.time[-1]])
        axx.legend(loc=1)
        axxxx.legend(loc=1)

        axouille.set_xlim([self.iterations[0], self.iterations[-1]])

        mT.set_spines(AX[1:])
        mT.set_spines([AX[0], axouille], two_xaxis=True)
        axx.spines['left'].set_visible(False)
        axx.spines['top'].set_visible(False)
        axx.spines['right'].set_position(('outward', 10))
        axx.spines['bottom'].set_position(('outward', 10))
        axxxx.spines['left'].set_visible(False)
        axxxx.spines['top'].set_visible(False)
        axxxx.spines['right'].set_position(('outward', 10))
        axxxx.spines['bottom'].set_position(('outward', 10))

        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/moments_1.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_part_moment(self, save_fig=False):

        itk = self.vel_part[0, 0]!=0

        vel_part_vec = np.mean(self.vel_part, axis=0)   ## Mean over ranks.
        vel_th_part = np.mean(self.vel_th_part, axis=0)   ## Mean over ranks.

        fig, AX = plt.subplots(2, 1, figsize=(20, 16), sharex=True)

        AX[0].plot(self.time[itk], vel_part_vec[0, itk], c='#F442CA', label='particles v_x mean')
        AX[0].plot(self.time[itk], vel_part_vec[1, itk], c='#42F496', label='particles v_y mean')
        AX[0].plot(self.time[itk], vel_part_vec[2, itk], c='#427AF4', label='particles v_z mean')


        axouille = AX[0].twiny()
        axouille.plot(self.iterations[itk], vel_part_vec[0, itk], alpha=0)
        axouille.set_xlabel('Iterations')

        AX[-1].set_xlabel('Time (s)')
        AX[0].set_ylabel('')

        AX[1].plot(self.time[itk], vel_th_part[0, itk], c='#F442CA', label='particles v_th_x mean')
        AX[1].plot(self.time[itk], vel_th_part[1, itk], c='#42F496', label='particles v_th_y mean')
        AX[1].plot(self.time[itk], vel_th_part[2, itk], c='#427AF4', label='particles v_th_z mean')

        for i, ax in enumerate(AX):
            ax.legend(loc=2)
            ax.set_xlim([self.time[0], self.time[-1]])

        axouille.set_xlim([self.iterations[0], self.iterations[-1]])

        mT.set_spines(AX[1])
        mT.set_spines([AX[0], axouille], two_xaxis=True)

        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/moments_particles.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_run_time(self, save_fig=False):

        fig, ax = plt.subplots(figsize=(18, 12))

        ax.plot(self.run_time, self.time, c=oC.rgb2[2])
        ax.set_xlabel('Run time (s)')
        ax.set_ylabel('Simulation time')

        mT.set_spines(ax)

        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/run_time.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()

    def plt_div_B(self, save_fig=False):

        fig, AX = plt.subplots(3, 1, figsize=(18, 12))

        div_B_mean = np.mean(self.div_B_mean, axis=0)
        div_B_var = np.mean(self.div_B_var, axis=0)
        div_B_max = np.amax(self.div_B_max, axis=0)

        AX[0].plot(self.time, div_B_mean, c=oC.rgb2[2], label='div_B mean')
        AX[1].plot(self.time, div_B_var, c=oC.rgb2[2], label='div_B variance')
        AX[2].plot(self.time, div_B_max, c=oC.rgb2[2], label='abs(div_B) max')

        AX[2].set_ylabel('Simulation time')

        for ax in AX:
            ax.legend()

        # AX[1].set_yscale('log')
        # AX[2].set_yscale('log')

        mT.set_spines(AX)

        plt.tight_layout()

        if save_fig:
            fn = f'{self.local_path}/products/plots/run_time.pdf'
            print(f'Saving {fn}')
            plt.savefig(fn)
        else:
            plt.show()






def norm(a):
    return np.sqrt(np.sum(a**2, axis=0))

def centers2edges(centers):
    dx = centers[1]-centers[0]
    edges = np.zeros(centers.size+1)
    edges[:-1] = centers-dx/2.
    edges[-1]  =  centers[-1]+dx/2
    return edges
def endges2centers(edges):
    return .5*(edges[:-1]+edges[1:])



def spher2cart(v_spher):
    """Coordinate system conversion
    """
    v_cart = np.zeros_like(v_spher)
    v_cart[0] = v_spher[0] * np.sin(v_spher[1]) * np.cos(v_spher[2])
    v_cart[1] = v_spher[0] * np.sin(v_spher[1]) * np.sin(v_spher[2])
    v_cart[2] = v_spher[0] * np.cos(v_spher[1])

    return v_cart


def cart2spher(v_cart):
    """Coordinate system conversion
    """
    v_spher = np.zeros_like(v_cart)
    v_spher[0] = np.sqrt(np.sum(v_cart ** 2, axis=0))
    v_spher[1] = np.arccos(v_cart[2] / v_spher[0])
    v_spher[2] = np.arctan2(v_cart[1], v_cart[0])
    itm = (v_spher[2] < 0.)
    v_spher[2][itm] += 2*np.pi

    return v_spher


def cyl2cart(v_cyl):
    """Coordinate system conversion
    """
    v_cart = np.zeros_like(v_cyl)
    v_cart[0] = v_cyl[0]*np.cos(v_cyl[1])
    v_cart[1] = v_cyl[0]*np.sin(v_cyl[1])
    v_cart[2] = v_cyl[2].copy()

    return v_cart


def cart2cyl(v_cart):
    """Coordinate system conversion
    """
    v_cyl = np.zeros_like(v_cart)
    v_cyl[0] = np.sqrt(v_cart[0]**2+v_cart[1]**2)
    v_cyl[1] = np.arctan2(v_cart[1], v_cart[0])
    v_cyl[2] = v_cart[2].copy()
    itm = (v_cyl[1] < 0.)
    v_cyl[1][itm] += 2*np.pi

    return v_cyl

def R_2vect(vector_orig, vector_fin):
    """
    Taken from:
    https://github.com/Wallacoloo/printipi/blob/master/util/rotation_matrix.py
    Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of
    rotation matrices possible.  Due to axially symmetry, the rotation axis
    can be any vector lying in the symmetry plane between the two vectors.
    Hence the axis-angle convention will be used to construct the matrix
    with the rotation axis defined as the cross product of the two vectors.
    The rotation angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |


    Parameters
    ----------
    R
        The 3x3 rotation matrix to update.
    vector_orig
        The unrotated vector defined in the reference frame.
    vector_fin
        The rotated vector defined in the reference frame.
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / np.linalg.norm(vector_orig)
    vector_fin = vector_fin / np.linalg.norm(vector_fin)

    # The rotation axis (normalised).
    axis = np.cross(vector_orig, vector_fin)
    axis_len = np.linalg.norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = math.acos(np.dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = np.cos(angle)
    sa = np.sin(angle)

    # Calculate the rotation matrix elements.
    Rot_mat = np.zeros((3, 3))
    Rot_mat[0, 0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    Rot_mat[0, 1] = -z*sa + (1.0 - ca)*x*y
    Rot_mat[0, 2] = y*sa + (1.0 - ca)*x*z
    Rot_mat[1, 0] = z*sa+(1.0 - ca)*x*y
    Rot_mat[1, 1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    Rot_mat[1, 2] = -x*sa+(1.0 - ca)*y*z
    Rot_mat[2, 0] = -y*sa+(1.0 - ca)*x*z
    Rot_mat[2, 1] = x*sa+(1.0 - ca)*y*z
    Rot_mat[2, 2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)

    return Rot_mat
