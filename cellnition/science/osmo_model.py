#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines classes and methods to model osmotic water flux across a cell membrane, cell volume change
and pressurization, and adaptive control strategies to maintain constant volume against changes to environmental
osmolyte concentrations. The cell is assumed to be a cylindrical shape.

Cellnition consists of:
- an analytical model of osmotic water flow and/or pressure change across a cell membrane or cell wall.
- a depiction of the state space of the physical/physiological process with and without osmoadaptation.
- the introduction of a control strategy to maintain cell volume against an external change in osmomolarity:
    - a classic PID (proportional-integral-derivative control strategy)
    - a simple biological case
    - a cascaded biological case
    - a controller designed from inspection of the state space topography
- finally, we explore the possibility of state space estimation, both for the beneficial case of developing a
state space model from biological data, but also with the idea that living organisms, via some kind of neural
network or embodied analogue gaussian process, may be able to construct their own state space estimates
in order to generate more effective (i.e. intelligent) responses.
'''

import numpy as np
from numpy import ndarray
from cellnition.science.model_params import ModelParams
from cellnition.science.analytic_model import AnalyticOsmoticModel


class OsmoticCell(object):
    '''

    '''

    def __init__(self):
        '''
        Initialize the model by creating the sympy analytical system.

        '''

        self.ana = AnalyticOsmoticModel() # This contains lambdifyied functions for numerical computing

    def osmo_p(self, m_o_f, m_i_f, p: ModelParams):
        '''
        Calculate the osmotic pressure difference across the membrane.
        '''
        return p.R * p.T * (m_i_f - m_o_f)


    def ind_p(self, vol_cell_f: ndarray|float, Y: float, d: float, p: ModelParams):
        '''
        Compute the induced pressure (i.e. Turgor pressure) induced by structural pushback against
        transmembrane osmotic water influx.
        '''

        return self.ana.P_ind_f(vol_cell_f, p.vol_cell_o, p.r_cell_o, Y, d)

    def osmo_vol_steady_state(self,
                              m_o_f: ndarray|float,
                              n_i_f: ndarray|float,
                              Y: float,
                              d: float,
                              p: ModelParams):
        '''
        Calculate the steady-state volume of a cell with fixed internal and external ion concentrations.
        This function assumes that a cell that shrinks from a state where it is not under mechanical stress does
        so freely without constraint from its mechanical properties. However, when a cell expands, it is assumed to
        encounter resistance from the elastic nature of the membrane, which reduces the amount of water influx due to
        the development of Turgor pressure.

        '''


        if type(m_o_f) is ndarray and type(n_i_f) is ndarray:
            Vss_vect_R1 = self.ana.Vss_R1_f(m_o_f, n_i_f)
            Vss_vect_R2 = self.ana.Vss_R2_f(m_o_f, n_i_f, p.R, p.T, d, Y, p.r_cell_o, p.vol_cell_o)

            ss_inds_R1 = (Vss_vect_R1 < p.vol_cell_o).nonzero()[0]
            ss_inds_R2 = (Vss_vect_R2 >= p.vol_cell_o).nonzero()[0]

            Vss_vect = np.zeros(m_o_f.ravel().shape)
            Vss_vect[ss_inds_R1] = Vss_vect_R1[ss_inds_R1]
            Vss_vect[ss_inds_R2] = Vss_vect_R2[ss_inds_R2]

            v_ss = Vss_vect

        else:
            Vss_R1 = self.ana.Vss_R1_f(m_o_f, n_i_f)
            Vss_R2 = self.ana.Vss_R2_f(m_o_f, n_i_f, p.R, p.T, d, Y, p.r_cell_o, p.vol_cell_o)

            if Vss_R1 < p.vol_cell_o:
                v_ss = Vss_R1

            else:
                v_ss = Vss_R2

        return v_ss

    # Next, write a function that will take numerical parameters and provide an updated cell volume using Euler's method
    def osmo_vol_update(self, vol_o_f, del_t_f, A_chan_f, N_chan_f, n_i_f, m_o_f, d_mem_f, Y_mem, p: ModelParams):
        '''
        Volume update for time-dependent transmembrane osmotic water flux for the case of low cell-wall regidity.
        '''

        # Calculate the volume change for this situation:
        dV_dt = self.osmo_vol_velocity(vol_o_f, A_chan_f, N_chan_f, n_i_f, m_o_f, d_mem_f, Y_mem, p)

        return vol_o_f + del_t_f*dV_dt

    def osmo_vol_velocity(self,
                          vol_o_f: ndarray|float,
                          n_i_f: ndarray|float,
                          m_o_f: ndarray|float,
                          d_mem_f: float,
                          Y_mem_f: float,
                          p: ModelParams):
        '''
        Returns the rate of volume change, which is equal to the transmembrane osmotic water volumetric flux.
        '''

        if type(vol_o_f) is ndarray and type(n_i_f) is ndarray and type(m_o_f) is ndarray:
            # Get indices for places in arrays where the volume specifies regime 1 or 2:
            i_VV_R1 = (vol_o_f < p.vol_cell_o).nonzero()[0]  # Regime 1, topological changes
            i_VV_R2 = (vol_o_f >= p.vol_cell_o).nonzero()[0]  # Regime 2, elastic stretch

            # Define a new array that will store the dV_dt data:
            dVdt_vect = np.zeros(vol_o_f.shape)

            dVdt_vect[i_VV_R1] = self.ana.dVdt_R1_f(vol_o_f[i_VV_R1], m_o_f[i_VV_R1],
                                                    n_i_f[i_VV_R1], p.A_chan_o,
                                           p.N_chan_o, p.R, p.T, d_mem_f, p.mu)

            dVdt_vect[i_VV_R2] = self.ana.dVdt_R2_f(vol_o_f[i_VV_R2], m_o_f[i_VV_R2],
                                                    n_i_f[i_VV_R2], p.A_chan_o,
                                                    p.N_chan_o, p.R, p.T, d_mem_f, p.mu,
                                                    Y_mem_f, p.r_cell_o, p.vol_cell_o)

            dVol_dt = dVdt_vect

        else:
            if vol_o_f < p.vol_cell_o:
                dVol_dt = self.ana.dVdt_R1_f(vol_o_f, m_o_f, n_i_f, p.A_chan_o,
                                           p.N_chan_o, p.R, p.T, d_mem_f, p.mu)

            else:
                dVol_dt = self.ana.dVdt_R2_f(vol_o_f, m_o_f, n_i_f, p.A_chan_o,
                                                    p.N_chan_o, p.R, p.T, d_mem_f, p.mu,
                                                    Y_mem_f, p.r_cell_o, p.vol_cell_o)

        return dVol_dt

    def glycerol_velocity(self, vol_i: ndarray|float, m_i_gly, p: ModelParams):
        '''
        Computes the change in intracellular glycerol concentration as a function of time, based on
        cell volume.

        '''

        dm_gly_dt = self.ana.dmi_dt_f(vol_i,
                                           p.K_1,
                                           p.K_2,
                                           p.b_1,
                                           p.b_2,
                                           p.d_gly_max,
                                           p.epsilon_o_1,
                                           p.epsilon_o_2,
                                           m_i_gly,
                                           p.r_gly_max,
                                           p.vol_cell_o)

        return dm_gly_dt

    def osmo_time_sim(self,
                      t_vect_f: ndarray,
                      mo_vect_f: ndarray|float,
                      cell_vol_o_f,
                      n_i_base,
                      d_wall_f,
                      Y_wall_f,
                      del_t_f,
                      samp_i_f: int,
                      p: ModelParams,
                      synth_gly: bool=True
                           ):
        '''
        A dynamic simulation of a single cell's volume changes given a time series vector representing external
        osmolyte concentrations (mo_vect), for the case of a biologically-relevant control strategy.
        This assumes a cylindrically-shaped cell.
        This osmotic flux model assumes that for the case of a plant cell, water can leave the cell freely in the case
        of a hypoosmotic environment, yet the cell wall pressurizes the cell so that water entry and volume change
        with hyperosmotic environment is more limited. For a cell without a wall, volume change is directly related
        to transmembrane water flux and structural pressure is assumed to be negligible.

        In this model the cell has a control strategy based on:
        sensing circumferential strain loss leads to closure of Fsp1 glycerol/aquaporin receptors
        sensing circumferential strain loss activates the SLN1 receptors
        When strain is lost, phosphorylation of SLN1 is lost and the HOG-MAPK signalling pathway is activated.
        HOG-MAPK increases the rate of glycerol synthesis and decreases the rate of glycerol efflux.
        Increased intracellular glycerol leads to influx of water and restoration of cell volume and strain.

        '''
        t_vect_i_f = []  # sampled time vector points
        mo_vect_i_f = []  # sampled env osmolyte concentration vector points
        Po_vect_f = []  # osmotic pressure as a function of time
        eh_vect_f = []  # circumferential strain as a function of time
        r_vect_f = []  # radius of the cell with time
        vol_vect_f = []  # cell volume as a function of time
        dvol_vect_f = []  # cell volume change as a function of time
        gly_vect_f = [] # intracellular glycerol concentration as a function of time
        mi_vect_f = [] # intracellular concentrations of osmolytes

        t_samps_f = t_vect_f[0::samp_i_f]

        cell_vol_i = cell_vol_o_f * 1  # initialize the working cell volume
        m_i_gly = p.m_i_gly # initialize intracellular glycerol concentration
        n_i = n_i_base + m_i_gly*cell_vol_i # initialize osmolyte moles in the cell
        m_i = n_i/cell_vol_i # initialize the osmolyte concentration in the cell

        for ii, ti in enumerate(t_vect_f):

            if type(mo_vect_f) is ndarray:
                m_o = mo_vect_f[ii]

            else:
                m_o = mo_vect_f

            # Calculate osmotic pressure:
            Po_f = self.osmo_p(m_o, m_i, p)

            # Calculate an osmotic volumetric flow rate:
            dV_dt = self.osmo_vol_velocity(cell_vol_i, n_i, m_o, d_wall_f, Y_wall_f, p)

            # update cell_vol_o:
            cell_vol_i += dV_dt * del_t_f

            eh = (cell_vol_i / p.vol_cell_o) - 1 # Calculate the hoop strain

            # Control module-----------------------------------------------------------------------------------------
            # Cell sensing of strain due to volume change and response by changing glycerol production and efflux.
            # phosphorylation level of the Sln1 receptor:

            # synthesis of glycerol:
            dm_gly_dt = self.glycerol_velocity(cell_vol_i, m_i_gly, p)
            # update the glycerol concentration:
            m_i_gly += del_t_f * dm_gly_dt
            # convert to moles of glycerol:
            n_i_gly = m_i_gly * cell_vol_i

            if synth_gly: # If glycerol is having an effect on the cell osmolytes and channel area
                n_i = n_i_base + n_i_gly # update total moles of osmoyltes in the cell

            # Update the concentration of osmolytes in the cell (which change with water flux and volume changes):
            m_i = n_i / cell_vol_i

            # Update the cell radius (in our model length doesn't change since eL = 0 with nu = 0.5):
            r_cell_f = (eh + 1) * p.r_cell_o

            if ti in t_samps_f:  # Then sample and record values
                t_vect_i_f.append(ti * 1)
                mo_vect_i_f.append(m_o * 1)
                Po_vect_f.append(Po_f * 1)
                eh_vect_f.append(eh * 1)
                r_vect_f.append(r_cell_f * 1)
                vol_vect_f.append(cell_vol_i * 1)
                dvol_vect_f.append(dV_dt * 1)
                gly_vect_f.append(m_i_gly*1)
                mi_vect_f.append(m_i*1)

        self.osmo_data_bio1 = np.column_stack(
            (t_vect_i_f, mo_vect_i_f,  mi_vect_f, eh_vect_f, r_vect_f, vol_vect_f, dvol_vect_f,
             gly_vect_f))

        return self.osmo_data_bio1

    def state_space_gen(self,
                        mo_vect_f: ndarray,
                        vol_vect_f: ndarray,
                        ni_vect_f: ndarray,
                        mi_gly: float,
                        d_wall_f: float,
                        Y_wall_f: float,
                        p: ModelParams,
                        synth_gly: bool=True
                        ):
        '''
        A dynamic simulation of a single cell's volume changes given a time series vector representing external
        osmolyte concentrations (mo_vect), for the case of a biologically-relevant control strategy.
        This assumes a cylindrically-shaped cell. The model further assumes Poisson's ratio for the material is nu=0.5,
        in order to create solvable analytic equations as axial strain goes to zero in that case.
        This osmotic flux model assumes that for the case of a plant cell, water can leave the cell freely in the case
        of a hypoosmotic environment, yet the cell wall pressurizes the cell so that water entry and volume change
        with hyperosmotic environment is more limited. For a cell without a wall, volume change is directly related
        to transmembrane water flux and structural pressure is assumed to be negligible.

        In this model the cell has a control strategy based on:
        sensing circumferential strain loss leads to closure of Fsp1 glycerol/aquaporin receptors
        sensing circumferential strain loss activates the SLN1 receptors
        When strain is lost, phosphorylation of SLN1 is lost and the HOG-MAPK signalling pathway is activated.
        HOG-MAPK increases the rate of glycerol synthesis and decreases the rate of glycerol efflux.
        Increased intracellular glycerol leads to influx of water and restoration of cell volume and strain.

        '''

        dvol_vect_f = []  # Instantaneous rate of cell volume change
        dni_vect_f = [] # Instantaneous rate of change of intracellular molarity

        Po_vect_f = [] # osmotic pressure

        m_i = ni_vect_f/vol_vect_f
        # Calculate osmotic pressure:
        Po_f = self.osmo_p(mo_vect_f, m_i, p)

        # Calculate an osmotic volumetric flow rate:
        dV_dt = self.osmo_vol_velocity(vol_vect_f, ni_vect_f, mo_vect_f, d_wall_f, Y_wall_f, p)

        # Control module-----------------------------------------------------------------------------------------
        # Cell sensing of strain due to volume change and response by changing glycerol production and efflux.
        # phosphorylation level of the Sln1 receptor:

        # synthesis of glycerol; update the glycerol concentration:
        dm_gly_dt = self.glycerol_velocity(vol_vect_f, mi_gly, p)

        if synth_gly: # If glycerol is having an effect on the cell osmolytes
            # convert to moles of glycerol, since base molarity is assumed constant the rate of change of
            # intracellular moles is equal to the rate of change of moles intracellular glycerol
            dn_dt = dm_gly_dt * vol_vect_f

        else:
            dn_dt = 0.0 # otherwise, if no adaptive response, no change to intracellular molarity

        # Store calculated values:
        Po_vect_f.append(Po_f * 1)
        dvol_vect_f.append(dV_dt * 1)
        dni_vect_f.append(dn_dt*1)

        # Pack data:
        # self.state_space_data_bio1 = np.column_stack(
        #     (dvol_vect_f, dni_vect_f, Po_vect_f))
        dvol_vect_f = np.asarray(dvol_vect_f)
        dni_vect_f = np.asarray(dni_vect_f)
        Po_vect_f = np.asarray(Po_vect_f)

        return dvol_vect_f, dni_vect_f, Po_vect_f
