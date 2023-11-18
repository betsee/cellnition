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

# TODO: save analytical model equations to the class
# TODO: print out analytical model equations as LaTeX

from beartype import beartype
import numpy as np
from numpy import ndarray
import sympy as sp
from cellnition.science.model_params import ModelParams


class OsmoticCell(object):
    '''

    '''

    def __init__(self):
        '''

        '''

        self.analytical_model() # Create the analytical model and any required numerical equations



    def analytical_model(self):
        '''
        This method uses Sympy to construct the analytical equations defining the osmotic cell model.
        These can be printed out in
        '''

        # Key Variables
        # --------------
        # Note the '_s' subscript indicates it is a symbolic, Sympy variable
        print("Generating the analytical model.")

        # Thermodynamic constants and parameters:
        R_s, T_s, F_s, t_s = sp.symbols('R, T, F, t_s', real=True, positive=True)

        # Dimensional parameters:
        r_cell_o_s, vol_cell_o_s, d_mem_s, L_cell_o_s, A_mem_o_s = sp.symbols(
            'r_cell_o, vol_cell_o, d_mem, L_cell_o, A_mem_o', real=True, positive=True)
        r_cell_s, L_cell_s, A_mem_s = sp.symbols('r_cell, L_cell, A_mem', real=True, positive=True)

        # Mechanical parameters:
        P_cell_s, sigma_H_s, sigma_L_s, epsilon_H_s, epsilon_L_s, Y_s, nu_s = sp.symbols(
            'P_cell, sigma_H, sigma_L, epsilon_H, epsilon_L, Y, nu', real=True)
        d_H_s, d_L_s = sp.symbols('d_H_s, d_L_s', real=True)

        # Osmotic and flow parameters:
        m_i_s, m_o_s, n_i_s, mu_s = sp.symbols('m_i, m_o, n_i_s, mu_s', real=True, positive=True)
        Pi_i_s, Pi_o_s, Pi_io_s, u_io_s, Q_io_s = sp.symbols('Pi_i, Pi_o, Pi_io, u_io, Q_io', real=True)

        # Physiological parameters:
        A_chan_s, N_chan_s = sp.symbols('A_chan, N_chan', real=True, positive=True)

        # vol_cell_s = sp.symbols('vol_cell', cls=sp.Function)
        vol_cell_s = sp.Function('vol_cell')(t_s)

        # Key Equations:
        # Situation 1: Volume change with transmembrane osmotic water flux

        # Osmotic pressure difference across membrane:
        Eq1_Pi_io_s = sp.Eq(Pi_io_s, (m_i_s - m_o_s) * R_s * T_s)

        # Osmotic water flux across membrane via water channels:
        Eq2_u_io_s = sp.Eq(u_io_s, (Pi_io_s * A_chan_s * N_chan_s) / (8 * mu_s * d_mem_s))

        # Substitute in Eq1 to Eq2:
        Eq3_u_io_s = sp.Eq(u_io_s, ((m_i_s - m_o_s) * R_s * T_s * A_chan_s * N_chan_s) / (8 * mu_s * d_mem_s))

        # Expression for volumetric flow rate:
        Eq4_Q_io_s = sp.Eq(sp.diff(vol_cell_s, t_s), u_io_s * A_chan_s * N_chan_s)

        # Substitute in Eq3 for u_io to Eq4 to obtain volumetric flow rate in terms of core parameters:
        Eq5_Q_io_s = sp.Eq(sp.diff(vol_cell_s, t_s),
                           ((m_i_s - m_o_s) * R_s * T_s * A_chan_s ** 2 * N_chan_s ** 2) / (8 * mu_s * d_mem_s))

        # Now we realize that while the concentration of osmolytes in the environment remains independent of osmotic water fluxes, that that in the cell changes:
        Eq6_mi_s = sp.Eq(m_i_s, n_i_s / vol_cell_s)

        # Finally, we substitute in Eq6 for m_i_s to Eq5 to obtain the master differential equation for the osmotic cell volume change expression:
        Eq7_Q_io_s = sp.Eq(sp.diff(vol_cell_s, t_s),
                           ((n_i_s / vol_cell_s - m_o_s) * R_s * T_s * A_chan_s ** 2 * N_chan_s ** 2) / (
                                       8 * mu_s * d_mem_s)).simplify()

        Eq8_vol_ss_s = sp.Eq(vol_cell_s, (m_i_s * vol_cell_o_s) / m_o_s)

        # Given strain, solve for the stress:
        EqA_epsilon_H = sp.Eq(epsilon_H_s, (1 / Y_s) * (sigma_H_s - nu_s * sigma_L_s))
        EqB_epsilon_L = sp.Eq(epsilon_L_s, (1 / Y_s) * (sigma_L_s - nu_s * sigma_H_s))

        solAB = sp.solve((EqA_epsilon_H, EqB_epsilon_L), (sigma_H_s, sigma_L_s))
        solAB[sigma_L_s].simplify()

        # Now taking a look at the infintessimal volume change and turgor pressure approach:

        # Hoop stress equations in terms of osmotic pressure:

        Eq9_sigma_H_s = sp.Eq(sigma_H_s, (Pi_io_s * r_cell_o_s) / d_mem_s)
        Eq10_sigma_L_s = sp.Eq(sigma_L_s, (Pi_io_s * r_cell_o_s) / (2 * d_mem_s))

        Eq11_sigma_H_s = sp.Eq(sigma_H_s, (((m_i_s - m_o_s) * R_s * T_s * r_cell_o_s) / d_mem_s))
        Eq12_sigma_L_s = sp.Eq(sigma_L_s, (((m_i_s - m_o_s) * R_s * T_s * r_cell_o_s) / (2 * d_mem_s)))

        # Hoop strain equation:
        Eq13_epsilon_H_s = sp.Eq(epsilon_H_s, ((Pi_io_s * r_cell_o_s) / (d_mem_s * Y_s)) * (1 - (nu_s / 2)))
        Eq14_epsilon_L_s = sp.Eq(epsilon_L_s, ((Pi_io_s * r_cell_o_s) / (d_mem_s * Y_s)) * ((1 / 2) - nu_s))

        Eq15_epsilon_H_s = sp.Eq(epsilon_H_s,
                                 (((m_i_s - m_o_s) * R_s * T_s * r_cell_o_s) / (d_mem_s * Y_s)) * (1 - (nu_s / 2)))
        Eq16_epsilon_L_s = sp.Eq(epsilon_L_s,
                                 (((m_i_s - m_o_s) * R_s * T_s * r_cell_o_s) / (d_mem_s * Y_s)) * ((1 / 2) - nu_s))

        # Displacements in the circumferential ('H') and axis ('L') directions:
        Eq17_d_H_s = sp.Eq(d_H_s,
                           (((m_i_s - m_o_s) * R_s * T_s * r_cell_o_s ** 2) / (d_mem_s * Y_s)) * (1 - (nu_s / 2)))
        Eq18_d_L_s = sp.Eq(d_L_s, (((m_i_s - m_o_s) * R_s * T_s * r_cell_o_s * L_cell_o_s) / (d_mem_s * Y_s)) * (
                    (1 / 2) - nu_s))

        # Finally, if we have a volume change what is the corresponding hoop strain and stress that is generated?

        Eq19_vol_strain_s = sp.Eq(vol_cell_s,
                                  2 * sp.pi * r_cell_o_s * L_cell_o_s * (1 + epsilon_H_s) * (1 + epsilon_L_s))



        Eq20_vol_strain_s = sp.Eq(vol_cell_s,
                                  2 * sp.pi * r_cell_o_s * L_cell_o_s * (
                                              1 + ((Pi_io_s * r_cell_o_s) / (d_mem_s * Y_s)) * (1 -
                                                                                                (nu_s / 2))) * (
                                              1 + ((Pi_io_s * r_cell_o_s) / (d_mem_s * Y_s)) * ((1 / 2) - nu_s)))

        sol_P = sp.solve(Eq20_vol_strain_s, Pi_io_s)  # Solve the vol_strain equation for the pressure

        self.pressure_from_volume = sp.lambdify([vol_cell_s, vol_cell_o_s, r_cell_o_s, L_cell_o_s, Y_s, nu_s, d_mem_s],
                                                sol_P[1])  # Correct solution for pressure from vol change

    def osmo_p(self, m_o_f, m_i_f, p: ModelParams):
        '''
        Calculate the osmotic pressure difference across the membrane.
        '''
        return p.R * p.T * (m_i_f - m_o_f)

    # Function that calculates the steady-state volume given initial concentration differences, indifferent to mechanical pressure:
    def osmo_vol_ss(self, m_o_f, m_i_f, p: ModelParams):
        '''
        Calculate the steady-state volume of a cell with fixed internal and external ion concentrations.
        '''
        return (m_i_f * p.cell_vol_o) / m_o_f

    def stress_from_strain(self, Y_f, eh_f, el_f, nu_f):
        '''
        Calculate hoop (H) and axial (L) stresses given corresponding strains.
        '''
        sh = -(Y_f * (eh_f + el_f * nu_f)) / (nu_f ** 2 - 1)
        sl = -(Y_f * (el_f + eh_f * nu_f)) / (nu_f ** 2 - 1)
        return sh, sl

    # Next, write a function that will take numerical parameters and provide an updated cell volume using Euler's method
    def osmo_vol_update(self, vol_o_f, del_t_f, A_chan_f, N_chan_f, n_i_f, m_o_f, d_mem_f, p: ModelParams):
        '''
        Volume update for time-dependent transmembrane osmotic water flux for the case of low cell-wall regidity.
        '''

        return vol_o_f + (del_t_f * (A_chan_f ** 2) * (N_chan_f ** 2) * p.R * p.T * (n_i_f - m_o_f * vol_o_f)) / (
                    8 * d_mem_f * p.mu * vol_o_f)

    def osmo_vol_change(self, vol_o_f, A_chan_f, N_chan_f, n_i_f, m_o_f, d_mem_f, p: ModelParams):
        '''
        Returns the osmotic water volumetric flux.
        '''
        beta = (p.R * p.T) / (8 * d_mem_f * p.mu)

        dVol_dt = (beta * (A_chan_f ** 2) * (N_chan_f ** 2) * (n_i_f - m_o_f * vol_o_f)) / vol_o_f

        return dVol_dt

    def strain_from_vol_change(self, vol_f, vol_o_f, nu_f, r_o_f, L_o_f, d_mem_f, Y_f):

        P = self.pressure_from_volume(vol_f, vol_o_f, r_o_f, L_o_f, Y_f, nu_f, d_mem_f)  # Pressure from volume change

        eps_H = ((P * r_o_f) / (d_mem_f * Y_f)) * (1 - (nu_f / 2))
        eps_L = ((P * r_o_f) / (d_mem_f * Y_f)) * ((1 / 2) - nu_f)

        return eps_H, eps_L

    def stress_strain_from_osmo_p(self, Pi_f, r_cell_o_f, l_cell_o_f, d_mem_f, Y_f, nu_f):
        '''
        Computes hoop stress and hoop strain given the osmotic pressure and assuming a stiff cell wall.
        Returns the strains and the new cell volume.
        '''

        sh = (Pi_f * r_cell_o_f) / d_mem_f
        sl = sh / 2
        eh = (1 / Y_f) * (sh - nu_f * sl)
        el = (1 / Y_f) * (sl - nu_f * sh)

        vol = 2 * np.pi * r_cell_o_f * l_cell_o_f * (1 + eh) * (1 + el)

        return eh, el, vol

    def osmo_time_sim_free(self,
                           t_vect_f: ndarray,
                           mo_vect_f: ndarray,
                           m_i_o_f,
                           cell_vol_o_f,
                           A_chan_o_f,
                           N_chan_o_f,
                           d_wall_f,
                           r_cell_o_f,
                           L_cell_o_f,
                           Y_wall_f,
                           nu_wall_f,
                           del_t_f,
                           samp_i_f: int,
                           p: ModelParams
                           ):
        '''
        A dynamic simulation of a single cell's volume changes given a time series vector representing external
        osmolyte concentrations (mo_vect), for the case of no control strategy.
        This assumes a cylindrically-shaped cell.
        This osmotic flux model assumes that for the case of a plant cell, water can leave the cell freely in the case
        of a hypoosmotic environment, yet the cell wall pressurizes the cell so that water entry and volume change
        with hyperosmotic environment is more limited. For an cell without a wall, volume change is directly related
        to transmembrane water flux and structural pressure is assumed to be negligable.

        In this model the cell has no control strategy, so doesn't have any means to change its volume in the face
        of an external osmolyte concentration challenge.

        '''
        t_vect_i_f = []  # sampled time vector points
        mo_vect_i_f = []  # sampled env osmolyte concentration vector points
        Po_vect_f = []  # osmotic pressure as a function of time
        eh_vect_f = []  # circumferential strain as a function of time
        el_vect_f = []  # axis strain as a function of time
        r_vect_f = []  # radius of the cell with time
        l_vect_f = []  # length of the cell with time
        vol_vect_f = []  # cell volume as a function of time
        dvol_vect_f = []  # cell volume change as a function of time

        t_samps_f = t_vect_f[0::samp_i_f]

        cell_vol_i = cell_vol_o_f * 1  # initialize the working cell volume

        n_i_o_f = m_i_o_f * cell_vol_o_f  # osmolytes in the cell

        for ii, m_o in enumerate(mo_vect_f):

            ti = t_vect_f[ii]

            # Calculate osmotic pressure:
            Po_f = self.osmo_p(m_o, m_i_o_f, p)

            if p.plant_model is True:

                # Calculate an osmotic volumetric flow rate:
                dV_dt_f = self.osmo_vol_change(cell_vol_i, A_chan_o_f, N_chan_o_f, n_i_o_f, m_o, d_wall_f, p)

                # If the water is entering the cell and the cell is fully inflated, use the volume change from pressure-limited scenario:
                if dV_dt_f >= 0.0 and cell_vol_i >= p.cell_vol_o:
                    eh, el, vol2 = self.stress_strain_from_osmo_p(Po_f, r_cell_o_f, L_cell_o_f, d_wall_f, Y_wall_f, nu_wall_f)

                else:  # otherwise, if negative, then update using the timestep:
                    vol2 = self.osmo_vol_update(cell_vol_i, del_t_f, A_chan_o_f, N_chan_o_f, n_i_o_f, m_o, d_wall_f, p)
                    eh, el = self.strain_from_vol_change(vol2, p.cell_vol_o, nu_wall_f, r_cell_o_f, L_cell_o_f, d_wall_f, Y_wall_f)

            else:
                # Calculate an osmotic volumetric flow rate:
                dV_dt_f = self.osmo_vol_change(cell_vol_i, A_chan_o_f, N_chan_o_f, n_i_o_f, m_o, d_wall_f, p)

                vol2 = self.osmo_vol_update(cell_vol_i, del_t_f, A_chan_o_f, N_chan_o_f, n_i_o_f, m_o, d_wall_f, p)
                eh, el = self.strain_from_vol_change(vol2, p.cell_vol_o, nu_wall_f, r_cell_o_f, L_cell_o_f, d_wall_f,
                                                     Y_wall_f)

            # update cell_vol_o:
            cell_vol_i = vol2 * 1

            # Update the concentration of osmolytes in the cell (which change with water flux and volume changes):
            m_i_o_f = n_i_o_f / cell_vol_i

            # Update the cell radius and length:
            r_cell_f = (eh + 1) * r_cell_o_f
            l_cell_f = (el + 1) * L_cell_o_f

            if ti in t_samps_f:  # Then sample and record values
                t_vect_i_f.append(ti * 1)
                mo_vect_i_f.append(m_o * 1)
                Po_vect_f.append(Po_f * 1)
                eh_vect_f.append(eh * 1)
                el_vect_f.append(el * 1)
                r_vect_f.append(r_cell_f * 1)
                l_vect_f.append(l_cell_f * 1)
                vol_vect_f.append(vol2 * 1)
                dvol_vect_f.append(dV_dt_f * 1)

        self.osmo_data_free = np.column_stack(
            (t_vect_i_f, mo_vect_i_f, Po_vect_f, eh_vect_f, el_vect_f, r_vect_f, l_vect_f, vol_vect_f, dvol_vect_f))

        return self.osmo_data_free


    def osmo_time_sim_bio1(self,
                           t_vect_f: ndarray,
                           mo_vect_f: ndarray,
                           cell_vol_o_f,
                           A_chan_o_f,
                           N_chan_o_f,
                           d_wall_f,
                           r_cell_o_f,
                           L_cell_o_f,
                           Y_wall_f,
                           nu_wall_f,
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
        el_vect_f = []  # axis strain as a function of time
        r_vect_f = []  # radius of the cell with time
        l_vect_f = []  # length of the cell with time
        vol_vect_f = []  # cell volume as a function of time
        dvol_vect_f = []  # cell volume change as a function of time
        gly_vect_f = [] # intracellular glycerol concentration as a function of time
        mi_vect_f = [] # intracellular concentrations of osmolytes

        t_samps_f = t_vect_f[0::samp_i_f]

        cell_vol_i = cell_vol_o_f * 1  # initialize the working cell volume

        m_i = p.m_i_o # initialize the osmolyte concentration in the cell
        m_i_gly = p.m_i_gly # initialize intracellular glycerol concentration

        n_i = p.n_i_o # initialize osmolyte moles in the cell
        # n_i_gly = p.n_i_gly # initialize the moles of glycerol in the cell

        for ii, m_o in enumerate(mo_vect_f):

            ti = t_vect_f[ii]

            # Calculate osmotic pressure:
            Po_f = self.osmo_p(m_o, m_i, p)

            if p.plant_model is True:

                # Calculate an osmotic volumetric flow rate:
                dV_dt_f = self.osmo_vol_change(cell_vol_i, A_chan_o_f, N_chan_o_f, n_i, m_o, d_wall_f, p)

                # If the water is entering the cell and the cell is fully inflated, use the volume change from pressure-limited scenario:
                if dV_dt_f >= 0.0 and cell_vol_i >= p.cell_vol_o:
                    eh, el, vol2 = self.stress_strain_from_osmo_p(Po_f, r_cell_o_f, L_cell_o_f, d_wall_f, Y_wall_f, nu_wall_f)

                else:  # otherwise, if negative, then update using the timestep:
                    vol2 = self.osmo_vol_update(cell_vol_i, del_t_f, A_chan_o_f, N_chan_o_f, n_i, m_o, d_wall_f, p)
                    eh, el = self.strain_from_vol_change(vol2, p.cell_vol_o, nu_wall_f, r_cell_o_f, L_cell_o_f, d_wall_f, Y_wall_f)

            else:
                # Calculate an osmotic volumetric flow rate:
                dV_dt_f = self.osmo_vol_change(cell_vol_i, A_chan_o_f, N_chan_o_f, n_i, m_o, d_wall_f, p)

                vol2 = self.osmo_vol_update(cell_vol_i, del_t_f, A_chan_o_f, N_chan_o_f, n_i, m_o, d_wall_f, p)
                eh, el = self.strain_from_vol_change(vol2, p.cell_vol_o, nu_wall_f, r_cell_o_f, L_cell_o_f, d_wall_f,
                                                     Y_wall_f)

            # update cell_vol_o:
            cell_vol_i = vol2 * 1

            # Control module-----------------------------------------------------------------------------------------
            # Cell sensing of strain due to volume change and response by changing glycerol production and efflux.
            # phosphorylation level of the Sln1 receptor:
            sln1_resp = 1 / (1 + np.exp(-p.K_sln1 * (eh - p.eo_sln1)))
            # When phosphorylated, sln1 will inhibit glycerol production and activate glycerol export:
            act_sln1 = ((sln1_resp / p.ka_sln1) ** p.na_sln1) / (1 + (sln1_resp / p.ka_sln1) ** p.na_sln1)
            inh_sln1 = 1 / (1 + (sln1_resp / p.ki_sln1) ** p.ni_sln1)

            # synthesis of glycerol; update the glycerol concentration:
            m_i_gly = del_t_f * (inh_sln1 * p.growth_gly_max - act_sln1 * p.decay_gly_max * m_i_gly) + m_i_gly
            n_i_gly = m_i_gly * cell_vol_i  # convert to moles of glycerol

            if synth_gly: # If glycerol is having an effect on the cell osmolytes
                n_i = p.n_i_base + n_i_gly # update total moles of osmoyltes in the cell

            # Update the concentration of osmolytes in the cell (which change with water flux and volume changes):
            m_i = n_i / cell_vol_i

            # Update the cell radius and length:
            r_cell_f = (eh + 1) * r_cell_o_f
            l_cell_f = (el + 1) * L_cell_o_f

            if ti in t_samps_f:  # Then sample and record values
                t_vect_i_f.append(ti * 1)
                mo_vect_i_f.append(m_o * 1)
                Po_vect_f.append(Po_f * 1)
                eh_vect_f.append(eh * 1)
                el_vect_f.append(el * 1)
                r_vect_f.append(r_cell_f * 1)
                l_vect_f.append(l_cell_f * 1)
                vol_vect_f.append(vol2 * 1)
                dvol_vect_f.append(dV_dt_f * 1)
                gly_vect_f.append(m_i_gly*1)
                mi_vect_f.append(m_i*1)

        self.osmo_data_bio1 = np.column_stack(
            (t_vect_i_f, mo_vect_i_f, Po_vect_f, eh_vect_f, el_vect_f, r_vect_f, l_vect_f, vol_vect_f, dvol_vect_f,
             gly_vect_f, mi_vect_f))

        return self.osmo_data_bio1
