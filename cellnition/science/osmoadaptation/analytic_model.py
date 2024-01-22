#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module defines a set of complete analytical equations, written in Sympy, to model osmotic water flux across a cell
membrane, with corresponding cell volume change and pressurization (i.e. development of Turgor pressure),
and adaptive control strategies to maintain constant volume against changes to environmental
osmolyte concentrations. The cell is assumed to be a cylindrical shape and to have a Poisson Ratio of nu = 0.5.

'''

# TODO: print out analytical model equations as LaTeX

import sympy as sp

class AnalyticOsmoticModel(object):
    '''

    '''
    def __init__(self):
        '''
        Initialize the model system.
        '''

        self._write_expressions()

    def _write_expressions(self):
        '''
        Note the '_s' subscript indicates it is a symbolic, Sympy variable or expression, whereas the '_f'
        subscript indicates it is a lambda function suitable for numerical computation.

        '''

        # Key Variables

        # Thermodynamic constants and parameters:
        R_s, T_s, F_s, mu_s, t_s = sp.symbols('R, T, F, mu, t', real=True, positive=True)

        # Dimensional parameter constants:
        r_cell_o_s, vol_cell_o_s, d_mem_s, L_cell_o_s, A_mem_o_s = sp.symbols(
            'r_cell_o, vol_cell_o, d_mem, L_cell_o, A_mem_o', real=True, positive=True)
        r_cell_s, vol_cell_s, L_cell_s, A_mem_s = sp.symbols('r_cell, vol_cell, L_cell, A_mem', real=True,
                                                             positive=True)

        # Mechanical variables and parameters:
        P_osmo_s, P_ind_s, sigma_H_s, sigma_L_s, epsilon_H_s, epsilon_L_s, Y_s, nu_s = sp.symbols(
            'P_osmo, P_ind, sigma_H, sigma_L, epsilon_H, epsilon_L, Y, nu', real=True)
        d_H_s, d_L_s = sp.symbols('d_H_s, d_L_s', real=True)

        # Osmotic and flow parameters:
        m_i_s, m_o_s, n_i_s = sp.symbols('m_i, m_o, n_i', real=True, positive=True)
        u_io_s, Q_io_s = sp.symbols('u_io, Q_io', real=True)

        # Physiological parameters:
        A_chan_s, N_chan_s = sp.symbols('A_chan, N_chan', real=True, positive=True)

        # Acceleration input parameters:
        dvol_cell_s, dn_i_s, dm_o_s = sp.symbols('dvol_cell_s, dn_i_s, dm_o_s', real=True)

        # define alpha = ro/(dmem*Y)
        alpha = sp.symbols('alpha', positive=True, real=True)

        # Regulatory parameters:
        d_gly_s, r_gly_s, n_gly_s, n_base = sp.symbols('d_gly, r_gly, n_gly, n_base', real=True, positive=True)
        m_gly_s = sp.symbols('m_gly', real=True, positive=True)
        K_1_s, epsilon_o_1_s, K_2_s, epsilon_o_2_s, b_1_s, b_2_s = sp.symbols(
            'K_1, epsilon_o_1, K_2, epsilon_o_2, b_1, b_2', real=True, positive=True)

        # Variables as functions that change in time:
        vol_cell_st = sp.Function('vol_cell')(t_s)
        m_o_st = sp.Function('m_o')(t_s)
        n_i_st = sp.Function('n_i')(t_s)
        n_gly_st = sp.Function('n_gly')(t_s)
        m_gly_st = sp.Function('m_gly')(t_s)

        # Volume change with transmembrane osmotic water flux:

        # Now we realize that while the concentration of osmolytes in the environment remains independent of osmotic water fluxes, that that in the cell changes:
        self.Eq0_mi_s = sp.Eq(m_i_s, n_i_s / vol_cell_s)

        # Osmotic pressure difference across membrane:
        self.Eq1_P_osmo_s = sp.Eq(P_osmo_s, (m_i_s - m_o_s) * R_s * T_s)

        # Substitute Eq0 into Eq1 for m_i_s:
        self.Eq2_P_osmo_s = sp.Eq(P_osmo_s, ((n_i_s / vol_cell_s) - m_o_s) * R_s * T_s)

        # Osmotic water flux across membrane via water channels:
        self.Eq3a_u_io_s = sp.Eq(u_io_s, (P_osmo_s * A_chan_s * N_chan_s) / (8 * mu_s * d_mem_s))

        # Substitute in Eq2 to Eq3 for P_osmo_s:
        self.Eq4_u_io_s = sp.Eq(u_io_s,
                           (((n_i_s / vol_cell_s) - m_o_s) * R_s * T_s * A_chan_s * N_chan_s) / (8 * mu_s * d_mem_s))

        # Expression for volumetric flow rate by multiplying by area over which flow occurs:
        self.Eq5_Q_io_s = sp.Eq(sp.diff(vol_cell_st, t_s), u_io_s * A_chan_s * N_chan_s)

        # Substitute in Eq3 for u_io to Eq4 to obtain volumetric flow rate in terms of core parameters:
        self.Eq6a_Q_io_s = sp.Eq(sp.diff(vol_cell_st, t_s),
                            ((n_i_s / vol_cell_s - m_o_s) * R_s * T_s * A_chan_s ** 2 * N_chan_s ** 2) / (
                                        8 * mu_s * d_mem_s)).simplify()

        self.Eq6a_Q_io_st = sp.Eq(sp.diff(vol_cell_st, t_s),
                             ((n_i_st / vol_cell_st - m_o_st) * R_s * T_s * A_chan_s ** 2 * N_chan_s ** 2) / (
                                         8 * mu_s * d_mem_s)).simplify()

        self.Eq7a_dQdt_io_st = sp.Eq(sp.diff(vol_cell_st, t_s, 2), sp.diff(
            ((n_i_st / vol_cell_st - m_o_st) * R_s * T_s * A_chan_s ** 2 * N_chan_s ** 2) / (8 * mu_s * d_mem_s),
            t_s)).simplify()

        # Steady-state cell volume value is very easy to find for this regime *in the uncontrolled sense*:
        # sp.solve(((n_i_s / vol_cell_s - m_o_s) * R_s * T_s * A_chan_s ** 2 * N_chan_s ** 2) / (8 * mu_s * d_mem_s),
        #          vol_cell_s)

        self.Eq8_vol_ss_s = sp.Eq(vol_cell_s, (m_i_s * vol_cell_o_s) / m_o_s)

        # Hoop stress equations in terms of pressure inside the cell, which is assumed to be generated by water influx leading to an elastic deformation of
        # the cell's perimeter (membrane or cell wall). To distinguish this mechanical pressure from osmotic pressure, we call it P_ind. At equillibrium, the P_osmo = P_ind:

        # Classical Hoop Stress Equations for a cylinder:
        self.Eq9_sigma_H_s = sp.Eq(sigma_H_s, (P_ind_s * r_cell_o_s) / d_mem_s)
        self.Eq10_sigma_L_s = sp.Eq(sigma_L_s, (P_ind_s * r_cell_o_s) / (2 * d_mem_s))

        # Hoop strain equations for a cylinder:
        # Given strain, solve for the stress:
        self.Eq11_epsilon_H = sp.Eq(epsilon_H_s, (1 / Y_s) * (sigma_H_s - nu_s * sigma_L_s))
        self.Eq12_epsilon_L = sp.Eq(epsilon_L_s, (1 / Y_s) * (sigma_L_s - nu_s * sigma_H_s))

        self.Eq13_epsilon_H_s = sp.Eq(epsilon_H_s, ((P_ind_s * r_cell_o_s) / (d_mem_s * Y_s)) * (1 - (nu_s / 2)))
        self.Eq14_epsilon_L_s = sp.Eq(epsilon_L_s, ((P_ind_s * r_cell_o_s) / (d_mem_s * Y_s)) * ((1 / 2) - nu_s))

        # Displacements in the circumferential ('H') and axis ('L') directions:
        self.Eq15_d_H_s = sp.Eq(d_H_s, epsilon_H_s * r_cell_o_s)
        self.Eq16_d_L_s = sp.Eq(d_L_s, epsilon_H_s * L_cell_o_s)

        self.solAB = sp.solve((self.Eq11_epsilon_H, self.Eq12_epsilon_L), (sigma_H_s, sigma_L_s))
        self.Eq17_sigma_H_s = self.solAB[sigma_H_s].simplify()
        self.Eq18_sigma_L_s = self.solAB[sigma_L_s].simplify()

        # If we have a volume at a particular strain, what is the corresponding Pressure that generated that strain?
        self.Eq19_vol_strain_s = sp.Eq(vol_cell_s, vol_cell_o_s * (1 + epsilon_H_s) * (1 + epsilon_L_s))

        # Substitute in expressions for epsilon_H and epsilon_L from the hoop stress model:
        self.Eq20_vol_strain_s = sp.Eq(vol_cell_s,
                                  vol_cell_o_s * (1 + ((P_ind_s * r_cell_o_s) / (d_mem_s * Y_s)) * (1 - (nu_s / 2))) * (
                                              1 + (((P_ind_s * r_cell_o_s) / (d_mem_s * Y_s)) * ((1 / 2) - nu_s))))


        self.Eq21_vol_strain_s = sp.Eq(vol_cell_s, vol_cell_o_s * (1 + ((P_ind_s * alpha) * (1 - (nu_s / 2)))) * (
                    1 + ((P_ind_s * alpha) * (sp.Rational(1, 2) - nu_s))))

        # Therefore, for the case where nu < 0.5, we have a solution for the inducing pressure as the positive root of:
        self.alpha = r_cell_o_s / (d_mem_s * Y_s)
        a = alpha ** 2 * (1 - nu_s / 2) * (sp.Rational(1, 2) - nu_s)
        b = sp.Rational(3, 2) * alpha * (1 - nu_s)
        c = 1 - (vol_cell_s / vol_cell_o_s)
        self.Eq22_Pind = sp.Eq(P_ind_s, (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a * c))

        # If we have a volume at a particular strain, and a Poisson ratio of 0.5, what is the corresponding Pressure that generated that strain?
        self.Eq23_vol_strain_s = sp.Eq(vol_cell_s, vol_cell_o_s * (1 + epsilon_H_s))

        # Substitute in expressions for epsilon_H and epsilon_L from the hoop stress model:
        self.Eq24_vol_strain_s = sp.Eq(vol_cell_s,
                                  vol_cell_o_s * (1 + ((P_ind_s * r_cell_o_s) / (d_mem_s * Y_s)) * (1 - (nu_s / 2))))

        sol_Pind_simple = sp.solve(self.Eq24_vol_strain_s, P_ind_s)[0].subs(nu_s, 0.5)  # P_ind for system with nu = 0.5

        self.Eq25_Pind = sp.Eq(P_ind_s, sol_Pind_simple)
        # If, Poisson's ratio can be said to be nu_s = 0.5, then the circumferential strain can be calculated simply in terms of volume:
        self.Eq26_vol_strain_simp_s = sp.Eq(epsilon_H_s, (vol_cell_s / vol_cell_o_s) - 1)

        # We next obtain velocity and acceleration of the volume change for this "elastic stretch" Regime where vol > vol_o:
        # Osmotic water flux across membrane via water channels:
        self.Eq3b_u_io_s = sp.Eq(u_io_s, ((P_osmo_s - P_ind_s) * A_chan_s * N_chan_s) / (8 * mu_s * d_mem_s))

        # Substitute in P_osmo as done before and P_ind as developed for nu = 0.5 case to obtain:
        self.Eq6b_Q_io_s = sp.Eq(sp.diff(vol_cell_st, t_s), ((((n_i_s / vol_cell_s - m_o_s) * R_s * T_s) -
                                                         ((4 * Y_s * d_mem_s * (vol_cell_s -
                                                                                vol_cell_o_s)) / (
                                                                      3 * r_cell_o_s * vol_cell_o_s))) * A_chan_s ** 2 * N_chan_s ** 2) / (
                                        8 * mu_s * d_mem_s))

        # Create an equivalent equation with time-dependent variables specified:
        self.Eq6b_Q_io_st = sp.Eq(sp.diff(vol_cell_st, t_s), ((((n_i_st / vol_cell_st - m_o_st) * R_s * T_s) -
                                                          ((4 * Y_s * d_mem_s * (vol_cell_st -
                                                                                 vol_cell_o_s)) / (
                                                                       3 * r_cell_o_s * vol_cell_o_s))) * A_chan_s ** 2 * N_chan_s ** 2) / (
                                         8 * mu_s * d_mem_s))

        # Obtain the acceleration by differentiating a second time:
        self.Eq7b_dQdt_io_st = sp.Eq(sp.diff(vol_cell_st, t_s, 2), sp.diff(((((n_i_st / vol_cell_st - m_o_st) * R_s * T_s) -
                                                                        ((4 * Y_s * d_mem_s * (vol_cell_st -
                                                                                               vol_cell_o_s)) / (
                                                                                     3 * r_cell_o_s * vol_cell_o_s))) * A_chan_s ** 2 * N_chan_s ** 2) / (
                                                                                  8 * mu_s * d_mem_s), t_s))

        self.Eq7a_dQdt_s = A_chan_s ** 2 * N_chan_s ** 2 * R_s * T_s * (
                    -n_i_s * dvol_cell_s - vol_cell_s ** 2 * dm_o_s + vol_cell_s * dn_i_s) / (
                                  8 * d_mem_s * mu_s * vol_cell_s ** 2)

        self.Eq7b_dQdt_s = A_chan_s ** 2 * N_chan_s ** 2 * (R_s * T_s * (
                    -n_i_s * dvol_cell_s / vol_cell_s ** 2 - dm_o_s + dn_i_s / vol_cell_s) - 4 * Y_s * d_mem_s * dvol_cell_s / (
                                                                   3 * r_cell_o_s * vol_cell_o_s)) / (
                                  8 * d_mem_s * mu_s)

        # Regulatory equations for osmoadaptation:
        self.f_act_1s = (1 - b_1_s) / (1 + sp.exp(
            -K_1_s * (epsilon_H_s + epsilon_o_1_s))) + b_1_s  # Expression for activator -- strain on glycerol decay
        self.f_inh_1s = 1 - (1 - b_2_s) / (1 + sp.exp(
            -K_2_s * (epsilon_H_s + epsilon_o_2_s)))  # Expression for inhibitor -- strain effect on glycerol production

        # Recalling that for our model with nu=0.5, the strain is in terms of the volume, we can substitute in this expression for eH:
        self.f_act_s = (1 - b_1_s) / (1 + sp.exp(-K_1_s * (((
                                                                   vol_cell_s / vol_cell_o_s) - 1) + epsilon_o_1_s))) + b_1_s  # Expression for activator -- strain on glycerol decay
        self.f_inh_s = 1 - (1 - b_2_s) / (1 + sp.exp(-K_2_s * (((
                                                                       vol_cell_s / vol_cell_o_s) - 1) + epsilon_o_2_s)))  # Expression for inhibitor -- strain effect on glycerol production

        self.Eq27_glyreg_s = sp.Eq(sp.diff(m_gly_st, t_s), self.f_inh_s * r_gly_s - self.f_act_s * d_gly_s * m_gly_s)

        # We can further see that the rate of change of moles in the cell is equal to the rate of change of glycerol as the base is assumed constant.
        self.Eq28a_ni_s = sp.Eq(n_i_st, n_base + n_gly_st)
        self.Eq28b_dnidt_s = sp.Eq(sp.diff(n_i_st, t_s), sp.diff(n_gly_st, t_s))

        # Finally, solve for steady-state solutions, where possible:
        # vol < vol_cell_o (topological regime):
        self.Eq29a_Vol_ss_R1_s = sp.solve(self.Eq6a_Q_io_s.rhs, vol_cell_s)[0]
        # vol >= vol_cell_o (elastic regime):
        self.Eq29b_Vol_ss_R2_s = sp.solve(self.Eq6b_Q_io_s.rhs, vol_cell_s)[1]  # two solutions; sol 1 is correct

        #------NUMERICAL TRANSFORMATION---------------------------------------------------------------------------------
        # Final step is to lambdify the equations to create functions suitable for numerical analysis:
        # Volume change velocity:
        # Regime 1 (vol < vol_o)
        self.dVdt_R1_f = sp.lambdify([vol_cell_s, m_o_s, n_i_s, A_chan_s, N_chan_s, R_s, T_s, d_mem_s, mu_s],
                                self.Eq6a_Q_io_s.rhs)
        # Regime 2 (vol >= vol_o)
        self.dVdt_R2_f = sp.lambdify(
            [vol_cell_s, m_o_s, n_i_s, A_chan_s, N_chan_s, R_s, T_s, d_mem_s, mu_s, Y_s, r_cell_o_s, vol_cell_o_s],
            self.Eq6b_Q_io_s.rhs)

        # Volume change acceleration:
        # Regime 1 (vol < vol_o)
        self.d2Vdt2_R1_f = sp.lambdify(
            [vol_cell_s, m_o_s, n_i_s, A_chan_s, N_chan_s, R_s, T_s, d_mem_s, mu_s, dvol_cell_s, dm_o_s, dn_i_s],
            self.Eq7a_dQdt_s)
        # Regime 2 (vol >= vol_o)
        self.d2Vdt2_R2_f = sp.lambdify(
            [vol_cell_s, m_o_s, n_i_s, A_chan_s, N_chan_s, R_s, T_s, d_mem_s, mu_s, Y_s, r_cell_o_s, vol_cell_o_s,
             dvol_cell_s, dm_o_s, dn_i_s], self.Eq7b_dQdt_s)

        # Steady-state solutions for volume:
        # Regime 1 (vol < vol_o)
        self.Vss_R1_f = sp.lambdify([m_o_s, n_i_s], self.Eq29a_Vol_ss_R1_s)
        # Regime 2 (vol >= vol_o)
        self.Vss_R2_f = sp.lambdify([m_o_s, n_i_s, R_s, T_s, d_mem_s, Y_s, r_cell_o_s, vol_cell_o_s], self.Eq29b_Vol_ss_R2_s)

        # rate of change of glycerol; rate of change of total osmolytes in cell:
        # dni_dt_f = sp.lambdify([epsilon_H_s, K_1_s, K_2_s, b_1_s, b_2_s, d_gly_s, epsilon_o_1_s, epsilon_o_2_s, n_gly_s, r_gly_s, vol_cell_o_s], Eq26_glyreg_s.rhs)
        # dni_dt_f = sp.lambdify([vol_cell_s, K_1_s, K_2_s, b_1_s, b_2_s, d_gly_s, epsilon_o_1_s, epsilon_o_2_s, n_gly_s, r_gly_s, vol_cell_o_s], Eq26_glyreg_s.rhs)
        self.dmi_dt_f = sp.lambdify(
            [vol_cell_s, K_1_s, K_2_s, b_1_s, b_2_s, d_gly_s, epsilon_o_1_s, epsilon_o_2_s, m_gly_s, r_gly_s,
             vol_cell_o_s], self.Eq27_glyreg_s.rhs)

        # Activation and inhibition functions:
        self.f_act_1f = sp.lambdify([epsilon_H_s, b_1_s, K_1_s, epsilon_o_1_s], self.f_act_1s)
        self.f_inh_1f = sp.lambdify([epsilon_H_s, b_2_s, K_2_s, epsilon_o_2_s], self.f_inh_1s)

        self.f_act_f = sp.lambdify([vol_cell_s, b_1_s, K_1_s, epsilon_o_1_s, vol_cell_o_s], self.f_act_s)
        self.f_inh_f = sp.lambdify([vol_cell_s, b_2_s, K_2_s, epsilon_o_2_s, vol_cell_o_s], self.f_inh_s)

        # Pressures:
        self.P_osmo_f = sp.lambdify([m_i_s, m_o_s, R_s, T_s], self.Eq1_P_osmo_s)
        self.P_ind_f = sp.lambdify([vol_cell_s, vol_cell_o_s, r_cell_o_s, Y_s, d_mem_s], self.Eq25_Pind)