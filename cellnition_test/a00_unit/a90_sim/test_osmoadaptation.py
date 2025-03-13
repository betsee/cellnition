#!/usr/bin/env python3
# --------------------( LICENSE                            )--------------------
# Copyright (c) 2023-2025 Alexis Pietak & Cecil Curry.
# See "LICENSE" for further details.

'''
**Osmoadaptation** unit tests.

This submodule unit tests the public API of the :mod:`cellnition.science.osmoadaptation`
subpackage.
'''
def test_osmo_model(tmp_path) -> None:
    '''
    Test the yeast osmoadaptation model.

    '''
    import numpy as np
    from scipy import signal
    from cellnition.science.osmoadaptation.model_params import ModelParams
    from cellnition.science.osmoadaptation.analytic_model import AnalyticOsmoticModel
    from cellnition.science.osmoadaptation.osmo_model import OsmoticCell

    # Write out all of the equations for the model in an analytic method:
    write_eqns = AnalyticOsmoticModel()

    # Define the numerical model:
    ocell = OsmoticCell()
    p = ModelParams()  # Define a model params object

    Np = 15
    vol_vect = np.linspace(0.35 * p.vol_cell_o, 2.0 * p.vol_cell_o, Np)
    ni_vect = np.linspace(0.25 * p.m_o_base * p.vol_cell_o, 1500.0 * p.vol_cell_o, Np)
    mo_vect = np.linspace(p.m_o_base, 1000.0, Np)

    MM, NN, VV = np.meshgrid(mo_vect, ni_vect, vol_vect, indexing='ij')

    dVdt_vect, dndt_vect, _ = ocell.state_space_gen(MM.ravel(),
                                                    VV.ravel(),
                                                    NN.ravel(),
                                                    p.m_i_gly,
                                                    p.d_wall,
                                                    p.Y_wall,
                                                    p,
                                                    synth_gly=True
                                                    )

    # Compute steady-state solutions:
    # Need to calculate solutions over the full domain first, then find solutinos that match the region criteria:
    Vss_vect = ocell.osmo_vol_steady_state(MM.ravel(), NN.ravel(), p.Y_wall, p.d_wall, p)

    # Compute the volume change vector magnitude for case where dndt is nonzero:
    dVNdt_vect = np.sqrt(dVdt_vect ** 2 + dndt_vect ** 2)

    # Time series testing:
    end_t = 3600 * 5
    del_t = 1.0e-1
    t_steps = int(end_t / del_t)
    t_vect = np.linspace(0.0, end_t, t_steps)

    m_o_d = 600.0  # Medium hypertonic

    mo_vect = (m_o_d - p.m_o_base) * (1 - ((signal.square(t_vect * 0.0004 * np.pi, 0.5) + 1) / 2)) + p.m_o_base
    mo_vect[150000:] = p.m_o_base
    samp_i = 10

    osmo_data_bio1 = ocell.osmo_time_sim(
                                        t_vect,
                                        mo_vect,
                                        p.vol_cell_o,
                                        p.n_i_base,
                                        p.d_wall,
                                        p.Y_wall,
                                        del_t,
                                        samp_i,
                                        p,
                                        synth_gly=True
                                    )

    osmo_data_f1 = ocell.osmo_time_sim(
                                        t_vect,
                                        mo_vect,
                                        p.vol_cell_o,
                                        p.n_i_base,
                                        p.d_wall,
                                        p.Y_wall,
                                        del_t,
                                        samp_i,
                                        p,
                                        synth_gly=False
                                    )