#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2025 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module implements a 'brute force' style approach to the network as a dynamic system,
allowing for plots and visualizations of phase portraits and optimization functions on
points of a grid in a phase space.

'''
import numpy as np
from numpy import ndarray
from cellnition.science.network_models.probability_networks import ProbabilityNet
import pyvista as pv

# FIXME: Add in linear plot
# FIXME: Add in 2d vector plot

class PhaseSpace(object):
    '''

    '''
    def __init__(self, pnet: ProbabilityNet):
        '''
        Initialize the PhaseSpace object.

        Parameters
        ----------
        pnet : GeneNetworkModel
            An instance of GeneNetworkModel, which has an analytical model already built.

        '''

        self._pnet = pnet

    def brute_force_phase_space(self,
                                N_pts: int=15,
                                constrained_inds: list|None = None,
                                constrained_vals: list|None = None,
                                beta_base: float | list=2.0,
                                n_base: float | list=3.0,
                                d_base: float | list=1.0,
                                zer_thresh: float=0.01,
                                ):
        '''
        Generate a sampling of the phase space of the system on multiple dimensions, and calculate
        the instantaneous change vector at each point of the space.

        Parameters
        ------------
        N_pts : int=15
            Number of points to sample along each axis of the phase space.

        cmin : float=0.0
            Minimum value of concentration to start sampling.
        cmax : float|list=1.0
            Maximum value of concentration to stop sampling.

        Ki : float|list=0.5
            Value or list of Hill constants for each concentration in the system.
            If a float is specified, all concentrations will use the same value.

        n_base : float|list=3.0
            Value or list of Hill exponents for each concentration in the system.
            If a float is specified, all concentrations will use the same value.

        ri : float|list=1.0
            Value or list of maximum production rates for each concentration in the system.
            If a float is specified, all concentrations will use the same value.

        d_base : float|list=1.0
            Value or list of maximum decay rates for each concentration in the system.
            If a float is specified, all concentrations will use the same value.

        zer_thresh : float=0.01
            Value to use as a threshold for assessing points where the magnitude of the rate of change is zero
            (value at which the equilibrium points are assessed).

        include_signals : bool = False
            Include any signal nodes in the network dynamics or exclude them?

        '''

        dcdt_vect_f, dcdt_jac_f = self._pnet.create_numerical_dcdt(constrained_inds=constrained_inds,
                                                             constrained_vals=constrained_vals)
        if constrained_inds is None or constrained_vals is None:
            unconstrained_inds = self._pnet._nodes_index
        else:
            unconstrained_inds = np.setdiff1d(self._pnet._nodes_index, constrained_inds).tolist()

        c_vect_set, c_lin_set, C_M_SET = self._pnet.generate_state_space(unconstrained_inds,
                                                                         N_pts)

        M_shape = C_M_SET[0].shape

        dcdt_M = np.zeros(c_vect_set.shape)

        function_args = self._pnet.get_function_args(constraint_vals=constrained_vals,
                                               d_base=d_base,
                                               n_base=n_base,
                                               beta_base=beta_base)

        for i, c_vecti in enumerate(c_vect_set):
            dcdt_i = dcdt_vect_f(c_vecti[unconstrained_inds], *function_args)
            dcdt_M[i, unconstrained_inds] = dcdt_i * 1

        dcdt_M[:, constrained_inds] = constrained_vals

        dcdt_M_set = []
        for dci in dcdt_M.T:
            dcdt_M_set.append(dci.reshape(C_M_SET.shape))

        dcdt_M_set = np.asarray(dcdt_M_set)
        dcdt_dmag = np.sqrt(np.sum(dcdt_M_set ** 2, axis=0))
        system_sols = ((dcdt_dmag / dcdt_dmag.max()) < zer_thresh).nonzero()

        return system_sols, dcdt_M_set, dcdt_dmag, c_lin_set, C_M_SET

    def plot_3d_streamlines(self,
                            c0: ndarray,
                            c1: ndarray,
                            c2: ndarray,
                            dc0: ndarray,
                            dc1: ndarray,
                            dc2: ndarray,
                            point_data: ndarray|None = None,
                            axis_labels: list|tuple|ndarray|None=None,
                            n_points: int=100,
                            source_radius: float=0.5,
                            source_center: tuple[float, float, float]=(0.5, 0.5, 0.5),
                            tube_radius: float=0.003,
                            arrow_scale: float=1.0,
                            lighting: bool = False,
                            cmap: str = 'magma'
                            ):
        '''

        '''

        pvgrid = pv.RectilinearGrid(c0, c1, c2)  # Create a structured grid for our space

        if point_data is not None:
            pvgrid.point_data["Magnitude"] = point_data.ravel()

        if axis_labels is not None:
            labels = dict(xtitle=axis_labels[0], ytitle=axis_labels[1], ztitle=axis_labels[2])
        else:
            labels = dict(xtitle='c0', ytitle='c1', ztitle='c2')

        vects_control = np.vstack((dc0.T.ravel(), dc1.T.ravel(), dc2.T.ravel())).T

        # vects_control = np.vstack((np.zeros(dndt_vect.shape), np.zeros(dndt_vect.shape), dVdt_vect/p.vol_cell_o)).T
        pvgrid["vectors"] = vects_control * 0.1
        pvgrid.set_active_vectors("vectors")

        streamlines, src = pvgrid.streamlines(vectors="vectors",
                                              return_source=True,
                                              n_points=n_points,
                                              source_radius=source_radius,
                                              source_center=source_center
                                              )

        arrows = streamlines.glyph(orient="vectors", factor=arrow_scale)

        pl = pv.Plotter()
        pl.add_mesh(streamlines.tube(radius=tube_radius), lighting=lighting, cmap=cmap)
        pl.add_mesh(arrows, cmap=cmap)
        pl.remove_scalar_bar("vectors")
        pl.remove_scalar_bar("GlyphScale")
        pl.show_grid(**labels)

        return pl