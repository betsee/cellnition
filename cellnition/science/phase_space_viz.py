#!/usr/bin/env python3
# --------------------( LICENSE                           )--------------------
# Copyright (c) 2023-2024 Alexis Pietak.
# See "LICENSE" for further details.

'''
This module implements a 'brute force' style approach to the network as a dynamic system,
allowing for plots and visualizations of phase portraits and optimization functions.

'''
import csv
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import colormaps
from scipy.optimize import minimize, fsolve
import sympy as sp
from cellnition.science.network_enums import EdgeType, GraphType, NodeType
from cellnition.science.gene_networks import GeneNetworkModel
import pyvista as pv

# FIXME: Add in linear plot
# FIXME: Add in 2d vector plot

class PhaseSpace(object):
    '''

    '''
    def __init__(self, gmod: GeneNetworkModel):
        '''

        '''

        if gmod.dcdt_vect_f is None:
            raise Exception("Must use the method build_analytical_model to generate attributes"
                            "to use this function.")

        self._gmod = gmod

    def brute_force_phase_space(self,
                                N_pts: int=15,
                                cmin: float=0.0,
                                cmax: float|list=1.0,
                                Ki: float|list=0.5,
                                ni:float|list=3.0,
                                ri:float|list=1.0,
                                di:float|list=1.0,
                                zer_thresh: float=0.01,
                                include_signals: bool = False
                                ):
        '''

        '''

        # Create parameter vectors for the model:
        self._gmod.create_parameter_vects(Ki, ni, ri, di)

        if self._gmod._reduced_dims and self._gmod._solved_analytically is False:
            N_nodes = len(self._gmod.c_vect_reduced_s)
            dcdt_vect_f = self._gmod.dcdt_vect_reduced_f
            c_vect_s = self._gmod.c_vect_reduced_s
        else:
            N_nodes = self._gmod.N_nodes
            dcdt_vect_f = self._gmod.dcdt_vect_f
            c_vect_s = self._gmod.c_vect_s

        c_vect_set, C_M_SET, c_lin_set = self._gmod.generate_state_space(c_vect_s,
                                                                   Ns=N_pts,
                                                                   cmin=cmin,
                                                                   cmax=cmax,
                                                                   include_signals=include_signals)

        M_shape = C_M_SET[0].shape

        dcdt_M = np.zeros(c_vect_set.shape)

        for i, c_vecti in enumerate(c_vect_set):
            if self._gmod._include_process is False:
                dcdt_i = dcdt_vect_f(c_vecti,
                                     self._gmod.r_vect,
                                     self._gmod.d_vect,
                                     self._gmod.K_vect,
                                     self._gmod.n_vect)
            else:
                dcdt_i = dcdt_vect_f(c_vecti,
                                     self._gmod.r_vect,
                                     self._gmod.d_vect,
                                     self._gmod.K_vect,
                                     self._gmod.n_vect,
                                     self._gmod.process_params_f)
            dcdt_M[i] = dcdt_i * 1

        dcdt_M_set = []
        for dci in dcdt_M.T:
            dcdt_M_set.append(dci.reshape(M_shape))

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