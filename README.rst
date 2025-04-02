.. # ------------------( SEO                                 )------------------
.. # Metadata converted into HTML-specific meta tags parsed by search engines.
.. # Note that:
.. # * The "description" should be no more than 300 characters and ideally no
.. #   more than 150 characters, as search engines may silently truncate this
.. #   description to 150 characters in edge cases.

.. #FIXME: Fill this description in with meaningful content, please.
.. meta::
   :description lang=en:
     Analyze gene regulatory networks (GRNs) via Network Finite State Machines (NFSMs).

.. # ------------------( SYNOPSIS                            )------------------

===================
|cellnition-banner|
===================

|ci-badge|

**Cellnition** is an open source simulator to create and analyze Network Finite
State Machines (NFSMs) from gene regulatory network (GRN) models.

Regulatory networks such as GRNs preside over complex phenomena in biological systems, 
yet given a specific regulatory network, how do we know what it's capable of doing?

**Cellnition** treats regulatory networks as analogue computers, where NFSMs map the sequential
logic inherent in the network as a dissipative dynamic system. As an extension and 
improvement upon attractor landscape analysis, NFSMs reveal the analogue computing 
operations inherent in GRNs, allowing for identification of associated "intelligent 
behaviors".  NFSMs capture the "analog programming" of GRNs, which aim to provide clear identification of:

* Interventions that may induce transitions between stable states (e.g. from "diseased" to "healthy") 
* Identification of path-dependencies, representing stable changes occurring after a transient intervention is applied (e.g. predicting if a transient treatment with pharmacological agent might permanently heal a condition)
* Identification of inducible cycles of behavior that may take the system through a complex multi-phase process (e.g. wound healing). 

NFSMs have a range of potential applications, including the identification of potential strategies to 
renormalize cancer (see `Tutorial 2`_). 

Read more about **Cellnition's** NFSMs in our pre-print publication: 
`Harnessing the Analogue Computing Power of Regulatory Networks with the 
Regulatory Network Machine <preprint_>`__. 

Please cite our pre-print in any work that utilizes **Cellnition**::

Pietak, Alexis, and Levin, Michael.
 “Harnessing the Analog Computing Power of Regulatory Networks 
 with the Regulatory Network Machine.” OSF Preprints, 2 Dec. 2024. Web.

**Cellnition** is `portably implemented <cellnition codebase_>`__ in Python_,
`continuously stress-tested <cellnition tests_>`__ via `GitHub Actions`_ **×**
tox_ **×** pytest_  **×** Codecov_, and `licensed <cellnition
license_>`__ under a non-commercial use, open source `APACHE license`_ with Tufts Open Source License Rider v.1. 
For maintainability, cellnition officially supports *only* the most recently released version of CPython_.

.. # ------------------( TABLE OF CONTENTS                   )------------------
.. # Blank line. By default, Docutils appears to only separate the subsequent
.. # table of contents heading from the prior paragraph by less than a single
.. # blank line, hampering this table's readability and aesthetic comeliness.

|

.. # Table of contents, excluding the above document heading. While the
.. # official reStructuredText documentation suggests that a language-specific
.. # heading will automatically prepend this table, this does *NOT* appear to
.. # be the case. Instead, this heading must be explicitly declared.

.. contents:: **Contents**
   :local:

.. # ------------------( DESCRIPTION                         )------------------

Install
=======

**Cellnition** is installable with pip_, the standard package installer
officially bundled with Python_:

.. code-block:: bash

   pip3 install cellnition

Documentation
=============

Comprehensive documentation for **Cellnition** is available at our `GitHub Pages hosted site <cellnition docs_>`__.

Features
=========
**Cellnition** embodies a range of functionality, including:

* Work with regulatory networks imported from **Cellnition's** network-library, use **Cellnition** to procedurally generate regulatory networks with random or scale-free degree distributions, or import your own user-defined regulatory networks as directed graphs with activating or inhibiting edge characteristics (see `Tutorial 1`_ and `Tutorial 2`_ for some examples).
* Analyze and characterize regulatory network graphs with a variety of metrics (see the `characterize_graph`_ method and `Tutorial 1`_ and `Tutorial 2`_). 
* Use directed graph representations of regulatory networks to build fully-continuous, differential equation based simulators of network dynamics (see `ProbabilityNet`_ class and `Tutorial 1`_). 
* Use directed graph representations of regulatory networks to build logic equation based Boolean simulators of network dynamics (see `BooleanNet`_ class and `Tutorial 2`_).
* Explore regulatory network dynamics with comprehensive equillibrium state search and characterization capabilities, along with temporal simulators (see `Tutorial 1`_ and `Tutorial 2`_ for some examples).
* Create simulated datasets, including simulation of automated gene-knockout experiments for a continuous regulatory network model (see `GeneKnockout`_ class). 
* Generate Network Finite State Machines (NFSMs) for continuous models (see `Tutorial 1`_) or for Boolean models (see `Tutorial 2`_). 
* Create and export a variety of plots and visualizations, including of the regulatory network analytic equations, regulatory network directed graphs, heatmaps of gene expressions in equilibrium states, gene expressions in temporal simulations, and depictions of the general and event-driven NFSMs (see `Tutorial 1`_ and `Tutorial 2`_ for some examples).     

Tutorials
=========

**Cellnition** tutorials are available as `Jupyter Notebooks <Jupyter_>`__:

* `Tutorial 1`_ : Create NFSMs from a continuous, differential-equation based GRN model.
* `Tutorial 2`_ : Create NFSMs from a Boolean, logic-equation based GRN model.

Acknowledgements
================

**Cellnition** creator Alexis Pietak is grateful for collaboration opportunities and funding support
from the `Levin Lab <Levin Lab_>`__ at `Tufts University <Tufts_>`__, via a `Templeton World Charity Foundation grant 0606 <TWCFGrant_>`__.
Thanks goes out to Brian Curry for his volunteer assistance with the *Cellnition* code repository. 

Disclaimer
==========

Alexis Pietak created **Cellnition**, and co-authored the associated `scientific manuscript <preprint_>`__, as an 
external contractor ("Affiliate Research Scientist") for `Tufts University <Tufts_>`__. Under the established `terms <TuftsRoyalties_>`__ of 
`Tufts University <Tufts_>`__, as an external contractor, Alexis recieves no royalties nor other financial benefits or entitlements in connection with any intellectual 
property associated with the **Cellnition** project or its associated `scientific manuscript <preprint_>`__.

Note that the functionality provided and work presented in **Cellnition** and its associated `scientific manuscript <preprint_>`__ are at a 
theoretical and computational stage, with any and all potential applications for the work requiring verification by real world experiments and comprehensive testing.   

License
=======

**Cellnition** is non-commerical use, open source software `licensed <cellnition license_>`__ under an
`Apache 2.0 license <APACHE license_>`__ with Tufts Open Source License Rider v.1, restricting use
to academic purposes only.

.. # ------------------( IMAGES                              )------------------
.. |cellnition-banner| image:: https://github.com/user-attachments/assets/50f45c9b-980a-473f-9362-361d3f62061a
   :target: https://github.com/betsee/cellnition
   :alt: Cellnition

.. # ------------------( IMAGES ~ badge                      )------------------
.. |app-badge| image:: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
   :target: https://cellnition.streamlit.app
   :alt: Cellnition web app (graciously hosted by Streamlit Cloud)
.. |ci-badge| image:: https://github.com/betsee/cellnition/workflows/test/badge.svg
   :target: https://github.com/betsee/cellnition/actions?workflow=test
   :alt: Cellnition continuous integration (CI) status

.. # ------------------( LINKS ~ cellnition : local          )------------------
.. _cellnition License:
   LICENSE
.. _Tutorial 1:
   ipynb/Tutorial1_ContinuousNFSM_v1.ipynb
.. _Tutorial 2:
   ipynb/Tutorial2_BooleanNFSM_v1.ipynb

.. # ------------------( LINKS ~ cellnition : package        )------------------
.. #FIXME: None of these exist, naturally. *sigh*
.. _cellnition Anaconda:
   https://anaconda.org/conda-forge/cellnition
.. _cellnition PyPI:
   https://pypi.org/project/cellnition

.. # ------------------( LINKS ~ cellnition : remote         )------------------
.. _cellnition:
   https://gitlab.com/betsee/cellnition
.. _cellnition app:
   https://cellnition.streamlit.app
.. _cellnition codebase:
   https://gitlab.com/betsee/cellnition
.. _cellnition docs:
   https://betsee.github.io/cellnition
.. _cellnition pulls:
   https://gitlab.com/betsee/cellnition/-/merge_requests
.. _cellnition tests:
   https://gitlab.com/betsee/cellnition/actions?workflow=tests

.. # ------------------( LINKS ~ cellnition : internal         )------------------
.. _characterize_graph:
   https://github.com/betsee/cellnition/blob/4b1e6b78c725cdb24bcf82b7c259ad6e726f42ce/cellnition/science/network_models/network_abc.py#L214
.. _ProbabilityNet:
   https://github.com/betsee/cellnition/blob/4b1e6b78c725cdb24bcf82b7c259ad6e726f42ce/cellnition/science/network_models/probability_networks.py#L30
.. _BooleanNet:
   https://github.com/betsee/cellnition/blob/4b1e6b78c725cdb24bcf82b7c259ad6e726f42ce/cellnition/science/network_models/boolean_networks.py#L25
.. _GeneKnockout:
   https://github.com/betsee/cellnition/blob/4b1e6b78c725cdb24bcf82b7c259ad6e726f42ce/cellnition/science/networks_toolbox/gene_knockout.py#L18

.. # ------------------( LINKS ~ github                      )------------------
.. _GitHub Actions:
   https://github.com/features/actions

.. # ------------------( LINKS ~ py                          )------------------
.. _Python:
   https://www.python.org
.. _pip:
   https://pip.pypa.io

.. # ------------------( LINKS ~ py : interpreter            )------------------
.. _CPython:
   https://github.com/python/cpython

.. # ------------------( LINKS ~ py : package : science      )------------------
.. _Jupyter:
   https://jupyter.org

.. # ------------------( LINKS ~ py : package : test         )------------------
.. _Codecov:
   https://about.codecov.io
.. _pytest:
   https://docs.pytest.org
.. _tox:
   https://tox.readthedocs.io

.. # ------------------( LINKS ~ py : package : web          )------------------
.. _Streamlit:
   https://streamlit.io

.. # ------------------( LINKS ~ py : service                )------------------
.. _Anaconda:
   https://docs.conda.io/en/latest/miniconda.html
.. _PyPI:
   https://pypi.org

.. # ------------------( LINKS ~ external                    )------------------
.. _preprint:
   https://osf.io/preprints/osf/tb5ys_v1
.. _Levin Lab:
   https://as.tufts.edu/biology/levin-lab
.. _Tufts:
   https://www.tufts.edu
.. _TWCFGrant:
   https://www.templetonworldcharity.org/projects-resources/project-database/0606
.. _TuftsRoyalties: 
   https://viceprovost.tufts.edu/policies-forms-guides/policy-rights-and-responsibilities-respect-intellectual-property

.. # ------------------( LINKS ~ soft : license             )------------------
.. _MIT license:
   https://opensource.org/licenses/MIT
.. _APACHE license:
   https://www.apache.org/licenses/LICENSE-2.0
