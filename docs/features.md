# Features

*Cellnition* embodies a range of functionality, including:

- Work with regulatory networks imported from *Cellnition's* 
[`network_library`][cellnition.science.network_models.network_library],
use *Cellnition* to
[procedurally generate regulatory networks][cellnition.science.network_models.network_abc.NetworkABC.randomly_generate_special_network] 
with random or scale-free degree distributions, or import your own user-defined regulatory
networks as directed graphs with activating or inhibiting edge characteristics
(see [Tutorial 1][] and [Tutorial 2][] for some examples).
- Analyze and characterize regulatory network graphs with a variety of metrics
(see the [`characterize_graph`][cellnition.science.network_models.network_abc.NetworkABC.characterize_graph] 
method and [Tutorial 1][] and [Tutorial 2][]).
- Use directed graph representations of regulatory networks to build fully-continuous,
differential-equation based simulators of network dynamics (see 
[`ProbabilityNet`][cellnition.science.network_models.probability_networks.ProbabilityNet.build_analytical_model] 
and [Tutorial 1][]).
- Use directed graph representations of regulatory networks to build logic-equation based Boolean
simulators of network dynamics (see [`BooleanNet`][cellnition.science.network_models.boolean_networks.BooleanNet.build_boolean_model]  
and [Tutorial 2][]).
- Explore regulatory network dynamics with comprehensive 
[equilibrium state search][cellnition.science.networks_toolbox.state_machine.StateMachine.steady_state_solutions_search] 
and characterization capabilities, along with 
[temporal simulators][cellnition.science.networks_toolbox.state_machine.StateMachine.sim_time_trajectory] 
(see [Tutorial 1][] and
[Tutorial 2][] for some examples).
- Create simulated datasets, including simulation of automated gene-knockout experiments
 for a continuous regulatory network model (see 
[`GeneKnockout`][cellnition.science.networks_toolbox.gene_knockout.GeneKnockout]).
- Generate NFSMs for continuous models (see 
[`StateMachine`][cellnition.science.networks_toolbox.state_machine.StateMachine]
and [Tutorial 1][])
 or for Boolean models (see 
[`BoolStateMachine`][cellnition.science.networks_toolbox.boolean_state_machine.BoolStateMachine]
and [Tutorial 2][]).
- Create and export a variety of plots and visualizations, including of the regulatory network
model [analytic equations][cellnition.science.network_models.probability_networks.ProbabilityNet.save_model_equations], 
[regulatory network directed graphs][cellnition.science.networks_toolbox.netplot.plot_network], 
[heatmaps of gene expressions][cellnition.science.network_models.network_abc.NetworkABC.plot_sols_array] 
in equilibrium states, 
gene expressions in [temporal simulations][cellnition.science.networks_toolbox.state_machine.StateMachine.plot_time_trajectory], 
and depictions of the [general NFSM][cellnition.science.networks_toolbox.state_machine.StateMachine.plot_state_transition_network] 
and the [event-driven NFSM][cellnition.science.networks_toolbox.state_machine.StateMachine.plot_state_perturbation_network]
(see [Tutorial 1][] and [Tutorial 2][] for some examples).


[Link References]::
[Levin Lab]: https://as.tufts.edu/biology/levin-lab
[CPython]: https://github.com/python/cpython
[Codecov]: https://about.codecov.io
[pytest]: https://docs.pytest.org
[tox]: https://tox.readthedocs.io
[Python]: https://www.python.org
[Github Actions]: https://github.com/features/actions
[Tufts University]: https://www.tufts.edu
[APACHE license]: https://www.apache.org/licenses/LICENSE-2.0
[license]: https://github.com/betsee/cellnition/blob/main/LICENSE
[Tutorial 1]: https://github.com/betsee/cellnition/blob/main/ipynb/Tutorial1_ContinuousNFSM_v1.ipynb
[Tutorial 2]: https://github.com/betsee/cellnition/blob/main/ipynb/Tutorial2_BooleanNFSM_v1.ipynb