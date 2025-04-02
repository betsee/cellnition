# Basic Usage

The Jupyter Notebook [Tutorials](https://github.com/betsee/cellnition#tutorials) are a great 
place to get started with *Cellnition*. 

The general workflow to create NFSMs in *Cellnition* comprises 6 main steps:

1. Define [via edges][cellnition.science.network_models.network_abc.NetworkABC.build_network_from_edges]
   (user-defined or [imported][cellnition.science.network_models.network_library]), or 
[procedurally generate][cellnition.science.network_models.network_abc.NetworkABC.randomly_generate_special_network] 
a regulatory network as a directed graph in a 
[computationally-agnostic][cellnition.science.network_models.basic_network.BasicNet], 
[Continuous][cellnition.science.network_models.probability_networks.ProbabilityNet], or 
[Boolean][cellnition.science.network_models.boolean_networks.BooleanNet] oriented computational workflow.
2. [Characterize the graph][cellnition.science.network_models.network_abc.NetworkABC.characterize_graph] 
to automatically categorize input, output & internal nodes,
along with other features such as cycles and node hierarchical level.
3. Build a computational model of the regulatory network for 
[Continuous][cellnition.science.network_models.probability_networks.ProbabilityNet.build_analytical_model] 
or 
[Boolean][cellnition.science.network_models.boolean_networks.BooleanNet.build_boolean_model] 
models.
4. Use the computational model in a state machine to identify all unique equilibrium 
output states for each input state via state space search
available for [Continuous][cellnition.science.networks_toolbox.state_machine.StateMachine.steady_state_solutions_search] 
or [Boolean][cellnition.science.networks_toolbox.boolean_state_machine.BoolStateMachine.steady_state_solutions_search] 
models.
5. Create the NFSMs in 
[Continuous][cellnition.science.networks_toolbox.state_machine.StateMachine.create_transition_network] 
or 
[Boolean][cellnition.science.networks_toolbox.boolean_state_machine.BoolStateMachine.create_transition_network] 
systems. Cellnition does this by starting the system in each equilibrium state, 
applying each input state, and determining the equilibrium state that the system transitions 
to.
6. Output, plot, and analyze the resulting Network Finite State Machines!

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
[TWCFGrant]: https://www.templetonworldcharity.org/projects-resources/project-database/0606
[TuftsRoyalties]: https://viceprovost.tufts.edu/policies-forms-guides/policy-rights-and-responsibilities-respect-intellectual-property