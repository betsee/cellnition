# Welcome to Cellnition Docs!

*Cellnition* is an open source simulator to create and analyze Network Finite State Machines (NFSMs)
from regulatory network models.

Regulatory networks, such as Gene Regulatory Networks (GRNs), preside over so many complex phenomena
in biological systems, yet given a specific regulatory network, how do we know what it's capable of doing?

*Cellnition* aims to provide a detailed answer to that question, by treating regulatory
networks as analogue computers, where NFSMs map the sequential logic, 
or analogue "software program",
inherent in the regulatory network's dynamics. 

As an extension and improvement upon attractor landscape analysis,
NFSMs reveal the analogue computing operations inherent in regulatory networks,
allowing for identification of associated "intelligent behaviors".

By capturing the analog programming of regulatory networks, 
NFSMs provide clear identification of:

- Interventions that can induce transitions between stable states (e.g. from "diseased" to "healthy").
- Identification of path-dependencies, representing stable changes occurring after a transient intervention
is applied (e.g. evaluating if a transient treatment with pharmacological agent can permanently heal a condition).
- Identification of inducible cycles of behavior that take the system through a complex
multi-phase process (e.g. wound healing).
- How to target regimes of dynamic activity in a regulatory network (e.g. how to engage a
genetic oscillator instead of a monotonic gene expression in time).

NFSMs have a range of applications, including facilitating the 
identification of strategies to renormalize cancer.

Read more about *Cellnition's* NFSMs in our publication: [Harnessing the Analogue Computing Power of
Regulatory Networks with the Regulatory Network Machine](https://osf.io/preprints/osf/tb5ys_v1).

Please cite our publication in any work that utilizes *Cellnition*:

```
Pietak, Alexis, and Michael Levin.
 “Harnessing the Analog Computing Power of Regulatory Networks 
 with the Regulatory Network Machine.” OSF Preprints, 2 Dec. 2024. Web.
```

## Installation

*Cellnition* is easily installable with [pip](https://pip.pypa.io), 
the standard package installer
officially bundled with [Python](https://www.python.org):

```bash
pip install cellnition
```

## Features

*Cellnition* embodies a range of functionality, including:

- Work with regulatory networks imported from *Cellnition's* 
[`network_library`][cellnition.science.network_models.network_library],
use *Cellnition* to
[procedurally generate regulatory networks][cellnition.science.network_models.network_abc.NetworkABC] 
with random or scale-free degree distributions, or import your own user-defined regulatory
networks as directed graphs with activating or inhibiting edge characteristics
(see [Tutorial 1][] and [Tutorial 2][] for some examples).
- Analyze and characterize regulatory network graphs with a variety of metrics
(see the [`characterize_graph`][cellnition.science.network_models.network_abc.NetworkABC] method
and [Tutorial 1][] and [Tutorial 2][]).
- Use directed graph representations of regulatory networks to build fully-continuous,
differential-equation based simulators of network dynamics (see 
[`ProbabilityNet`][cellnition.science.network_models.probability_networks.ProbabilityNet] 
and [Tutorial 1][]).
- Use directed graph representations of regulatory networks to build logic-equation based Boolean
simulators of network dynamics (see [`BooleanNet`][cellnition.science.network_models.boolean_networks.BooleanNet]  
and [Tutorial 2][]).
- Explore regulatory network dynamics with comprehensive equilibrium state search and
 characterization capabilities, along with temporal simulators (see [Tutorial 1][] and
[Tutorial 2][] for some examples).
- Create simulated datasets, including simulation of automated gene-knockout experiments
 for a continuous regulatory network model (see [`GeneKnockout`]()).
- Generate NFSMs for continuous models (see 
[`StateMachine`][cellnition.science.networks_toolbox.state_machine.StateMachine]
and [Tutorial 1][])
 or for Boolean models (see 
[`BoolStateMachine`][cellnition.science.networks_toolbox.boolean_state_machine.BoolStateMachine]
and [Tutorial 2][]).
- Create and export a variety of plots and visualizations, including of the regulatory network
model analytic equations, regulatory network directed graphs, heatmaps of gene expressions in
equilibrium states, gene expressions in temporal simulations, and depictions of the general and event-driven NFSMs
(see [Tutorial 1][] and [Tutorial 2][] for some examples).

## Getting Started


## About

Cellnition is portably implemented in [Python](),
continuously stress-tested via [GitHub Actions]() **×**
[tox]() **×** [pytest]()  **×** [Codecov](), and [licensed][license] under
a non-commercial use, open source [APACHE license][] with Tufts Open Source License Rider v.1.
For maintainability, cellnition officially supports *only* the most recently released
version of [CPython]().

## Acknowledgements 

*Cellnition* creators are grateful for collaboration opportunities and funding support
from the [Levin Lab]()
at [Tufts University](), with funds for this work supplied from the [Templeton World Charity
Foundation grant TWCF0606](https://www.templetonworldcharity.org/projects-resources/project-database/0606) 
and a sponsored research agreement from [Astonishing Labs](https://astonishinglabs.com/).

## License

*Cellnition* is non-commerical use open source software [licensed][license] under an
[Apache 2.0 license][APACHE license] with Tufts Open Source License
Rider v.1, restricting use to academic purposes only.

[Link References]::
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