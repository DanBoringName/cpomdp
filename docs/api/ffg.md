# Factor graph models

A `CouplingGraph` declares a rooted tree of Gaussian-coupled variables. It holds the shape a chain cannot: a node with three or more neighbours, inferred through the ones around it. `Coupling` edges carry the within-slice drive `child = W·parent + noise`, and the factor types below supply each node's dynamics and its observation. Run one through a [`CouplingGraphBackend`](backends.md).

::: cpomdp.CouplingGraph

::: cpomdp.Coupling

::: cpomdp.GaussianCoupling

::: cpomdp.GaussianTransition

::: cpomdp.GaussianObservation

::: cpomdp.CallableGaussianObservation
