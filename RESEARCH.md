# My research ramblings as I build a continuous Passive Observer Markov Decision Process modelling toolbox

## Problem 1 -> Should I enforce positive semi-definiteness (PSD)

Positive covariance, which is a must as a belief has no negative context, using an eigenvalue evaluation to determine positivity is an O(n^3) operation. This is extremely expensive and I would like to avoid it if possible.

**Pymdp** operates in discrete space. No Gaussians. This means vector probabilities fit the **simplex constraint**

### Solution?
Move the expensive linear algebra to agent construction. Keep the inference loop free of any O(n³) operation.

The key discipline, stated once and enforced everywhere:

>Front-load the structure of the computation, never the values.

The posterior mean and covariance depend on incoming observations — that is the inference, it runs every loop. What does not depend on the data is (a) the conditional-independence structure of the model and (b), for a linear time-invariant Gaussian model, the entire covariance/gain sequence. Both are fixed by the model at construction and can be solved once.

For the linear-Gaussian case this collapses to: solve the discrete algebraic Riccati equation (DARE) once at build time, store the steady-state gain K∞, and run a fixed-gain filter in the loop. No covariance update, no inversion, ever, in the hot path.

This is based off of a steady-state Kalman filter. I am choosing to expose it as the default architecture rather than re-deriving the Kalman gain every step.
