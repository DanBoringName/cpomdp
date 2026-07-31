# Warrant ledger

### What cpomdp can show, prove, and falsify — and what it cannot, ever

*Companion to the falsification battery. This document fixes the epistemic scope of the instrument before any test is designed against it. The rule it enforces: no test may claim more than its prover licenses. Written against cpomdp v0.4.2 as built, with v0.4.3–v0.5 capability marked as gated rather than available.*

*Vocabulary warning, because two "tier" scales collide in this programme. **Certification tiers A/B/C** classify how well a number is known. **Register tiers 1/2** classify deferred work by whether it ships a feature. They are unrelated. This document writes "Tier A/B/C" and "register Tier 1/2" and never abbreviates either.*

---

## 1. Three provers, three different licences

Everything the programme asserts is established by one of three mechanisms. They are not interchangeable, and most overclaiming in this literature comes from silently swapping one for another.

**Prover 1 — pen-and-paper theorem.** Establishes universal statements within stated hypotheses: Paper 1's Theorem 1, Lemmas 1 and 2, Corollaries 1 and 2. The only prover that establishes *for all models in class M, for all policies, for all observation sequences*. Its scope is exactly its hypotheses (H1, H2, H3/H3′) and not an inch further. Where the hypotheses are unverified for a model, the theorem says nothing about it.

**Prover 2 — symbolic computation.** Establishes closed-form identities and algebraic non-existence results, with a CAS as checker. Part 2's degree count sits here. Warrant is theorem-grade provided the symbolic reduction is faithful to the analytic claim, which is a human obligation the CAS does not discharge.

**Prover 3 — numerical execution.** Three sub-modes with genuinely different licences, and conflating them is the most common error in this area.

*3a — Floating-point sampling of a continuum.* Sweeping actions on a grid, running seeds, scanning parameters. Can establish an existence claim by exhibiting one construction. Can refute a universal by one counterexample. **Can never prove a universal or a negative existential over the continuum**, no matter how many samples. This is a fact about quantifiers over infinite domains, not about induction.

*3b — Exhaustive enumeration over a finite domain.* Where the quantified set is finite and fully enumerated, execution *decides* the universal, since ¬∃ ≡ ∀¬ and the check is complete. This is live here: policy enumeration over a small action set at H ≤ 3 is a finite set, so "no policy in the enumerated set flips sign" is **proved**, not corroborated. Note the contrast with 3a: an action *sweep* over a continuous range is a finite grid over an infinite domain and only corroborates. The distinction is whether the quantified domain is finite, not whether the computation is.

*3c — Validated numerics.* Interval arithmetic and certified error bounds prove universals over compact domains by construction. This is standard computer-assisted-proof machinery, and it is exactly what the certified discretisation bound is: a device licensing *for all x in the domain, |p_grid − p_exact| ≤ δ*. Any claim resting on that bound inherits proof-grade warrant over the domain, not sampling-grade warrant.

So the honest form of the prohibition is narrower than "execution cannot prove universals": **floating-point sampling of a continuum cannot.** Modes 3b and 3c can, which is why the certified-bound work is worth its cost. It converts Tier B claims from corroboration into proof over a compact domain.

**The canonical illustration, already in Paper 1.** "No fixed linear-Gaussian filter reproduces the agent" is a negative existential over an infinite family of candidate filters, and mode 3a cannot touch it. What the code establishes is weaker and logically different: cpomdp's own flattening route refuses this specification, as a typed, data-free, structural check firing before any filtering runs. Paper 1 §5 splits it explicitly — the library detects that two linearisation points diverge, the theorem proves nothing closes the gap, and the generality "belongs to the theorem alone." Shorthand: **the theorem proves, the code witnesses.**

**And the theorem's own ceiling, which both this ledger and the battery must carry.** Theorem 1 is finite-dimensional (Paper 1's Limitation 4). It rules out fixed linear-Gaussian reductions of a finite-dimensional model. Whether some infinite-dimensional filtering representation reproduces the agent is left open by the theorem and deliberately by the demonstration code. The universal is bounded, and citing Theorem 1(ii) without that qualifier overstates it.

---

## 2. Certification tiers

Tiers classify *how well a number is known*. They cut across the prover question; both must be stated to fix a claim's status.

| Tier | Meaning | Established by | Licenses | Availability |
| --- | --- | --- | --- | --- |
| **A** | Closed-form, provably optimal reference; every distribution Gaussian under a fixed covariance recursion | Provers 1–2, checked numerically at machine precision | Exact identities; agreement of two independently computed closed forms | **v0.4.2** |
| **B** | Numerically certified: a stated error bar or a certified bracket | Prover 3, with 3c where a bound is proved | Interval claims; separations exceeding the bound; comparisons whose difference exceeds combined bars | **Partly now, partly gated** — see below |
| **C** | Computed, no statable error bar at feasible cost | Prover 3a alone | Suggestion, exploration, figures. Never certification; the word is "computed" | Available, disclosed as such |

**Tier B is not one thing, and the v0.4.4 gate does not bind all of it.** Two populations:

*Tier B available now (v0.4.2).* Claims computed inside the agent's own closed-form Gaussian recursion, which touch no grid: Theorem 1(i)'s covariance separation, Corollary 1's scalar epistemic separation, Example 1's closed-form level set, the typed structural guard. The programme already treats fixed-R and R(x) as its two *exhibited* model classes, with the deferred Hierarchical Gaussian Filter offered as "a second Tier B witness", which presupposes a first. Battery tests B1–B5 sit here legitimately at v0.4.2.

*Tier B gated on v0.4.4.* Any number routed through the exact reference filter, which is where the certified discretisation bound lives: the absolute inference gap under R(x), the fidelity ladder, the scaling exponent. In battery terms C6, D1, D2, and the reference-dependent parts of D3.

The gate's failure clause is likewise scoped: if the bound is not statable at the pre-agreed factor, *Part 2's numbers* become uncertified and the paper is a different paper. That is existential for Part 2 and correctly gated at v0.4.4 rather than discovered at drafting. It is not a claim that the tier as a category is empty today.

**Tier C's two causes** — grid cost exponential in latent dimension, and genuine infinite-dimensionality — behave differently but **both admit promotion routes**, contrary to the intuition that structural limits are permanent. The first retreats with better quadrature. The second retreats by *certificate*: the deferred delay-differential model class is Tier C by genuine infinite-dimensionality, with a finite truncation plus a truncation certificate landing it in Tier B. That is the register's own example and it is the only place the third certificate kind gets exhibited rather than asserted.

---

## 3. What claim shapes are available

Read a proposed result's logical form; this says whether the tooling delivers it.

| Claim shape | Example | Provable? | Refutable? | Notes |
| --- | --- | --- | --- | --- |
| **Existence (strict inequality)** | "There is a model, Gaussian in every conditional, whose epistemic value is action-dependent" | **Yes, outright** | No, never (its negation is a universal) | One construction settles it permanently. Robust to floating point because the claim is an inequality |
| **Existence requiring a matched equality** | "Two agents the one-term accounting scores identically and the three-term one separates" | **Only as a Tier B interval claim** | No, never | The matching conjunct is an *equality*, satisfiable only to tolerance. Needs certified bars exactly as an ordering does. This is C5, and it is not free |
| **Universal within a class** | "Every fixed-R, additive-control LG model has constant epistemic value" | Via Prover 1; or Prover 3b if the domain is finite and enumerated | **Yes, by one counterexample** | Paper 1 eq. (4) supplies the proof. Continuum sweeps only corroborate |
| **Negative existential** | "No fixed LG filter reproduces the agent" | **Not by sampling.** By Prover 1 within its hypotheses, and note the finite-dimensional ceiling | **Yes, by exhibiting one** | Theorem 1(ii) proves it; the code witnesses one route failing |
| **Comparative / ordering** | "The smoothed rule's gap is below the plug-in rule's" | As a Tier B interval claim | Yes, if the difference exceeds the error on the difference (§4) | Ties are a third outcome, neither confirmation nor refutation |
| **Quantitative magnitude** | "gap ∝ curvature × spread²" | Corroborable to a stated interval; proved only over a compact domain via 3c | Yes, if the fit lands outside the registered interval | Needs Tier B *and* a demonstrably non-empty fit window (§5) |
| **Structural / type-level** | "This specification admits no flattening by this route" | **Yes, outright and data-free** | Yes | Fires on the declared model before any data. The closest thing here to a compile-time proof |
| **About a real organism** | "E. coli minimises free energy" | **No** | **No** | Identifiability wall (§7) |

**The design heuristic, with its cost attached.** Existence claims are what the instrument settles outright, so phrase results that way where the science permits. But note the trade-off rather than burying it: settled-ness and severity are anti-correlated here. The battery deliberately weights pre-registered magnitudes above sign checks and sign checks above existence demonstrations, because a claim that is easy to establish is usually one a false theory would also satisfy. A drafter optimising only for warrant produces well-certified, low-severity papers. Both criteria have to be stated in the same breath.

---

## 4. What "zero to machine precision" actually licenses

The calibration cross rests on readings like "below 1e-12", and the credit-assignment claim is built on them.

A term reading below 1e-12 licenses **"this term is smaller than 1e-12"** and never "this term is zero". For the off-diagonal cells that suffices, because the claim is a *contrast*: one term moves by O(1) while the other stays below tolerance. What carries the argument is the **ratio**, roughly twelve orders, not the small number alone.

The reported quantities are KL divergences in nats, chosen precisely because they are reparameterisation-invariant, so the scale question is not one of arbitrary units. It is still live in a different form, and three conditions must be printed alongside any separation assertion:

**The moving term's magnitude, beside the pinned one.** "Term B is below 1e-12" is empty if term B is naturally O(1e-13) in this configuration. The assertion is the ratio; a cell reporting only the small number is not evidence.

**Condition numbers.** These are matrix inversions and log-determinants. An ill-conditioned Σ or S can manufacture or destroy twelve orders, and the conditioning belongs in the printed output for any cell whose claim is a separation.

**Absence of catastrophic cancellation.** A term computed as a difference of large quantities can read small while hiding structure.

**On the H(p\*) rule, stated with the mechanism the sources actually give.** "Never obtain a term by subtracting H(p\*)" is an *avoid-estimation and preserve-invariance* rule: under a common exogenous action sequence H(p\*) is a shared constant that cancels, leaving two KL divergences that are reparameterisation-invariant and computable without any entropy estimate, with the evaluator returning both divergences directly. Under that path H(p\*) is never computed at all. Worth noting the one exception, because it is where the rule's cost lands: the additivity check does require H(p\*) to be estimated, which is why its residual carries an entropy bar.

**Error-bar propagation, corrected.** The additivity residual is `E[F]_measured − (H(p*) + D₂ + D₃)`, and the measured E[F] is itself an expectation under p\* carrying its own bar. The bound is **δ_F + δ_H + δ₂ + δ₃**, not δ_H + δ₂ + δ₃. Omitting δ_F under-bounds the residual and would let a real failure of closure read as within tolerance.

**Comparisons need the error on the difference, which is not always the sum of the bars.** The conservative rule is that two quantities whose bars overlap are a tie. But where several quantities are scored against the *same* reference filter, its discretisation error is **common-mode and largely cancels in their differences**, so the error on a difference can be far smaller than δᵢ + δⱼ. This matters directly: the fidelity ladder's rungs are all scored against one reference, so its real resolving power lives in that correlation structure. Using the sum-of-bars rule there is safe but understates the test. The correct protocol is to propagate the error on the *difference*, exploiting the shared reference, and to pre-register that quantity as the resolution threshold.

---

## 5. Two places the quantitative legs need re-scoping

Both findings are already folded into the battery's D1 and D2 entries. Recorded here with the reasoning, since the reasoning is what generalises.

**The fidelity ladder needs a third outcome.** An ordering is reportable only when the difference between adjacent rungs exceeds the error on that difference (computed with common-mode cancellation, §4, not the sum of the bars). The source registers R7 as "reported whichever way it comes out", which is binary and has no home for a tie. Overlapping intervals must be pre-committed to **"not resolved"**, which is neither confirmation nor refutation. Without Tier B on the reference filter, the ladder yields four computed numbers in an order, and an order among uncertified numbers is not a result in either direction.

**The scaling exponent may have an empty measurement window, in which case it is not a test.** The fit needs a regime where the second-order term dominates and where the gap is large relative to the certified bound. Those squeeze from opposite ends: large spread contaminates the exponent with higher-order terms, small spread drives the gap toward zero where relative error against the bound diverges. Roughly, with gap ∝ curvature × spread², the lower edge sits near √(k·δ_ref/curvature) and the upper near √(c₂/c₄), so the window is non-empty iff δ_ref/curvature ≪ c₂/c₄.

Two corrections to how this was first stated. It is **not** true that only re-scoping helps: the window's lower edge is set by δ_ref, so a tighter certified bound (adaptive quadrature) widens it directly. Window existence is partly a function of code. And the analysis must run on **both** swept factors, not just spread — the curvature axis has its own ceiling, because the belief-smoothed rung requires E[R(x)] under the Gaussian prediction to exist, and H1 puts no ceiling on how fast R may grow, so at high curvature a rung can drop off the ladder entirely. That is a second, independent way both D1 and D2 can fail to be tests.

The prerequisite deliverable stands: demonstrate a non-empty window, print its bounds and the signal-to-bound ratio across it, before any exponent is fitted.

---

## 6. What can be pinned, and what cannot (the Duhem–Quine ledger)

No test refutes a lone proposition. Every measurement confronts a conjunction, and the instrument's value is how many conjuncts it holds fixed while one varies.

**Pinned by construction:** the world process p\*, because the world is built rather than inferred; the agent's model p, separately, with a type seam making circularity impossible rather than discouraged; preferences, by fiat; the exogenous action sequence, driven identically into every agent under comparison; the evaluation rule, swappable through the rule-family interface; inference quality, degradable through the frozen-gain, wrong-fixed-R and diagonal-covariance variants; the random seed stream, matched across arms, which is what makes the off-diagonal attribution stable rather than a seed artefact; and the planning horizon, fixed inside the instrument.

That is an unusually long list, and it is the programme's real methodological advantage. When a defect appears with p = p\* asserted at machine precision, it cannot be attributed to misspecification. Most of this apparatus arrives with the v0.4.3 scoring harness and the v0.4.4 rule family, not at v0.4.2.

**Not pinnable, external:** whether the model class fits any natural phenomenon; whether an organism's generative model resembles p; whether attributed preferences are the ones held; whether the horizon used matches a real agent's.

**Not pinnable, internal, and this one is easy to miss:** whether the three-term decomposition is *complete*. The additivity check's own falsifier is three-way ambiguous between an incomplete decomposition, an undiscovered fourth term, and a mis-estimated term. No amount of pinning inside the instrument resolves that, because the instrument's dimensionality is the thing in question.

**A pinned conjunct that costs something.** Driving a common exogenous action sequence is what makes H(p\*) cancel, and it is not free: it severs the control loop. Under R(x) an agent choosing its own actions steers toward low-noise regions and thereby changes its own inference gap, so a gap measured under an imposed sequence can systematically misrepresent the closed-loop gap. Paper 1 meets the same seam when it asserts its dissociation extension on the objective rather than on the path. Exogenous action belongs on both lists: pinned, and a contestable modelling choice.

**The practical consequence.** Every refutation this programme issues has the form: *this functional, on this model class, with these preferences, this evaluation rule, this horizon, and this exogenous action sequence, makes prediction X, and X is false.* That is a real refutation. It is not a refutation of active inference, and a defender who swaps a conjunct has not cheated. Progressive versus degenerating is judged over a series, never at one.

---

## 7. The permanent boundary

Distinguish capability gaps that a release closes from logical limits no release touches.

**Capability gaps, closing on the roadmap.** Multi-step expected free energy beyond horizon 1, the largest single item, today blocking the crossover, the horizon>1 control leg, and any accumulated-epistemic-value claim; it forces per-branch covariance trajectories because the planning covariance is policy-dependent by Theorem 1(i). The exact reference filter, blocking absolute inference-gap measurement **under R(x)** — not in fixed-R, where the Kalman filter is the exact Bayesian filter and the gap is closed-form. The certified bound that unlocks the gated half of Tier B. Then register Tier 1: parameter estimation, point-process emissions, rate-distortion, directed information.

**Scope limits, structural for now.** Grid-routed claims are confined to roughly one or two latent dimensions, since the reference filter is low-dimensional and uniform; this does **not** confine closed-form Tier B claims, which is why the two-node coupled-tree T-maze is fine at v0.4.2. Policy enumeration caps at H ≤ 3 with a small action set. The theorem's model class is a single chain while demonstrations run coupled trees, so tree results are exhibits rather than instances of Theorem 1 as stated. Three separate open items that should not be welded together: vector-case *positivity* for Part 2 needs its own rank hypothesis, since C injective (full column rank, n_o ≥ n) is **not** Paper 1's H2 (full row rank, n_o ≤ n) and they coincide only for square invertible C; H5's sufficient conditions are open under a time-varying gain schedule; and the title-strength claim requires H3′ rather than H3, where determinant-visibility is generic but generic is not universal.

**Two qualifications on what the existing results mean, both currently missing from the battery.** Theorem 1 is finite-dimensional, so the no-flattening universal is bounded and the infinite-dimensional question stays open. And the dual effect exhibited is a property of the *agent's maintained Gaussian recursion*, the covariance it propagates; whether the exact non-Gaussian conditional covariance of the generative model carries a dual effect in the strict Bar-Shalom–Tse sense is a separate open question. That second one bounds what the dual-effect test buys and what the dual-control bridge may claim, and it should be stated wherever either appears.

**Logical limits, permanent.** Floating-point sampling of a continuum proves no universal and no negative existential; only Provers 1 and 2, and modes 3b and 3c within their domains, do. No behavioural measurement identifies whether a system minimises free energy, since the generative model and the preferences are both free and any behaviour is free-energy-minimising for some pair; the one unexploited lever is multi-environment identifiability, where a model shared across environments is more constrained than one fitted per environment. No in-silico result transfers to an organism by measurement, only by structural correspondence, which attributes nothing to the organism. And nothing here bears on the FEP's core, now or after every gate passes.

---

## 8. Standing rules, restated as warrant rules

The three standing rules are the right ones; stating their mechanism makes them harder to erode under drafting pressure.

*No figure quoted in prose unless printed and asserted in the check suite.* This keeps Prover 3a's output from silently acquiring Prover 1's authority. A number in prose reads as established; a number in an asserted check is established at exactly its tier.

*No threshold discovered by raising a parameter until the result fires.* Post-hoc thresholds have no severity, since a false hypothesis passes them at the same rate as a true one. Registering the crossover early and requiring the sign flip at the registered point and one below is what makes it a test rather than a curve-fit.

*Pre-register the diagnosis, not the conclusion.* Stated symmetrically in the source: say in advance what a mismatch would first be attributed to *and how it should behave*, on the expectation that a bug is the most likely cause. The complement is worth adding: state in advance what a null result gets reported as, since a test whose null outcome has no registered home tends to get quietly re-run until it speaks.

---

## 9. What this means for Family G

Family G asks which free-energy functional an agent is running. Reading that through this ledger fixes its shape before any test is written.

**First, disambiguate what "coincide" means, because the answer differs.** Two functionals can agree *in prescribed action* while differing *in value*, and Family G must say which it discriminates on. Value differences are visible wherever the functionals differ at all; action differences are what behaviour reveals. A constant offset changes the value and not the argmin.

**The fixed-R trap, stated precisely.** In the fixed-R additive-control linear-Gaussian regime the epistemic term is constant across policies — the predicted mean is absent from the closed form — and Koudahl et al. harden this: for the full functional, parts of the instrumental and epistemic terms cancel exactly, leaving KL control plus an additive constant regardless of how the objective is cut. A per-step policy-independent constant sums to a policy-independent constant, so the argument survives horizon summation. Consequently any two functionals differing **only in their epistemic term** are argmin-identical in fixed-R, including λ-weighted information bonuses, since λ·constant is still constant. A discrimination test run only there will find spurious behavioural agreement.

**But the trap is narrower than "build everything in R(x)".** It bites only on epistemic-term discriminations. Functionals differing in the **risk or preference convention** stay discriminable in fixed-R at v0.4.2, and Paper 1's Appendix A names exactly that family: definitions of expected free energy differ in which factor of the joint is replaced by the preference prior, and the variants coexist in the literature. So Family G splits cleanly by target: preference-convention discriminations run now on Tier A closed forms; epistemic-term discriminations require R(x), and their quantitative half sits behind the v0.4.4 gate.

**Best-warranted, and available now.** Where two fully specified functionals give different *values* on the same pinned Tier A setup, that difference is exact arithmetic rather than estimation. This is the strongest warrant in the programme.

**Available as an existence claim, which is the right form.** "There exists a model class on which F₁ and F₂ prescribe different actions" is settled outright by one construction. Its negation is **not** equally cheap: "on this class the two are observationally equivalent" is a universal over the class, needing Prover 1, or Prover 3b if the policy set is finite and exhaustively enumerated — which at H ≤ 3 with a small action set it *is*, so equivalence on an enumerated policy set is provable by execution. Equivalence over a continuum of models or parameters is not. That asymmetry decides how each Family G result must be phrased.

**Not available.** Any claim that a real agent runs one functional rather than another. Any universal that two functionals differ in general across a continuous model class, absent a theorem. Any quantitative discrimination routed through the exact reference filter, until v0.4.4.

---

## 10. The crossover pre-registration (R10)

R10 — the multi-step horizon H\* at which the epistemic pull overtakes the pragmatic gradient — is the programme's headline output, and standing rule 2 forbids discovering it by extending H until the flip appears. So the statistic, its sign convention, the reach/walk pair, and the H=1 anchors are all fixed *before* the H-sweep harness measures H\*. The witness lives in cpomdp: `cpomdp.crossover` (the statistic and the `crossover_horizon` definition of H\*), `tests/test_crossover.py` (the H=1 reduction and the sign convention, asserted), and the anchor numbers in cpomdp's `warrant_numbers.md`. cpomdp does not know these are R10's pre-registration; that framing is the programme's, and it is recorded here rather than in the library.

**Pre-registration commit.** A reader verifies the order from the cpomdp commit that introduced `cpomdp/crossover.py` and `tests/test_crossover.py`, which predates any sweep-harness commit.

**Crossover pre-registration commit:** `38df72deb57f8f2417b69fdb0d5acee4bbbbf91a`

**The registered falsifiers (D3).** The crossover is falsified, not confirmed, by any of these. The testable ones are asserted at H=1 in cpomdp now; the rest bind the sweep harness when it lands:

- *No crossover at any feasible H.* `crossover_horizon` returns `None` rather than a number, so a miss stays visible instead of being laundered into an H\*.
- *The flip is not a clean sign change at H\* and H\*−1.* Checked at the sweep against the measured H\*.
- *Not reproducible across seeds.* The anchors carry no observation draw and are asserted bit-identical on recompute; the closed-loop reproducibility binds the sweep.
- *H\* not stable under a declared refinement of the action set.* The reach/walk identification is asserted stable from the fine grid to the declared `crossover-v1` set now; full H\*-stability binds the sweep.

The anchor magnitudes (pull 1.72, gradient 4.49, ΔG +2.77 nats node-restricted; 2.42 whole-state) and the tolerance `ANCHOR_TOL = 1e-4` live as *numbers* in cpomdp's `warrant_numbers.md`. This section records only what makes them pre-registered *evidence*.
