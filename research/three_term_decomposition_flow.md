# How a cell gets its two numbers

I'm writing this as the evaluator gets built, one step per commit, so the logic flow
is readable without opening the diffs. The maths contract is
`research/fep_falsification_battery.md` section C. The code is `cpomdp.scoring`.

## The cross comes first

`build_cross` builds every cell of the two declared axes and refuses to hand back fewer
than `|model axis| × |inference axis|`. The certificate is a
`ProductCompletenessCertificate` naming both axes and both versions, so a run over a
different cross cannot be mistaken for this one. This is the Route 2 purchase from the
build plan, and it is why every separation reported later is a comparison between two
cells of one decided set rather than two cells someone happened to run.

## The shape comes next

`Decomposition` holds `misspecification` and `inference_gap`. Nothing else. The floor
`H(p*)` belongs to the process, not to any cell, and prohibition 1 bans reaching a
term by subtracting it. The type has no slot to put an entropy in, and a test pins the
field list, so growing it is a visible decision.
