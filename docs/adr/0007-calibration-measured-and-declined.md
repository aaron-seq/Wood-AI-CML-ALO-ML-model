# ADR-0007: Calibration is implemented, measured, and left off

**Status:** Accepted · **Date:** 2026-07-31

## Context

`elimination_probability` is a Random Forest vote share. Documentation and
the model card both say it must not be read as a probability, and
"calibrate it" had been on the future-work list since the first audit —
the obvious fix, and the one a reviewer would expect.

So it was implemented and then measured on the 500-row dataset, holding
everything else fixed:

| Method | ROC-AUC | F1 | Brier ↓ | Log loss ↓ | ECE ↓ |
| --- | --- | --- | --- | --- | --- |
| uncalibrated | 0.692 | **0.529** | 0.2179 | 0.6287 | 0.0690 |
| sigmoid | 0.691 | 0.476 | **0.2144** | **0.6193** | **0.0595** |
| isotonic | 0.689 | 0.485 | 0.2241 | 0.6479 | 0.1263 |

Sigmoid improves Brier by 1.6% relative and costs 10% of F1. Isotonic is
worse on every measure — the expected overfit on a 100-row test set. None
of these gaps exceeds the noise at that sample size.

## Decision

Ship the capability, not the change: `CalibratedClassifierCV` is wired into
the trainer behind `make train CALIBRATE=sigmoid|isotonic`, and the served
artifact stays uncalibrated.

Every training run now reports Brier score and expected calibration error
whether or not calibration is enabled, so "the probabilities are
uncalibrated" becomes a measured claim instead of an assumption, and the
decision can be revisited against numbers rather than intuition.

## Alternatives considered

**Enable sigmoid calibration by default**, since it does improve the
calibration metrics. Rejected: it costs 10% of F1 — the metric the model
is actually selected on — to buy an improvement well inside the noise. That
is a worse model shipped for a nicer-sounding property.

**Do not implement it at all**, given the measurement. Rejected: the
measurement is the deliverable. Without the code there is nothing to
re-run when the dataset changes, and the next person would have to
rediscover the same result.

**Report the metrics but skip the trainer flag.** Half the work, and it
leaves the obvious follow-up ("fine, now calibrate it") still to do.

## Consequences

The claim that probabilities are uncalibrated is now backed by numbers in
the model card, and turning calibration on is one flag rather than a
project.

Nothing about the served model changes, so this ADR records a decision
*not* to act — which is the kind most likely to be quietly reversed by
someone who reads "uncalibrated" as an oversight rather than a finding.

The underlying problem is unchanged and is not a calibration problem:
at ROC-AUC 0.69 the ranking is weak, and calibration cannot add information
that is not there. Validating against real inspection outcomes remains the
prerequisite, and this should be re-measured at that point.
