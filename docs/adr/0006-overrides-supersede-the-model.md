# ADR-0006: Expert overrides supersede the model in the response

**Status:** Accepted · **Date:** 2026-07-31

## Context

`SMEOverrideManager.apply_overrides_to_predictions` existed, was tested,
and was called by nothing. A grep found references only inside its own
module and its own test file. `/score-cml-data` never consulted the store.

So an engineer could record "KEEP this CML — high-consequence area next to
a fired heater", the decision would be persisted and shown on the SME
Overrides page, and the next scoring run would still return ELIMINATE. The
documentation described the override system as the safety mechanism that
keeps a human in the loop. It was a logbook.

## Decision

Scoring reads the override store and folds the decisions into the
response. The contested question was which field carries the final answer.

`recommendation` — the obvious field, the one an existing client already
reads — now carries **the decision to act on**: the override where one
exists, otherwise the model's output. `model_recommendation` is added to
preserve what the model said on its own, and `sme_override` carries who
overruled it, when and why.

This changes the meaning of an existing field, which is normally the wrong
move. It is the right one here: the alternative leaves the default field
showing a recommendation an engineer has explicitly overruled, and the
failure mode of that is a CML being retired against expert judgement. A
client that reads only `recommendation` should get the safe answer.

Nothing is lost. `predicted_elimination_flag` and
`elimination_probability` still report the raw model output, so the audit
trail is complete — you can always see what the model thought and what the
human decided.

`/generate-report` applies overrides before aggregating, so its totals
describe what will be acted on.

## Alternatives considered

**Add `final_decision` and leave `recommendation` as the model output.**
Non-breaking, and rejected for exactly that reason: every existing client
keeps acting on un-overridden recommendations until it is updated, and
nothing makes that failure visible.

**Return both and let the caller decide.** That is what this does — the
disagreement was only about which one gets the unqualified name.

**Apply overrides at the storage layer**, so the model's prediction is
overwritten. Rejected: it destroys the audit trail, and the agreement rate
between experts and the model is a signal worth keeping.

## Consequences

The human-in-the-loop is now load-bearing rather than decorative, and
`sme_overrides_applied` makes it visible how often the model is being
overruled — a number worth watching.

A client that was reading `recommendation` will start seeing overridden
decisions. That is the intended behaviour change and it is called out in
the API reference.

Scoring now reads the override file on every request. It is small and read
whole, so this is one extra file read per request; if the store grows or
moves to a database, that read should be cached with an invalidation hook.
