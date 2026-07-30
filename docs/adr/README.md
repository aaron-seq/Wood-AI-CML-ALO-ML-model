# Architecture Decision Records

One file per decision that would otherwise get re-litigated or silently
reversed. Each states the situation, what was decided, what was rejected and
what it costs.

These follow a plain lightweight format rather than a story-linked template:
the repository has no issue tracker with story IDs to reference, and inventing
identifiers would make the records look traceable when they are not. Where a
decision came out of a specific GitHub issue, that issue is linked.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-single-canonical-api-entry-point.md) | One API module, not four parallel copies | Accepted |
| [0002](0002-pin-dependencies-to-the-model-artifact.md) | Pin dependencies to the model artifact's scikit-learn | Accepted |
| [0003](0003-share-feature-engineering-between-training-and-serving.md) | Share feature engineering across training and serving | Accepted |
| [0004](0004-centralise-upload-validation.md) | Centralise upload validation; client errors are 400s | Accepted |
