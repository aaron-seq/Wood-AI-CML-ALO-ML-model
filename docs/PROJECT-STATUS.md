# Project status and continuation

Last verified: 2026-10-25 (UTC), at the head of PR #19.
Treat this as a starting point to re-verify, not as truth: check live Git and
GitHub state before acting on it.

| | |
| --- | --- |
| Repository | <https://github.com/aaron-seq/Wood-AI-CML-ALO-ML-model> |
| Active PR | [#19](https://github.com/aaron-seq/Wood-AI-CML-ALO-ML-model/pull/19) — **draft**, not merged |
| Branch | `claude/repo-modernization-audit-gq5x7n`, based on `main` at `cdf8ccf` (merge of #14) |
| Backlog | [Open issues](https://github.com/aaron-seq/Wood-AI-CML-ALO-ML-model/issues), ordered in [DEPLOYMENT.md](DEPLOYMENT.md#future-work) |
| Project board / milestones | None. No tool available to the session that produced this could create them; see *Gaps* |

## Current state

`main` went red after #14 merged without any commit causing it (a setuptools
CVE on the 3.11 runner; a NumPy 2.5 deprecation escalated by the model-load
check). That made all three open Dependabot PRs fail too. PR #19:

- **Restores green CI** and incorporates Dependabot #15 (Python 3.14 image),
  #16 (Actions v7) and #18 (pip group: pandas 3, mypy 2, Streamlit 1.66) as
  cherry-picks with authorship kept, plus the fixes they needed.
- **Fixes a shipped test override** that retired CML-209 on every fresh install.
- **Bounds the cost of an upload** in rows, width, decompressed size and
  shared strings, and parses off the event loop. One case found in retest — a
  4 MB CSV accepted after 187 s and 3.8 GB — now fails in 10 ms.
- **Closes API-surface gaps**: an ungated `/metrics` at scope `all`, unbounded
  and log-forging override fields, exception text in 500s, unbounded `limit`,
  absolute paths in `/health` and `/model/info`, CSV formula injection in
  exports, forged text-log lines, a 500 on a non-ASCII key, a 500 on
  `Infinity`/`NaN` JSON, and a 500 on corrupt workbooks.
- **Adds `POST /predict`** (#10), sharing one scoring path with batch uploads;
  a parity test asserts identical results.
- **Documents** the datasets and how their labels were made (#7),
  configuration (#12), `SECURITY.md`, and the upgrade step that removes the
  shipped CML-209 record from existing volumes.

Issues: #6, #8 and #11 were closed as already resolved on `main`, with
evidence. #7, #9, #10 and #12 close when #19 merges. #5 stays open, superseded
by #20. #20–#27 were created for deferred work.

## Validation

| Check | Where | Result |
| --- | --- | --- |
| ruff lint + format, mypy (15 files) | local, Python 3.14 | clean |
| pytest, 364 tests, 92% coverage (floor 90%) | local, Python 3.14; three random orders | all pass, zero warnings |
| Full suite | local, Python 3.11, 3.12, 3.13, 3.14 (earlier revision, 283 tests) | all pass |
| 22 Hypothesis properties | local, 20 consecutive seeds | all pass |
| pip-audit | local 3.13/3.14; CI on 3.14 | no known vulnerabilities |
| All 9 CI jobs, including the Docker build and its image assertions | GitHub Actions at `0ecec75` | green |

Commands: `make check` (or `ruff check . && ruff format --check . && mypy &&
pytest --cov`); `pip-audit`; for the properties,
`pytest tests/test_properties.py -p randomly --randomly-seed=N --hypothesis-seed=N`.

The Docker image has been verified only in CI; the session had no Docker
daemon.

## Agents used

Delegated reviews in the session that produced #19. They are perspectives
from the same system, not independent human review:

| Role | Scope | Outcome | Verified by the orchestrator |
| --- | --- | --- | --- |
| Issue auditor (read-only) | #5–#12 against `cdf8ccf` | 5 resolved, 3 partial | Spot-checked each residual gap in code |
| Security reviewer (read-only + local probes) | API, uploads, store, Docker, CI | 1 Medium, 5 Low, Informational | Reproduced the Medium; line numbers it cited were wrong and corrected |
| Branch archaeologist (read-only) | `feature/multi-model-ml-support` | Nothing to port; idea tracked in #21 | Read the cited diff |
| Fresh adversarial verifier | All fixes at `966d73f` | Broke the bounds (width, shared strings) and found 4 Lows | Reproduced each with independent probes before fixing; retested after |

## Gaps and decisions needed

- **Owner decisions:** whether to replace the served model (#20); identity
  provider for override attribution (#23); approval for SHA-pinning CI (#25).
- **Not done by the tooling:** no GitHub Project board or milestones were
  created; there was no tool to do so. Labels are limited to the repository
  defaults (`bug`, `enhancement`, `documentation`).
- **Unknown:** whether GitHub private vulnerability reporting is enabled.
  `SECURITY.md` says what to do either way.
- **Branch hygiene:** `feature/multi-model-ml-support` is unmerged and
  superseded; archive it once #21 is decided.

## Resume here

1. Check `git status`, the branch head, and the PR #19 state and CI on its
   current head. A result for an older commit validates only that commit.
2. If #19 has merged, start follow-up work from the latest `main`. Dependabot
   #15, #16 and #18 should then close themselves; close any that do not, as
   superseded by #19.
3. Pick the next issue in the order in [DEPLOYMENT.md](DEPLOYMENT.md#future-work).
   #20 and #23 need an owner's decision first. #26 starts with a load test, and
   #27 is a set of small self-contained items.
4. Run `make check` before pushing.
