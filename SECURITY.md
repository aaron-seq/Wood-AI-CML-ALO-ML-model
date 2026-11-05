# Security policy

## Supported versions

Only the `main` branch is maintained. There are no versioned releases, so
fixes land on `main` and are not backported.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability, and do not include
details in commits, branch names or pull requests before a fix is available.

Report it privately through GitHub: on the repository's **Security** tab,
choose **Report a vulnerability**. If that option is not shown, private
reporting is not enabled; open an issue titled "Security contact request",
with no details, and the maintainer will arrange a private channel.

Include the affected commit, the endpoint or file involved, the steps to
reproduce with synthetic data, and the impact you observed. Please do not
test against deployments you do not operate.

This is a volunteer-maintained project. There is no bug bounty and no
guaranteed response time.

## Scope and deployment responsibility

The maintainers do not operate a hosted instance. Whoever deploys the service
is responsible for its exposure, and the defaults assume a trusted network:

- Authentication is **off** unless `API_KEY` is set, and even then it is a
  shared secret, not an identity ([API reference](docs/API_DOCUMENTATION.md#authentication)).
- `docker compose` publishes on 127.0.0.1 only for that reason.
- Request bodies must be capped at a gateway; the application bounds what it
  reads and parses, but the framework receives the whole body first.

The [production checklist](docs/DEPLOYMENT.md#production-checklist) lists what
to configure before exposing the service to users.

## What the project checks

CI fails on a known CVE in any shipped dependency (`pip-audit`), and runs the
test suite, which includes regression tests for every security fix, on Python
3.11–3.14. Neither is a substitute for a security review of your deployment.
