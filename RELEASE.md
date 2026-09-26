<!--
  ~ Copyright (c) 2023-2026 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

# Making a release

A tag releases `agent-skills` to PyPI, with no stored token: PyPI trusts
`.github/workflows/release.yaml` through OIDC (trusted publishing, GitHub
environment `pypi`). The version lives in `agent_skills/__version__.py`, which
hatch reads too.

## Steps

1. Bump `__version__` on a branch and open a pull request.

1. Merge, then tag the merge commit and push the tag:

   ```bash
   git checkout main && git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

1. The `Release` workflow checks that the tag names the version, builds the
   wheel and the sdist, publishes them unless PyPI already has that version,
   and creates a GitHub release with generated notes.

## Trusted publishing

PyPI project `agent-skills`: owner `datalayer`, repository `agent-skills`, workflow
`release.yaml`, environment `pypi`. The registry matches the repository,
the workflow filename and the environment exactly; renaming any of them
means re-registering.
