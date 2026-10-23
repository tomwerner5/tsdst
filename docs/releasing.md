# Publishing tsdst

Both publishing workflows run when a GitHub release is created. They build the
code selected by that release. Push the approved commits and create a release
from the intended version before publishing. Re-running an old failed job uses
its original commit and workflow revision; it does not pick up later fixes.

## PyPI

The workflow is `.github/workflows/python-publish-tsdst.yml`. It builds the
source distribution and wheel, runs Twine's strict metadata checks, and uploads
with verbose diagnostics.

The upload username is the literal `__token__`. The `PYPI_PASSWORD` repository
secret must contain a production PyPI API token, including its `pypi-` prefix,
with permission to upload `tsdst`. It cannot contain the account password or a
TestPyPI token. The older `PYPI_USERNAME` secret is no longer used.

To inspect the existing secrets without publishing, push
`.github/workflows/check-pypi-secrets.yml` to the default branch. In GitHub,
open **Actions**, select **Check PyPI secret format**, and select **Run workflow**.
It checks whether the old username matches `__token__`, and whether the password
secret is present, has a `pypi-` prefix, and has no surrounding whitespace.
It prints no secret values and performs no upload. Passing this check does not
prove that PyPI accepts the token or that its project scope is correct.

The upload endpoint `https://upload.pypi.org/legacy/` is still correct for Twine.
A progress bar reaching 100% means the file was sent, not that PyPI accepted it.
For a 403, use the detailed response to distinguish invalid authentication from
insufficient project permissions. GitHub does not expose stored secret values,
so the workflow source alone cannot confirm that a secret contains a valid token.

The attempted 2.0.0 upload failed after a successful build. The public PyPI
metadata checked during this investigation still listed 1.0.11, with no files
for 2.0.0 or 2.1.0. This does not identify the rejected credential by itself.

## Conda

Conda remains maintained. The old publishing action used an obsolete Python 3.7
image and passed Python 3.6, 3.7, and 3.8 together to conda-build. The resulting
Python-version list reached code that expected a scalar, causing the reported
`TypeError: unhashable type: 'list'` before the package build started.

The replacement `.github/workflows/python-package-conda.yml` uses current
Miniforge and conda-build, with one Python 3.12 variant setting. The recipe
requires Python 3.12 or later. It produces a `noarch: python` package because
tsdst ships Python source; compiled dependencies remain separate conda packages.

The workflow builds and import-tests the package before authentication. It
retains the existing `ANACONDA_USERNAME` and `ANACONDA_PASSWORD` secrets, as
requested. It uses `anaconda org login` with the supported legacy login mode
for these credentials, then `anaconda org upload`. Current clients also have
a different top-level `anaconda login` command, so the `org` prefix matters.

The local validation used conda-build 26.9.1. It built
`tsdst-2.1.0-py_0.conda` and passed imports of `tsdst`, `tsdst.eda`, and `tsdst.nn`
in the conda test environment. Automatic uploads were disabled. No login or
upload was performed during validation.

## References

- [PyPI token setup](https://docs.pypi.org/publishing/publishing-to-pypi/#using-an-api-token)
- [PyPI upload troubleshooting](https://docs.pypi.org/publishing/troubleshooting/)
- [GitHub workflow reruns](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)
- [Conda build variants](https://docs.conda.io/projects/conda-build/en/stable/resources/variants.html)
- [Anaconda client releases](https://github.com/anaconda/anaconda-client/releases)
