# Public release preparation

The public repository is [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper).
The public name is Jumper. The Python distribution remains `mjrl-lab`, its imports remain
`mjrl`, and configuration keys remain `MJRL_*` for compatibility.

## Initial publication

Prepare the first public commit from a reviewed tracked-file snapshot, without the internal
Git history. Keep the internal repository and its branches intact. Do not mirror all internal
branches or tags into the public repository.

Design is an independent repository linked from the README and agent instructions.
The public snapshot contains no Design gitlink or vendored checkout. Only the documented
showcase thumbnails are copied here; preserve their source attribution.

The internal history contains Rockchip SDK files removed from the current tree. Their
redistribution terms need separate review; the current build fetches them from upstream
under their own licence. See [the runtime provenance](../deploy/fsm/vendor/rknpu2/README.md).
A clean snapshot avoids publishing those removed files or historical internal configuration.
It does not replace a review of the files retained in that snapshot.

Before pushing the prepared public `main`:

- Confirm publication rights for Jumper mechanical assets, pretrained weights, reference
  motion clips, face animation, and README artwork. Imported motion provenance is recorded
  by [the dance importer](../tools/import_wbc_dances.py) and
  [the gesture importer](../tools/import_wbc_gestures.py); provenance alone is not a licence.
- Review tracked configuration and generated bundle contents, including binary archives,
  for information unsuitable for publication. `.env` contains shared runtime defaults;
  `.env.local` must remain excluded.
- Review `LICENSE`, `NOTICE`, and third-party notices against the materials being shipped.
- Choose the public commit author and email explicitly. Workstation-derived identities
  should not become the accidental identity of the first public release.
- Check that the destination remains empty; inspect any new remote commits before pushing.

## Example assets

The initial candidate retains the committed example policies and bundles. Deployment
manifests and several tests read them directly, so deleting them would break existing paths.
Existing build records retain their original source identities and commit IDs; do not edit
those records to pretend the binaries were rebuilt from the new public repository.
New exports identify `KingKongRobotics/jumper` as their source.

Moving downloadable bundles and runtimes to GitHub Releases is follow-up work. That change
must include versioned download URLs, checksums, an explicit download command, and updates to
the affected documentation and tests. No download URL is advertised before an asset exists.
If reference clips or weights move to LFS, add `.gitattributes` and document the extra clone
step in the same change. Keep README images directly available to repository viewers.

## Checks and repository settings

The GitHub workflow runs lightweight layout and translation checks. It is not evidence of
simulation, policy quality, native runtime builds, or real-hardware operation. Full local
validation is described in [Contributing](../CONTRIBUTING.md).

After publication, select `main` as the default branch and configure branch protection around
the checks that actually run. Add a repository description and topics. Enable private
vulnerability reporting before advertising it as a support route. Releases and public tags
should describe tested public versions, rather than inheriting internal tags blindly.
