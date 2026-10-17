#!/usr/bin/env bash
# frictiontune CLI entrypoint for the `data`, `train` and `dev` images.
#
# The image owns the toolchain (ROCm, Python, the heavy wheels). The repository
# is bind-mounted at /work at run time, and its `packages/*/src` are placed ahead
# of the baked install on PYTHONPATH, so pipeline *code* stays live ("edit and
# re-run") while the expensive layers are never rebuilt for a one-line change.
#
# This also keeps the run-id contract intact: `code_revision()` hashes the git
# state of the mounted repo, so a container run and a host run of the same tree
# resolve to the same run_id and share the stage/artifact cache.
set -euo pipefail

if [ -d /work/packages/frictiontune/src ]; then
  _shadow="/work/packages/frictiontune/src:/work/packages/frictiontune-train/src"
  export PYTHONPATH="${_shadow}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Run from the mounted repo so relative config/artifact paths resolve as they do
# on the host.
if [ -d /work ]; then
  cd /work
fi

exec frictiontune "$@"