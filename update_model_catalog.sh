#!/usr/bin/env sh
# The shared runtime/catalog is an independent repository (this directory)
# mounted as the ``agent-common`` submodule of every checkout.  There are no
# per-checkout copies to install: edit and commit here, then check out the
# pinned submodule commit in each checkout.

set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(dirname "$here")

for repo in generic-debug-pipeline paper-refine paper-refine-eval; do
    checkout="$root/$repo"
    if [ ! -d "$checkout" ]; then
        printf 'skip (missing checkout): %s\n' "$checkout"
        continue
    fi
    if [ ! -e "$checkout/.git" ]; then
        printf 'skip (not a git checkout): %s\n' "$checkout"
        continue
    fi
    git -C "$checkout" submodule update --init --recursive agent-common
    printf 'synced submodule %s/agent-common\n' "$checkout"
done

printf 'OK: agent-common submodule checked out; run check_model_catalog.py\n'
