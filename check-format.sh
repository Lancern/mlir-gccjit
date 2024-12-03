#!/usr/bin/env bash

set -eu

diff=$(git clang-format -q --extensions c,cc,cpp,h,hpp --diff HEAD~1)
if [[ -n diff ]]; then
    # clang-format outputs some diff, which indicates the commit is not properly
    # formatted.
    exit 1
fi
