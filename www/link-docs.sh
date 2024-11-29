#!/usr/bin/env bash

set -e

if [[ $# -ne 1 ]]; then
    echo "link-docs.sh requires 1 argument"
    exit 1
fi

rm -f content/dialect/*.md

hugo new content --kind dialect content/dialect/_index.md
hugo new content --kind ops content/dialect/ops.md
hugo new content --kind types content/dialect/types.md
hugo new content --kind attrs content/dialect/attrs.md
hugo new content --kind passes content/dialect/passes.md

cat "$1/docs/gccjit/GCCJITDialect.md" >> content/dialect/_index.md
cat "$1/docs/gccjit/GCCJITOps.md" >> content/dialect/ops.md
cat "$1/docs/gccjit/GCCJITTypes.md" >> content/dialect/types.md
cat "$1/docs/gccjit/GCCJITAttrs.md" >> content/dialect/attrs.md
cat "$1/docs/gccjit/GCCJITPasses.md" >> content/dialect/passes.md
