#!/bin/bash
set -e

echo 'shellcheck tests'
args='-o all'
for f in launcher.sh launcher-{ar,flex,bison}.sh; do
  shellcheck ${args} ${f} && printf "%20s OK\n" "$f"
done
