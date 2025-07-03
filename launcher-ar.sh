#!/bin/bash
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
if [ -z "$MAKE_TIMING_OUTPUT" ]; then
  exec ${AR:-ar} "$@"
fi

precise_time() {
  date +%s.%N
}

f=$(precise_time)
${AR:-ar} "$@"
ret=$?
t=$(precise_time)

if [ "$2" ]; then
  target=$(echo "$2" | sed "s|^$PWD/||")
  echo "$f $t $target" >> "$MAKE_TIMING_OUTPUT"
fi
exit $ret
