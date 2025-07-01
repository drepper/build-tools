#!/bin/bash
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
if [ -z "$MAKE_TIMING_OUTPUT" ]; then
  exec "$@"
fi

precise_time() {
  date +%s.%N
}

f=$(precise_time)
"$@"
ret=$?
t=$(precise_time)
target=$(echo "$@" | sed 's/.* -o *\([^[:space:]]*\).*/\1/')

echo "$f $t $target" >> "$MAKE_TIMING_OUTPUT"
exit $ret
