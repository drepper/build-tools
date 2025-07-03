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
target=$(echo "$@" | sed -n 's/\(^\|.* \)-o *\([^[:space:]]*\).*/\2/p')

if [ "$target" ]; then
  echo "$f $t $target" >> "$MAKE_TIMING_OUTPUT"
fi
exit $ret
