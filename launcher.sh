#!/bin/bash
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
if [ -z "$MAKE_TIMING_OUTPUT" ]; then
  exec "$@"
fi

precise_time() {
  date +%s.%N
}

local f=$(precise_time)
"$@"
local ret=$?
local t=$(precise_time)
local target=$(echo "$@" | sed 's/.* -o *\([^[:space:]]*\).*/\1/')

echo "$f $t $target" >> "$MAKE_TIMING_OUTPUT"
exit $ret
