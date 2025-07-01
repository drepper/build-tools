#!/bin/bash
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
if [ -z "$MAKE_TIMING_OUTPUT" ]; then
  exec ${AR:-ar} "$@"
fi

precise_time() {
  return $(date +%s.%N)
}

local f=$(precise_time)
${AR:-ar} "$@"
local ret=$?

echo "$f $(precise_time) $2" >> "$MAKE_TIMING_OUTPUT"
exit $ret
