#!/bin/bash
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
if [ -z "$MAKE_TIMING_OUTPUT" ]; then
  exec ${BISON:-bison} "$@"
fi

precise_time() {
  date +%s.%N
}

f=$(precise_time)
${BISON:-bison} "$@"
ret=$?
t=$(precise_time)
target=$(echo "$@" | sed -n 's/\(^\|.* \)-o *\([^[:space:]]*\).*/\2/p')
echo "----" >> /tmp/AAA
echo "$@" >> /tmp/AAA
echo "$target" >> /tmp/AAA

if [ "$target" ]; then
  echo "$f $t $target" >> "$MAKE_TIMING_OUTPUT"
fi
exit $ret
