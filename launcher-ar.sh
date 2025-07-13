#!/bin/bash
# Copyright © 2025 Ulrich Drepper
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
if [[ -z "${MAKE_TIMING_OUTPUT}" ]]; then
  exec ${AR:-ar} "$@"
fi

precise_time() {
  date +%s.%N
}

f=$(precise_time)
${AR:-ar} "$@"
ret=$?
t=$(precise_time)

if [[ -n "$2" ]]; then
  echo "${f} ${t} ${2/#${PWD}\//}" >> "${MAKE_TIMING_OUTPUT}"
fi
exit "${ret}"
