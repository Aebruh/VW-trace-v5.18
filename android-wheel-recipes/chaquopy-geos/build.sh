#!/bin/bash
set -euxo pipefail

./configure --host="$HOST" --prefix="$PREFIX"
make -j "$CPU_COUNT"
make install

rm -rf "$PREFIX/bin"
rm -f "$PREFIX"/lib/*.a
# Keep only the C API shared library in the wheel. With this GEOS build,
# libgeos_c is linked with the C++ implementation as expected by Chaquopy.
rm -f "$PREFIX"/lib/libgeos-*.so "$PREFIX"/lib/libgeos.la
