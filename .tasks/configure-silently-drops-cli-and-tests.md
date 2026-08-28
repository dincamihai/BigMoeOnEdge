# A reconfigure without llama.cpp silently drops cli/ and tests/, and the build still says it succeeded

column: Todo
created: 2026-08-28

## What happened

`third_party/llama.cpp` was momentarily absent -- a `git checkout -- .` had replaced the
working symlink with the empty submodule directory -- and `cmake ..` happened to run in that
window. The root `CMakeLists.txt` adds llama.cpp only `if(EXISTS
third_party/llama.cpp/CMakeLists.txt)`, and `cli/` and `tests/` hang off `BMOE_HAVE_LLAMA`. So
the build reconfigured itself into a tree with ONE target, `bmoe_core`.

Nothing said so. Every `cmake --build .` from then on printed

    [100%] Built target bmoe_core

and exited 0, which reads exactly like a successful full build. `bmoe-server`, `bmoe-cli` and
every test binary were simply no longer part of `all`. They stayed on disk at their old
timestamps, so `ctest` kept passing and the gates kept reporting 32/32 -- against binaries
built hours earlier, from source that had since changed.

Two edits were verified this way before it was noticed, and both verifications were false. The
symptom that finally exposed it: a fix was applied, rebuilt, and the running service crashed
with the exact error the fix removed. `stat` on the binary showed it was two hours old.

## Why it is worth fixing rather than remembering

A build that reports success while building LESS than asked is worse than one that fails: the
failure mode is a green check on work that was never compiled. The submodule being a symlink
here makes the window easy to hit -- any tool that restores paths from git can close it for a
moment, and a `cmake` run in that moment poisons the build directory until someone reconfigures.

## Options

1. **Fail the configure.** If `third_party/llama.cpp/CMakeLists.txt` is missing, `message(FATAL_ERROR)`
   with the submodule/symlink instruction instead of quietly configuring a core-only tree. The
   cheapest fix, and it turns a silent wrong build into a loud stop. Check first whether a
   core-only configure is a supported use (a host without the submodule may want just the library).
2. **Say it loudly at build time, not only at configure.** The configure line
   `BigMoeOnEdge: llama.cpp present = ON/OFF` already exists but scrolls past once; nothing repeats
   it per build.
3. **A one-line check in `scripts/build-host.sh`**: after building, assert the expected binaries
   exist and are newer than the newest source file, and fail if not.

## Diagnostic that identifies it in seconds

    cmake --build . --target help | grep bmoe

If that does not list `bmoe-server`, `bmoe-cli` and the test targets, the build directory is
configured without llama.cpp and `cmake --build .` is building a fraction of the project. The fix
is to restore `third_party/llama.cpp` and re-run `cmake ..`.
