# Scripts

Run from the project directory; set `WF_BUILD_DIR` if the build directory is not `build`.

| Script | Purpose |
|---|---|
| `run_benchmarks.sh [--quick]` | runs `wfbench` for every scene in `bench/protocol.json` and writes `results/bench-<scene>.json`; stops only on an error, never on a weakened verdict |
| `profile_ncu.sh [--quick]` | Nsight Compute on one pass per arm (divergence scene); stores CSVs, or the exact failing command and message when counters are unavailable |
| `profile_stats.sh` | Nsight Compute for all three scenes and both arms with raw metric ids; writes `results/profiles/stats/<scene>-<mode>.csv` and the command used |
| `run_sanitizers.sh [--quick]` | `compute-sanitizer` memcheck on both GPU arms and the ASan/UBSan CPU run; records every command, exit code and error count in `results/sanitizers.json` |
| `view.sh [scene] [arm]` | opens the viewer; without a display it writes one frame to `/tmp/wfviewer-frame.png` instead |

Scripts never contain absolute paths; the tests check that.
