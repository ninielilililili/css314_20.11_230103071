import os
import sys
import json
import time
import hashlib
import platform
import threading
import subprocess
import numpy as np
from numba import njit, uint64

try:
    import resource
except ImportError:
    resource = None

# Replace with your exact Git handle before executing
GIT_USERNAME = "ninielilililili"
NUM_THREADS = 4
ITERATIONS_PER_THREAD = 60_000_000
STRIDES = {"A_contiguous": 1, "B_half_line": 4, "C_line_padded": 8, "D_prefetch_safe": 16}
REPEATS = 3          # each layout is run REPEATS times; the best (minimum) time is reported
MASK64 = (1 << 64) - 1


def detect_cache_line_size() -> int:
    """Query L1 data cache coherency line size in bytes from OS sysfs/sysctl."""
    linux_path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(linux_path):
        with open(linux_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    if sys.platform == "darwin":
        out = subprocess.check_output(["sysctl", "-n", "hw.cachelinesize"], encoding="utf-8")
        return int(out.strip())
    return 64


def cpu_model_string() -> str:
    """Best-effort CPU model string (used for the identity seed)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           encoding="utf-8").strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# ==============================================================================
# TASK 1: Hardware Topology & Student Identity Context
# ==============================================================================
def init_system_context(git_user: str) -> dict:
    assert git_user != "REPLACE_WITH_YOUR_GIT_USERNAME" and len(git_user.strip()) > 0, (
        "Set GIT_USERNAME to your real Git handle before running."
    )
    logical_cores = os.cpu_count() or 1
    if logical_cores < NUM_THREADS:
        raise RuntimeError(f"Requires at least {NUM_THREADS} logical cores; detected {logical_cores}.")

    line_size = detect_cache_line_size()
    cpu_model = cpu_model_string()
    cpu_sig = f"{platform.machine()}::{cpu_model}::{logical_cores}"

    # 64-bit seed: SHA-256(git_user :: cpu_sig), first 16 hex digits as base-16 int
    digest = hashlib.sha256(f"{git_user.strip()}::{cpu_sig}".encode("utf-8")).hexdigest()
    instance_seed = int(digest[:16], 16)

    # core affinity: pick NUM_THREADS distinct cores from the set the OS lets us use
    affinity_supported = hasattr(os, "sched_setaffinity")
    if affinity_supported:
        allowed = sorted(os.sched_getaffinity(0))
        core_ids = allowed[:NUM_THREADS]
        if len(core_ids) < NUM_THREADS:
            affinity_supported = False
            core_ids = []
    else:
        core_ids = []

    return {
        "git_username": git_user.strip(),
        "logical_cores": logical_cores,
        "cache_line_size": line_size,
        "cpu_model": cpu_model,
        "cpu_sig": cpu_sig,
        "instance_seed": instance_seed,
        "affinity_supported": affinity_supported,
        "core_ids": core_ids,
    }


def aligned_uint64_buffer(n_elems: int, align_bytes: int = 128) -> np.ndarray:
    """Zeroed uint64 array whose first element sits on an `align_bytes` boundary."""
    extra = align_bytes // 8
    raw = np.zeros(n_elems + extra, dtype=np.uint64)
    offset = ((-raw.ctypes.data) % align_bytes) // 8
    buf = raw[offset:offset + n_elems]
    assert buf.ctypes.data % align_bytes == 0
    return buf


# ==============================================================================
# TASK 2: GIL-free native counter kernel
# ==============================================================================
@njit(nogil=True)
def worker_strided_accumulate(buffer, slot, iters, seed, guard):
    """In-place read-modify-write on buffer[slot] every iteration.

    `guard` is a runtime value that is always 0. The store address is
    slot + (v & guard), i.e. it depends on the value just loaded, so LLVM cannot prove that
    every iteration touches the same address and therefore cannot promote buffer[slot] into a
    register (scalar promotion). Each iteration must issue a real load and a real store.
    """
    inc = (seed & uint64(0xFF)) | uint64(1)
    for _ in range(iters):
        idx = slot + (buffer[slot] & guard)
        buffer[idx] = buffer[idx] + inc


# ==============================================================================
# TASK 4: Thread-local register privatization baseline
# ==============================================================================
@njit(nogil=True)
def worker_private_accumulate(buffer, slot, iters, seed):
    """Accumulate in a local scalar (register), write to shared memory exactly once."""
    inc = (seed & uint64(0xFF)) | uint64(1)
    acc = uint64(0)
    for _ in range(iters):
        acc += inc
    buffer[slot] = acc


def thread_seed(instance_seed: int, tid: int) -> int:
    return (instance_seed + (tid + 1) * 0x9E3779B97F4A7C15) & MASK64


def expected_final(instance_seed: int, tid: int, iters: int) -> int:
    inc = (thread_seed(instance_seed, tid) & 0xFF) | 1
    return (inc * iters) & MASK64


def _pin(core):
    if core is not None and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core})   # pid 0 = the calling *thread* on Linux
        except OSError:
            pass


def _run_threads(ctx, target, args_for_tid):
    """Run NUM_THREADS threads released together; return (elapsed_ns, csw_voluntary, csw_invol)."""
    barrier = threading.Barrier(NUM_THREADS + 1)
    cores = ctx["core_ids"] if ctx["affinity_supported"] else [None] * NUM_THREADS

    def runner(tid):
        _pin(cores[tid])
        barrier.wait()
        target(*args_for_tid(tid))

    threads = [threading.Thread(target=runner, args=(t,)) for t in range(NUM_THREADS)]
    for t in threads:
        t.start()
    ru0 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    barrier.wait()
    t0 = time.perf_counter_ns()
    for t in threads:
        t.join()
    t1 = time.perf_counter_ns()
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    vol = (ru1.ru_nvcsw - ru0.ru_nvcsw) if resource else -1
    invol = (ru1.ru_nivcsw - ru0.ru_nivcsw) if resource else -1
    return t1 - t0, vol, invol


# ==============================================================================
# TASK 3: Multi-threaded stride sweep
# ==============================================================================
def run_stride_benchmark(ctx: dict, iters: int = ITERATIONS_PER_THREAD) -> dict:
    seed = ctx["instance_seed"]
    results = {}
    guard = np.uint64(0)

    # JIT warm-up (compile outside the timed region)
    warm = aligned_uint64_buffer(NUM_THREADS * 16 + 16)
    worker_strided_accumulate(warm, 0, 10, np.uint64(1), guard)
    worker_private_accumulate(warm, 0, 10, np.uint64(1))

    for name, stride in STRIDES.items():
        best = None
        for rep in range(REPEATS):
            buf = aligned_uint64_buffer(NUM_THREADS * stride + 16)
            ns, vol, invol = _run_threads(
                ctx, worker_strided_accumulate,
                lambda tid: (buf, tid * stride, iters, np.uint64(thread_seed(seed, tid)), guard))
            finals = [int(buf[tid * stride]) for tid in range(NUM_THREADS)]
            if best is None or ns < best["elapsed_ns"]:
                best = {"stride": stride, "stride_bytes": stride * 8, "elapsed_ns": ns,
                        "ns_per_update": ns / iters, "ctx_voluntary": vol,
                        "ctx_involuntary": invol, "finals": finals}
        results[name] = best

    # privatized baseline (stride = 8 slots, one write at the very end)
    best = None
    for rep in range(REPEATS):
        buf = aligned_uint64_buffer(NUM_THREADS * 8 + 16)
        ns, vol, invol = _run_threads(
            ctx, worker_private_accumulate,
            lambda tid: (buf, tid * 8, iters, np.uint64(thread_seed(seed, tid))))
        finals = [int(buf[tid * 8]) for tid in range(NUM_THREADS)]
        if best is None or ns < best["elapsed_ns"]:
            best = {"stride": 8, "stride_bytes": 64, "elapsed_ns": ns,
                    "ns_per_update": ns / iters, "ctx_voluntary": vol,
                    "ctx_involuntary": invol, "finals": finals}
    results["private_register"] = best
    return results


# ==============================================================================
# Optional (Q5): pure-Python-bytecode threads under the GIL
# ==============================================================================
def _py_counter(buf, slot, iters):
    for _ in range(iters):
        buf[slot] += 1


def run_gil_experiment(iters: int = 1_000_000) -> dict:
    out = {}
    for n in (1, 4):
        buf = [0] * 16
        ths = [threading.Thread(target=_py_counter, args=(buf, t, iters)) for t in range(n)]
        t0 = time.perf_counter_ns()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        out[f"threads_{n}_ns"] = time.perf_counter_ns() - t0
        out[f"threads_{n}_total_updates"] = n * iters
    out["slowdown_4_vs_1_per_total_work"] = (
        (out["threads_4_ns"] / out["threads_4_total_updates"])
        / (out["threads_1_ns"] / out["threads_1_total_updates"]))
    return out


# ==============================================================================
# TASK 5: Exact state parity & LLVM IR verification
# ==============================================================================
def audit_llvm_and_parity(ctx: dict, results: dict, iters: int, gil: dict) -> dict:
    seed = ctx["instance_seed"]
    expected = [expected_final(seed, t, iters) for t in range(NUM_THREADS)]
    parity = {name: (r["finals"] == expected) for name, r in results.items()}
    bit_exact = all(parity.values())

    ir = next(iter(worker_strided_accumulate.inspect_llvm().values()))
    store_i64 = ir.count("store i64")
    load_i64 = ir.count("load i64")
    # PyEval_SaveThread/RestoreThread in the wrapper is the nogil GIL release (expected).
    # (PyNumber_* appears only in the argument-unboxing wrapper.) Object mode would call generic item APIs.
    py_fallback = any(tok in ir for tok in ("PyObject_GetItem", "PyObject_SetItem",
                                            "PyLong_FromLong", "PyFloat_", "PyObject_Call"))
    llvm_ok = store_i64 > 0 and load_i64 > 0 and not py_fallback

    t = {k: v["elapsed_ns"] for k, v in results.items()}
    manifest = {
        "git_username": ctx["git_username"],
        "instance_seed": ctx["instance_seed"],
        "cpu_model": ctx["cpu_model"],
        "logical_cores": ctx["logical_cores"],
        "cache_line_size": ctx["cache_line_size"],
        "affinity_pinning": ctx["core_ids"] if ctx["affinity_supported"] else "unsupported",
        "num_threads": NUM_THREADS,
        "iterations_per_thread": iters,
        "timings": {k: {kk: vv for kk, vv in v.items() if kk != "finals"} for k, v in results.items()},
        "slowdown_stride1_vs_stride8": t["A_contiguous"] / t["C_line_padded"],
        "slowdown_stride4_vs_stride8": t["B_half_line"] / t["C_line_padded"],
        "ratio_stride16_vs_stride8": t["D_prefetch_safe"] / t["C_line_padded"],
        "ratio_stride8_vs_private": t["C_line_padded"] / t["private_register"],
        "ratio_stride1_vs_private": t["A_contiguous"] / t["private_register"],
        "expected_finals": expected,
        "bit_exact_parity": bit_exact,
        "parity_per_layout": parity,
        "llvm_store_i64": store_i64,
        "llvm_load_i64": load_i64,
        "llvm_python_fallback_detected": py_fallback,
        "llvm_audit_passed": llvm_ok,
        "gil_experiment": gil,
        "verification_passed": bool(bit_exact and llvm_ok),
    }
    with open("cache_lab_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ctx = init_system_context(GIT_USERNAME)
    print(f"Git user          : {ctx['git_username']}")
    print(f"CPU model         : {ctx['cpu_model']}")
    print(f"Logical cores     : {ctx['logical_cores']}")
    print(f"Cache line size   : {ctx['cache_line_size']} B")
    print(f"INSTANCE_SEED     : {ctx['instance_seed']}")
    print(f"Affinity cores    : {ctx['core_ids'] if ctx['affinity_supported'] else 'unsupported'}")
    print("-" * 72)

    results = run_stride_benchmark(ctx, ITERATIONS_PER_THREAD)
    for name, r in results.items():
        print(f"{name:18s} stride={r['stride']:2d} ({r['stride_bytes']:3d} B)  "
              f"{r['elapsed_ns']:>14,d} ns  {r['ns_per_update']:8.3f} ns/update  "
              f"csw vol/invol = {r['ctx_voluntary']}/{r['ctx_involuntary']}")
    gil = run_gil_experiment()
    m = audit_llvm_and_parity(ctx, results, ITERATIONS_PER_THREAD, gil)
    print("-" * 72)
    print(f"slowdown stride1 / stride8   : {m['slowdown_stride1_vs_stride8']:.2f}x")
    print(f"slowdown stride4 / stride8   : {m['slowdown_stride4_vs_stride8']:.2f}x")
    print(f"stride16 / stride8           : {m['ratio_stride16_vs_stride8']:.2f}x")
    print(f"stride8  / private register  : {m['ratio_stride8_vs_private']:.2f}x")
    print(f"bit-exact parity             : {m['bit_exact_parity']}")
    print(f"LLVM store i64 / load i64    : {m['llvm_store_i64']} / {m['llvm_load_i64']}  "
          f"(python fallback: {m['llvm_python_fallback_detected']})")
    print(f"GIL experiment (4 vs 1 thr)  : {gil['slowdown_4_vs_1_per_total_work']:.2f}x per update")
    print(f"verification_passed          : {m['verification_passed']}")
    print("cache_lab_manifest.json written.")


if __name__ == "__main__":
    main()