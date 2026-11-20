import os
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2          # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
NL = 256                           # interior cells per rank per axis
PAD = NL + 2                       # 258x258 padded local array
DX = 1.0 / 513.0
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15
SHM_SLOTS = 64


def log(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def get_gpu_uuid() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")

    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()

    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}::{line_bytes}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "little")          # 64-bit INSTANCE_SEED

    # non-periodic 2x2 Cartesian communicator
    cart = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    up, down = cart.Shift(0, 1)       # axis 0 = rows (y)
    left, right = cart.Shift(1, 1)    # axis 1 = cols (x)
    coords = cart.Get_coords(cart.Get_rank())

    # node-local shared-memory window (64 uint64 slots, 64-byte aligned view)
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if node_comm.Get_size() != 4:
        raise RuntimeError("All 4 ranks must live on one node for the shared window.")
    nslots_alloc = SHM_SLOTS + 8      # 64 B of slack so we can force 64-byte alignment
    nbytes = nslots_alloc * 8 if node_comm.Get_rank() == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, MPI.INFO_NULL, node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint64, shape=(nslots_alloc,))
    off = ((-raw.ctypes.data) % 64) // 8
    offs = node_comm.allgather(off)
    assert len(set(offs)) == 1, f"ranks disagree on alignment offset: {offs}"
    shm_view = raw[off:off + SHM_SLOTS]
    shm_aligned = (shm_view.ctypes.data % 64) == 0

    return dict(
        comm=comm, rank=rank, cart=cart, coords=coords,
        nb=(up, down, left, right),
        node_comm=node_comm, win=win, shm=shm_view, shm_aligned=bool(shm_aligned),
        gpu_uuid=gpu_uuid, cpu_sig=cpu_sig, line_bytes=line_bytes,
        seed=seed, github_user=github_user.strip(),
    )


# ==============================================================================
# TASK 2: Rank-local fields + halo pack/unpack kernels
# ==============================================================================
def initialize_fields(ctx: dict) -> dict:
    cy, cx = ctx["coords"]
    i = np.arange(PAD, dtype=np.float64)[:, None]
    j = np.arange(PAD, dtype=np.float64)[None, :]
    y = (cy * NL + i) * DX            # global coordinates, no global grid allocated
    x = (cx * NL + j) * DX
    phase = 2.0 * math.pi * (ctx["seed"] % 10007) / 10007.0

    u = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.02)
    u[0, :] = 0.0; u[-1, :] = 0.0; u[:, 0] = 0.0; u[:, -1] = 0.0
    kappa = 1.0 + 0.5 * np.sin(2 * math.pi * x + phase) * np.cos(2 * math.pi * y)
    src = 0.01 * np.exp(-((x - 0.25) ** 2 + (y - 0.75) ** 2) / 0.005)
    return dict(u=np.ascontiguousarray(u), kappa=np.ascontiguousarray(kappa),
                src=np.ascontiguousarray(src))


@cuda.jit
def pack_halos_kernel(u, sL, sR, sT, sB):
    k = cuda.grid(1)
    if k < NL:
        sL[k] = u[k + 1, 1]        # strided column
        sR[k] = u[k + 1, NL]       # strided column
        sT[k] = u[1, k + 1]        # contiguous row
        sB[k] = u[NL, k + 1]       # contiguous row


@cuda.jit
def unpack_halos_kernel(u, rL, rR, rT, rB):
    k = cuda.grid(1)
    if k < NL:
        u[k + 1, 0] = rL[k]
        u[k + 1, NL + 1] = rR[k]
        u[0, k + 1] = rT[k]
        u[NL + 1, k + 1] = rB[k]


# ==============================================================================
# TASK 3: 16x16 shared-memory tiled stencil + dual-stream pipeline
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(u, unew, kappa, src, i0, i1, j0, j1, r, dt, lam):
    tile = cuda.shared.array((SHARED_DIM, SHARED_DIM), dtype=float64)
    ty = cuda.threadIdx.y
    tx = cuda.threadIdx.x
    gi = i0 + cuda.blockIdx.y * TILE_DIM + ty
    gj = j0 + cuda.blockIdx.x * TILE_DIM + tx

    if gi < PAD and gj < PAD:
        tile[ty + 1, tx + 1] = u[gi, gj]
    else:
        tile[ty + 1, tx + 1] = 0.0
    if ty == 0 and gi > 0 and gj < PAD:
        tile[0, tx + 1] = u[gi - 1, gj]
    if ty == TILE_DIM - 1 and gi + 1 < PAD and gj < PAD:
        tile[SHARED_DIM - 1, tx + 1] = u[gi + 1, gj]
    if tx == 0 and gj > 0 and gi < PAD:
        tile[ty + 1, 0] = u[gi, gj - 1]
    if tx == TILE_DIM - 1 and gj + 1 < PAD and gi < PAD:
        tile[ty + 1, SHARED_DIM - 1] = u[gi, gj + 1]
    cuda.syncthreads()

    if gi < i1 and gj < j1:
        c = tile[ty + 1, tx + 1]
        lap = (tile[ty, tx + 1] + tile[ty + 2, tx + 1]
               + tile[ty + 1, tx] + tile[ty + 1, tx + 2] - 4.0 * c)
        unew[gi, gj] = c + r * kappa[gi, gj] * lap + dt * src[gi, gj] - dt * lam * c


def launch_region(S, i0, i1, j0, j1, stream=0):
    bx = (j1 - j0 + TILE_DIM - 1) // TILE_DIM
    by = (i1 - i0 + TILE_DIM - 1) // TILE_DIM
    tiled_stencil_kernel[(bx, by), (TILE_DIM, TILE_DIM), stream](
        S.u_in, S.u_out, S.kappa, S.src, i0, i1, j0, j1, S.r, S.dt, LAMBDA_DAMP)


class State:
    pass


KEYS = ("L", "R", "T", "B")


def make_state(ctx, fields):
    S = State()
    S.u_in = cuda.to_device(fields["u"])
    S.u_out = cuda.to_device(fields["u"])
    S.kappa = cuda.to_device(fields["kappa"])
    S.src = cuda.to_device(fields["src"])
    S.r = fields["r"]
    S.dt = fields["dt"]
    S.d_send = {k: cuda.device_array(NL, dtype=np.float64) for k in KEYS}
    S.d_recv = {k: cuda.to_device(np.zeros(NL)) for k in KEYS}
    S.h_send = {k: cuda.pinned_array(NL, dtype=np.float64) for k in KEYS}
    S.h_recv = {k: cuda.pinned_array(NL, dtype=np.float64) for k in KEYS}
    for k in KEYS:
        S.h_send[k][:] = 0.0
        S.h_recv[k][:] = 0.0
    S.stream_interior = cuda.stream()
    S.stream_boundary = cuda.stream()
    return S


def post_exchange(ctx, S):
    cart = ctx["cart"]
    up, down, left, right = ctx["nb"]
    reqs = [
        cart.Irecv(S.h_recv["B"], source=down, tag=0),
        cart.Irecv(S.h_recv["T"], source=up, tag=1),
        cart.Irecv(S.h_recv["R"], source=right, tag=2),
        cart.Irecv(S.h_recv["L"], source=left, tag=3),
        cart.Isend(S.h_send["T"], dest=up, tag=0),
        cart.Isend(S.h_send["B"], dest=down, tag=1),
        cart.Isend(S.h_send["L"], dest=left, tag=2),
        cart.Isend(S.h_send["R"], dest=right, tag=3),
    ]
    return reqs


def step_serialized(ctx, S):
    d = [S.d_send[k] for k in KEYS]
    rv = [S.d_recv[k] for k in KEYS]
    pack_halos_kernel[1, NL](S.u_in, *d)
    cuda.synchronize()
    for k in KEYS:
        S.d_send[k].copy_to_host(S.h_send[k])
    MPI.Request.Waitall(post_exchange(ctx, S))
    for k in KEYS:
        S.d_recv[k].copy_to_device(S.h_recv[k])
    unpack_halos_kernel[1, NL](S.u_in, *rv)
    launch_region(S, 1, NL + 1, 1, NL + 1)
    cuda.synchronize()
    S.u_in, S.u_out = S.u_out, S.u_in


def step_overlapped(ctx, S):
    si, sb = S.stream_interior, S.stream_boundary
    d = [S.d_send[k] for k in KEYS]
    rv = [S.d_recv[k] for k in KEYS]
    # deep interior [2..255]^2 runs while the halos travel over PCIe + MPI
    launch_region(S, 2, NL, 2, NL, stream=si)
    pack_halos_kernel[1, NL, sb](S.u_in, *d)
    for k in KEYS:
        S.d_send[k].copy_to_host(S.h_send[k], stream=sb)
    sb.synchronize()
    MPI.Request.Waitall(post_exchange(ctx, S))
    for k in KEYS:
        S.d_recv[k].copy_to_device(S.h_recv[k], stream=sb)
    unpack_halos_kernel[1, NL, sb](S.u_in, *rv)
    launch_region(S, 1, 2, 1, NL + 1, stream=sb)            # top row
    launch_region(S, NL, NL + 1, 1, NL + 1, stream=sb)      # bottom row
    launch_region(S, 2, NL, 1, 2, stream=sb)                # left col
    launch_region(S, 2, NL, NL, NL + 1, stream=sb)          # right col
    si.synchronize()
    sb.synchronize()
    S.u_in, S.u_out = S.u_out, S.u_in


def local_energy(S):
    cuda.synchronize()
    u = S.u_in.copy_to_host()
    return float(np.sum(u[1:NL + 1, 1:NL + 1] ** 2)) * DX * DX


def run_mode(ctx, fields, mode, steps):
    cart = ctx["cart"]
    S = make_state(ctx, fields)
    step = step_serialized if mode == "serialized" else step_overlapped
    energies = {0: cart.allreduce(local_energy(S), op=MPI.SUM)}
    cuda.synchronize()
    cart.Barrier()
    t0 = time.perf_counter()
    for n in range(1, steps + 1):
        step(ctx, S)
        if n % 100 == 0:
            energies[n] = cart.allreduce(local_energy(S), op=MPI.SUM)
    cuda.synchronize()
    cart.Barrier()
    ms = (time.perf_counter() - t0) * 1e3
    ms = cart.allreduce(ms, op=MPI.MAX)
    return S.u_in.copy_to_host(), energies, ms


# ==============================================================================
# TASK 4: Host false-sharing sweep inside the MPI-3 shared window
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(arr, idx, idx2, iters):
    # load from idx, store to idx2 (equal at runtime, but LLVM cannot prove it),
    # so the load/store pair cannot be promoted to a register.
    for _ in range(iters):
        v = arr[idx]
        arr[idx2] = v + np.uint64(1)


def benchmark_shm_strides(ctx, iters=SHM_ITERS, reps=3):
    nc = ctx["node_comm"]
    shm = ctx["shm"]
    r = nc.Get_rank()
    host_shm_accumulate(shm, r, r, 1000)          # JIT warm-up
    nc.Barrier()
    results = {}
    for stride in (1, 8, 16):
        best = float("inf")
        ok_all = True
        for _ in range(reps):
            if r == 0:
                shm[:] = 0
            nc.Barrier()
            idx = r * stride
            t0 = time.perf_counter()
            host_shm_accumulate(shm, idx, idx, iters)
            t = time.perf_counter() - t0
            nc.Barrier()
            tmax = nc.allreduce(t, op=MPI.MAX)
            ok = nc.allreduce(int(int(shm[idx]) == iters), op=MPI.MIN) == 1
            ok_all = ok_all and ok
            best = min(best, tmax)
        results[stride] = dict(ms=best * 1e3, bytes=stride * 8, counters_ok=bool(ok_all))
    return results


# ==============================================================================
# TASK 5: audits + main
# ==============================================================================
def first_value(x):
    return next(iter(x.values())) if isinstance(x, dict) else x


def audit_ptx():
    ptx = first_value(tiled_stencil_kernel.inspect_asm())
    return {
        "st.shared.f64": ptx.count("st.shared.f64"),
        "ld.shared.f64": ptx.count("ld.shared.f64"),
        "bar.sync": ptx.count("bar.sync"),
    }


def audit_llvm():
    ir = first_value(host_shm_accumulate.inspect_llvm())
    return {
        "load_i64": ir.count("load i64"),
        "store_i64": ir.count("store i64"),
        "ir_lines": ir.count("\n"),
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: hybrid_mpi_cuda_cache.py <github_username>")
    ctx = init_hybrid_context(sys.argv[1])
    cart, rank = ctx["cart"], ctx["rank"]
    log(rank, f"[ctx] user={ctx['github_user']} seed={ctx['seed']:#018x} gpu={ctx['gpu_uuid']}")
    log(rank, f"[ctx] cpu={ctx['cpu_sig']} line={ctx['line_bytes']}B shm_aligned64={ctx['shm_aligned']}")

    fields = initialize_fields(ctx)
    kappa_max = cart.allreduce(float(fields["kappa"][1:NL + 1, 1:NL + 1].max()), op=MPI.MAX)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = 0.9 * dt_max
    fields["dt"] = dt
    fields["r"] = dt / (DX * DX)
    log(rank, f"[cfl] kappa_max={kappa_max:.6f} dx={DX:.8e} dt_max={dt_max:.6e} dt={dt:.6e} stable={dt <= dt_max}")

    run_mode(ctx, fields, "serialized", 3)         # warm-up (JIT)
    run_mode(ctx, fields, "overlapped", 3)

    u_ser, e_ser, ms_ser = run_mode(ctx, fields, "serialized", STENCIL_STEPS)
    u_ovl, e_ovl, ms_ovl = run_mode(ctx, fields, "overlapped", STENCIL_STEPS)

    max_diff = cart.allreduce(float(np.max(np.abs(u_ser - u_ovl))), op=MPI.MAX)
    log(rank, f"[gpu] gpu_serialized_ms={ms_ser:.3f} gpu_overlapped_ms={ms_ovl:.3f} ratio={ms_ser / ms_ovl:.4f}")
    log(rank, f"[parity] max_diff={max_diff:.3e} (< 1e-12: {max_diff < 1e-12})")
    for n in sorted(e_ovl):
        log(rank, f"[energy] step {n:3d}: serialized={e_ser[n]:.12e} overlapped={e_ovl[n]:.12e}")
    evals = [e_ovl[n] for n in sorted(e_ovl)]
    assert max_diff < 1e-12, "serialized/overlapped parity failed"
    assert all(a > b for a, b in zip(evals, evals[1:])), "energy is not dissipating"

    shm = benchmark_shm_strides(ctx)
    for s, d in shm.items():
        log(rank, f"[shm] stride={s:2d} ({d['bytes']:3d} B): {d['ms']:.2f} ms counters_ok={d['counters_ok']}")
    fs_ratio = shm[1]["ms"] / shm[8]["ms"]
    pad_ratio = shm[16]["ms"] / shm[8]["ms"]
    log(rank, f"[shm] false_sharing_slowdown(stride1/stride8)={fs_ratio:.3f} stride16/stride8={pad_ratio:.3f}")

    ptx = audit_ptx()
    llvm = audit_llvm()
    log(rank, f"[ptx] {ptx}")
    log(rank, f"[llvm] {llvm}")
    assert ptx["st.shared.f64"] > 0 and ptx["ld.shared.f64"] > 0 and ptx["bar.sync"] > 0

    if rank == 0:
        manifest = dict(
            github_username=ctx["github_user"], instance_seed=ctx["seed"],
            gpu_uuid=ctx["gpu_uuid"], cpu_signature=ctx["cpu_sig"],
            cpu_line_bytes=ctx["line_bytes"], shm_aligned_64=ctx["shm_aligned"],
            kappa_max=kappa_max, dx=DX, dt=dt, dt_max=dt_max, cfl_ok=bool(dt <= dt_max),
            gpu_serialized_ms=ms_ser, gpu_overlapped_ms=ms_ovl, max_diff=max_diff,
            energy_serialized={str(k): v for k, v in e_ser.items()},
            energy_overlapped={str(k): v for k, v in e_ovl.items()},
            shm_ms={str(k): v["ms"] for k, v in shm.items()},
            false_sharing_ratio=fs_ratio, stride16_vs_stride8=pad_ratio,
            ptx_audit=ptx, llvm_audit=llvm,
            mpi_ranks=4, stencil_steps=STENCIL_STEPS, shm_iters=SHM_ITERS,
        )
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("[manifest] wrote hybrid_lab_manifest.json", flush=True)


if __name__ == "__main__":
    main()
