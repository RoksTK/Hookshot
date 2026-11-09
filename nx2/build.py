"""Compile the lifted game (work/gen, from `lift.py --write`) into work/build/hookshot_game.dll (or a static library).

The flags are a64-kit's for lifted code (its build.py, cmd_native): -O1, LoopRotate's header copies capped (most of
the compile time in goto-dense giant functions otherwise), capture tracking past LLVM's 100 uses (the context
parameter is A64_NOESC), one section per function. The guest model is the 64-bit window (rtgen.py), shadowing the
vendored rt/, with the loop-header interrupt polls on (A64_POLLING, nx2/poll.py). Incremental: an object is rebuilt
only when its source or the flags changed.

hookshot_game.dll = the lifted chunks + a64-kit's tables and runtime + nx2/rt/eden_shim.c (the emulator ABI, mkw_host.h)
+ mkw_info.c (the image base and main's build id, generated here from work/image/image.json). Linking it is also the
check that the whole lifted game closes over itself: any unresolved symbol fails the link.

Heavy: 44 chunks of ~25 MB, ~100-200 s and ~2-3.5 GB each at -O1. The PC is shared with games: run it when the
machine is idle, and keep --jobs low otherwise.

--standalone builds work/sa/build/hookshot.exe instead: the standalone lift (work/sa/gen, `lift.py --standalone --write`)
+ the native runtime in host/ (TOTKPort's hos/, adapted). Objects are cached by their key in work/objcache, shared by
both builds: the two lifts share their chunk boundaries, so every chunk without a hooked function is the same source
and is not compiled twice.

Usage:  python -I nx2/build.py [--jobs 1] [--only 0,7] [--opt O1] [--no-polls] [--dll] [--lib] [--standalone]
  --only: just these code_NNN.c chunks (to measure); --dll: link hookshot_game.dll; --lib: archive hookshot_game.lib
  HOOKSHOT_TIMES=<build_times.json>: make_game.py's record of the earlier builds. With it, this run says which files it
  really rebuilds and how long each took before (@@plan, @@file, @@eta), which the builder windows show as a countdown.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # (python -I leaves the script folder off)
import kit
import lifted_c
import rtgen
import logo_rgba
import toolchain
import version
import win_manifest

NX2_RT = os.path.join(kit.HERE, "rt")
HOST = os.path.join(kit.HERE, "..", "host")
OPUS = os.path.join(kit.HERE, "..", "third_party", "opus")
OBJ_CACHE = os.path.join(kit.WORK, "objcache")
# (hookshot) work/hot_order.txt: the functions that run, hottest first (sampled), packed together by the linker; RX_NO_ORDER=1: off
PGO = os.environ.get("RX_PGO", "")


# (linux) Linux builds: an ELF executable and hookshot_vk.so, the Win32 calls of host/ and gpu/vk on POSIX
# (host/posix: its windows.h is found first), the window and sound on SDL2. HOOKSHOT_SYSROOT=<dir>: compile and link
# against that root (the AppImage's: glibc 2.35, so the game runs on older systems than the one that built it).
# HOOKSHOT_VK_PREBUILT=<hookshot_vk.so>: the Vulkan back end as built already (the AppImage's), not compiled here.
LINUX = not toolchain.windows()   # (hookshot: RX_TARGET=windows on Linux builds the Windows game, with llvm-mingw)
POSIX = os.path.join(kit.HERE, "..", "host", "posix")
SYSROOT = os.environ.get("HOOKSHOT_SYSROOT", "")
EXE = "hookshot" if LINUX else "hookshot.exe"
VK_LIB = "hookshot_vk.so" if LINUX else "hookshot_vk.dll"


def target():
    """toolchain.native_target(), with the sysroot on Linux"""
    return toolchain.native_target() + (["--sysroot=" + SYSROOT] if LINUX and SYSROOT else [])


def no_throttle(handle=None):
    """(issue 87) Windows' power throttling (EcoQoS) slows a background window's processes to almost nothing on some machines: the
    build sat at 0% CPU. Turned off for this process, or the one whose handle this is"""
    if os.name != "nt":
        return
    import ctypes
    class State(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong), ("StateMask", ctypes.c_ulong)]
    k = ctypes.WinDLL("kernel32")
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    s = State(1, 1, 0)   # (PROCESS_POWER_THROTTLING_EXECUTION_SPEED in the mask, off in the state)
    try:
        k.SetProcessInformation(ctypes.c_void_p(int(handle)) if handle else k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))
    except (OSError, ValueError, TypeError):
        pass

def HOT_ORDER():
    p = os.path.join(kit.WORK, "hot_order.txt")
    if not os.path.exists(p) or os.environ.get("RX_NO_ORDER"):
        return []
    if LINUX:
        return ["-Wl,--symbol-ordering-file=" + p, "-Wl,--no-warn-symbol-ordering"]
    return [toolchain.link_opt("/order:@" + p.replace("\\", "/")), toolchain.link_opt("/ignore:4037")]


HOST_LIBS = ["-lpthread", "-ldl", "-lm"] if LINUX else \
    ["-lwinhttp", "-luser32", "-lwinmm", "-ladvapi32", "-lole32", "-lshell32", "-lgdi32", "-ldbghelp"]
GPU = os.path.join(kit.HERE, "..", "gpu")
# FFmpeg (libavcodec's VP8/VP9 decoders, swscale) for nn::codec's video decoders (host/nn_video.c): the static build
# in the Eglantine clone's package cache
FFMPEG = None   # (found in the CPM caches below, once cpm_dirs is defined)
# the renderer's WebGPU: a wgpu-native release (include/, lib/); TOTKPort's build of it by default
WGPU = os.environ.get("WGPU_NATIVE", os.path.join(os.path.expanduser("~"), "Desktop", "TOTKPort", "third_party",
                                                  "wgpu-native"))


def gpu_sources():
    """The renderer (gpu/render, TOTKPort's, NVN replayed with WebGPU) and what it builds on, from suyu
    (gpu/third_party/suyu): the shader recompiler (Maxwell frontend, IR passes, the WGSL backend; the Ampere
    lowering is header-only), surface formats, texture decoders (block linear, ASTC)."""
    suyu = os.path.join(GPU, "third_party", "suyu")
    out = []
    for d, dirs, files in os.walk(os.path.join(suyu, "shader_recompiler")):
        dirs[:] = [x for x in dirs if x != "glsl"]   # (the GLSL backend: TOTKPort's shader tests' reference only)
        out += [os.path.join(d, f) for f in files if f.endswith(".cpp") and f not in ("attribute.cpp", "flow_test.cpp")]
    vc = os.path.join(suyu, "video_core")
    out += [os.path.join(vc, *f.split("/")) for f in ("surface.cpp", "texture_cache/format_lookup_table.cpp",
                                                      "textures/astc.cpp", "textures/decoders.cpp")]
    render = os.path.join(GPU, "render")
    out += [os.path.join(render, f) for f in os.listdir(render) if f.endswith(".cpp")]
    return sorted(out)


def tree_headers(*roots, skip=()):
    """Every header under these folders, as one dependency blob (suyu's headers sit in many folders); folders named
    in skip are left out"""
    h = hashlib.sha1()
    for root in roots:
        found = []
        # (the walk pruned as it goes: sorting the whole walk first, as this once did, made skip do nothing, and every
        # gpu/vk header change rebuilt the WebGPU renderer)
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in skip)
            found += [os.path.join(d, f) for f in files if f.endswith((".h", ".hpp", ".inc"))]
        for path in sorted(found):
            h.update(os.path.basename(path).encode() + open(path, "rb").read())
    return h.hexdigest().encode()


def vk_sources():
    """The Vulkan back end (hookshot_vk.dll, gpu/vk/vk_api.h): Eden's shader recompiler with its SPIR-V back end
    (gpu/vk/eden), sirit with Eden's mesh shader patch (gpu/vk/sirit) and the back end itself (gpu/vk/*.cpp)"""
    vk = os.path.join(GPU, "vk")
    out = []
    for sub in ("eden", os.path.join("sirit", "src")):
        for d, _, files in os.walk(os.path.join(vk, sub)):
            out += [os.path.join(d, f) for f in files if f.endswith(".cpp")]
    out += [os.path.join(vk, f) for f in os.listdir(vk) if f.endswith(".cpp")]
    vc = os.path.join(GPU, "third_party", "suyu", "video_core")
    out += [os.path.join(vc, "textures", f) for f in ("decoders.cpp", "astc.cpp")]   # (the WebGPU back end's unswizzle)
    return sorted(out)


def cpm_dirs():
    """CPM packages: RX_CPM, packaged tools/cpm, then legacy project caches.
    The packages provide Vulkan and SPIR-V headers, VMA and FFmpeg."""
    eden_ns2 = os.path.dirname(kit.ROOT)
    return [d for d in (os.environ.get("RX_CPM"), os.path.join(kit.ROOT, "cpm"),
                        os.path.join(kit.WORK, "eg", ".cache", "cpm"),
                        os.path.join(eden_ns2, "mkw-recomp", "work", "eg", ".cache", "cpm"),
                        os.path.join(eden_ns2, "Eglantine", ".cache", "cpm")) if d and os.path.isdir(d)]


def cpm_package(name):
    """The newest version of a package in the first CPM cache that has it (cpm_dirs), or None"""
    for root in cpm_dirs():
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.listdir(d):
            return os.path.join(d, sorted(os.listdir(d))[-1])
    return None


def cache_put(obj, key):
    """Hard-link a built object into the cache under its key (a copy across volumes)"""
    os.makedirs(OBJ_CACHE, exist_ok=True)
    dst = os.path.join(OBJ_CACHE, key + ".o")
    if os.path.exists(dst):
        return
    try:
        os.link(obj, dst)
    except OSError:
        import shutil
        shutil.copyfile(obj, dst)


class MemoryStatus(ctypes.Structure):   # (Windows' MEMORYSTATUSEX, for GlobalMemoryStatusEx)
    _fields_ = [("len", ctypes.c_uint32), ("load", ctypes.c_uint32)] + \
               [(f, ctypes.c_uint64) for f in ("total", "avail", "pt", "pa", "vt", "va", "ve")]


class ProcessMemory(ctypes.Structure):   # (Windows' PROCESS_MEMORY_COUNTERS, for GetProcessMemoryInfo)
    _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + \
               [(f, ctypes.c_size_t) for f in ("peak_ws", "ws", "qpp", "qp", "qnpp", "qnp", "pf", "peak_pf")]


# (what one compiler is counted at where its own need is not worked out: a hot piece (0.8-1.7 GB measured, -O2 and the profile),
# and the compile countdown's guess of how many the free memory holds. About the median chunk's peak)
COMPILER_MB = 1500


def free_mb():
    """The memory free now (MemAvailable on Linux, the available physical memory on Windows), in MB; None: not known"""
    try:
        if os.name == "nt":
            m = MemoryStatus()
            m.len = ctypes.sizeof(MemoryStatus)
            return m.avail >> 20 if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) else None
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) >> 10
    except (OSError, ValueError, AttributeError):
        return None
    return None


def tree_rss(pid):
    """(linux) The resident bytes of a process and the processes it started (a driver that runs its compiler as a child, or ccache)"""
    with open("/proc/%d/statm" % pid) as f:
        n = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    for task in os.listdir("/proc/%d/task" % pid):
        try:
            with open("/proc/%d/task/%s/children" % (pid, task)) as f:
                for child in f.read().split():
                    try:
                        n += tree_rss(int(child))
                    except (OSError, ValueError, IndexError):
                        pass
        except OSError:
            pass
    return n


def rss_mb(proc):
    """A running compiler's resident memory in MB (0: not known, or not started yet)"""
    if proc is None:
        return 0
    try:
        if os.name == "nt":
            c = ProcessMemory()
            c.cb = ctypes.sizeof(ProcessMemory)
            ok = ctypes.WinDLL("psapi").GetProcessMemoryInfo(ctypes.c_void_p(int(proc._handle)), ctypes.byref(c), c.cb)
            return c.ws >> 20 if ok else 0
        return tree_rss(proc.pid) >> 20
    except (OSError, ValueError, AttributeError, IndexError):
        return 0


def mem_need(src):
    """What compiling a file takes at its peak, in MB, a little over. Measured on a cold build (clang 23, x86-64): a chunk of lifted
    code 0.4 GB + 50 MB per MB of C + 250 MB per MB of its largest function (1.5 GB for 24 MB of small functions, 2.2 GB for a 6 MB
    function alone); a hot piece 0.8-1.3 GB whatever its size (-O2, the profile, inlining); the runtime's C files ~0.2 GB, the C++
    files up to ~1 GB"""
    name = os.path.basename(src)
    if name.startswith("code_hot"):
        return COMPILER_MB
    if name.startswith("code_"):
        with open(src, "rb") as f:
            text = f.read()
        # (each function from the newline before its definition, as this was first measured)
        starts = [m.start() - 1 for m in lifted_c.HEAD_BYTES.finditer(text)] + [len(text)]
        largest = max([b - a for a, b in zip(starts, starts[1:])] or [0])
        return int(400 + 50 * len(text) / (1 << 20) + 250 * largest / (1 << 20))
    return 1000 if src.endswith(".cpp") else 400


class MemoryGate:
    """(issue 418) Compilers started by the memory there is, not only by --jobs: one per CPU thread ran PCs out of memory (the kernel
    stopped a compiler, or the whole build), and a count low enough for the biggest files and the PC's other programs left most of
    the threads idle. A compile starts when the memory free, less what the compilers running may still take on top of what they
    hold now (mem_need), keeps its own need and HOOKSHOT_MEM_RESERVE_GB (default 1.5) free. The floor is make_game's before this:
    on Windows up to 4 (half the threads on smaller CPUs) always run, as Windows pages rather than fails (issues 502, 503, 511: 1-2
    compilers on 16 GB PCs took hours); on Linux one (issue 531: Linux stops a compiler that does not fit, and it is compiled again
    alone). The free memory is read again as compiles start and end, so a game or browser opened during the build makes later
    compiles wait"""

    def __init__(self):
        self.cv = threading.Condition()
        self.running = {}   # (per-compile lease: [need, Popen or None]; one source can have several objects)
        self.reserve = float(os.environ.get("HOOKSHOT_MEM_RESERVE_GB") or 1.5) * 1024
        self.floor = max(1, min(4, (os.cpu_count() or 2) // 2)) if os.name == "nt" else 1

    def enter(self, src):
        need = mem_need(src)
        with self.cv:
            while len(self.running) >= self.floor:
                free = free_mb()
                if free is None:
                    break
                owed = sum(max(0, n - rss_mb(p)) for n, p in self.running.values())
                if free - owed - self.reserve >= need:
                    break
                self.cv.wait(1.0)
            lease = object()
            self.running[lease] = [need, None]
            return lease

    def started(self, lease, proc):
        with self.cv:
            self.running[lease][1] = proc

    def expected(self, jobs):
        """How many compilers the memory free now is likely to hold at once (COMPILER_MB each), between the floor and the jobs:
        the compile countdown's estimate"""
        free = free_mb()
        if free is None:
            return jobs
        return max(self.floor, min(jobs, int((free - self.reserve) // COMPILER_MB)))

    def leave(self, lease):
        with self.cv:
            self.running.pop(lease, None)
            self.cv.notify_all()


GATE = None   # (main's MemoryGate; None: compiles start as the jobs allow)


FFMPEG = next((os.path.join(d, n) for d in [os.path.join(r, "ffmpeg") for r in cpm_dirs()] if os.path.isdir(d)
               for n in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, n, "lib"))
               and "linux" not in n.lower()), None)   # (hookshot: a cache with both systems' builds, cross-building)
FFMPEG_LIBS = ("avcodec", "swscale", "swresample", "avutil")


def ffmpeg_linux():
    """(linux) FFmpeg's compile and link flags: HOOKSHOT_FFMPEG=<prefix> with static libraries (the AppImage's, with
    dav1d for OoT's AV1 movies), else the system's through pkg-config; None without either"""
    pre = os.environ.get("HOOKSHOT_FFMPEG")
    if pre and os.path.exists(os.path.join(pre, "lib", "libavcodec.a")):
        libs = [os.path.join(pre, "lib", "lib%s.a" % l) for l in FFMPEG_LIBS]
        libs += [os.path.join(pre, "lib", "libdav1d.a")] if os.path.exists(os.path.join(pre, "lib", "libdav1d.a")) else []
        return ["-I" + os.path.join(pre, "include")], libs + ["-lm", "-lpthread"]
    try:
        names = ["lib" + l for l in FFMPEG_LIBS]
        cflags = subprocess.run(["pkg-config", "--cflags"] + names, capture_output=True, text=True, check=True).stdout.split()
        libs = subprocess.run(["pkg-config", "--libs"] + names, capture_output=True, text=True, check=True).stdout.split()
        return cflags, libs
    except (OSError, subprocess.CalledProcessError):
        return None


if LINUX:
    FFMPEG_LX = ffmpeg_linux()
    FFMPEG = None


def ffmpeg_version_key():
    """The FFmpeg headers this build compiles against, so a host object from another major version is not reused.
    Codec numbers and struct layouts differ between them."""
    incs = []
    if FFMPEG:
        incs.append(os.path.join(FFMPEG, "include"))
    if LINUX and FFMPEG_LX:
        incs += [f[2:] for f in FFMPEG_LX[0] if f.startswith("-I")]
    incs.append("/usr/include")
    for inc in incs:
        parts = []
        for name in ("version_major.h", "version.h"):
            path = os.path.join(inc, "libavcodec", name)
            if os.path.exists(path):
                parts.append(open(path, "rb").read())
        if parts:
            return b"".join(parts)
    return b""


# the soft write watch (rt/a64.h A64_MARK): lifted stores mark their 4 KB page, and the renderer reads the marks
# instead of Windows' write watch (GetWriteWatch over the 6 GB MKW's GPU sees took ~20 ms a scan, in one thread or
# four: the race's replay waited ~12 ms a frame for it). RX_SOFT_WRITE_WATCH=0: Windows' write watch (a full rebuild)
WRITE_TRACK = [] if os.environ.get("RX_SOFT_WRITE_WATCH") == "0" else ["-DA64_WRITE_TRACK"]

# (hookshot) the host CPU the lifted game and the runtime are built for: x86-64-v3 (AVX2, FMA, BMI1/2, LZCNT,
# MOVBE; Haswell and Zen 1 on). With FMA the FPX1 fast paths cover double-precision FMA (RECOMP_FPX_FMA64; it took
# the exact software body on baseline x86-64) and the bit counts are single instructions. No contraction, so a C
# expression is rounded as written (the exact bodies and FPX1's proofs assume it). RX_MARCH=x86-64: the old baseline
# (issue 363: a Xeon E5-2667 v2 (Ivy Bridge, no AVX2) built the game and then stopped at its first AVX2 instruction,
# 0xc000001d. The game is built on the PC it runs on: a CPU without x86-64-v3's features gets x86-64-v2, or x86-64)
from host_cpu import host_march   # (make_game.py asks it too: the AppImage's renderer)

MARCH = os.environ.get("RX_MARCH") or host_march()
if MARCH != "x86-64-v3":
    print("build: this CPU has no AVX2: the game is built for %s (a little slower)" % MARCH, flush=True)
CPU = ["-march=" + MARCH, "-ffp-contract=off"]


def flags(gen, rt64, opt, polls):
    return target() + CPU + [
        "-I" + rt64, "-I" + gen, "-I" + os.path.join(kit.KIT, "rt"), "-I" + NX2_RT, "-Wno-everything",
        "-fno-strict-aliasing", "-mcx16", "-" + opt, "-g0", "-mllvm", "-rotation-max-header-size=0",
        "-ffunction-sections", "-mllvm", "-capture-tracking-max-uses-to-explore=5000"] + \
        (["-DA64_POLLING"] if polls else []) + WRITE_TRACK


def write_info(out):
    meta = json.load(open(os.path.join(kit.WORK, "image", "image.json")))
    mods = {m["name"]: m for m in meta["modules"]}
    path = os.path.join(out, "mkw_info.c")
    text = ('/* Generated by nx2/build.py from work/image/image.json. */\n#include <stdint.h>\n'
            'const uint64_t mkw_image_base = 0x%xULL;\nconst char mkw_main_build_id[] = "%s";\n'
            % (meta["base"], mods["main"]["build_id"]))
    if not os.path.exists(path) or open(path).read() != text:
        open(path, "w", newline="\n").write(text)
    return path


def flag_text(fl):
    """The flags an object key is named by. Include folders are left out (their headers are in deps). The builder's
    own folder is <root> (issue 565: versions\\<tag>\\... made every update look new). --sysroot= is <sysroot>: the
    AppImage mount is a new /tmp/.mount_... path every start, and that path is not a change in the headers."""
    kept = []
    for f in fl:
        if f.startswith("-I"):
            continue
        if f.startswith("--sysroot="):
            f = "--sysroot=<sysroot>"
        else:
            f = f.replace(kit.ROOT, "<root>")
        kept.append(f)
    return " ".join(kept)


def object_key(src, fl, deps):
    """What an object depends on: its source, the flags (flag_text), and the headers."""
    return hashlib.sha1(open(src, "rb").read() + flag_text(fl).encode() + deps).hexdigest()


def times_from(path):
    """The seconds each module took on this PC, from make_game.py's build_times.json, or {} when there is no record to
    use (a first build, one half written, or one another version of the tools wrote): the countdown then falls back to
    the rate this PC's other measurements ran at."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data["seconds"].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def eta_left(pending, jobs):
    """How long the rest of the compile takes, in seconds, with this many compilers: the work spread over them, with
    the longest file still left as the floor (each file is still one compiler)."""
    if not pending:
        return 0.0
    return max(max(pending), sum(pending) / max(1, jobs))


def per_file_seconds(jobs, history):
    """One expected duration for each job, in order: the seconds this PC measured for that module before, or, for a
    module it never compiled (a new release adds or renames files), the rate those measurements ran at. On a PC with
    no measurements at all, the first build uses the rates a full build on the development PC showed."""
    sizes = [max(0.0, os.path.getsize(src) / (1 << 20)) for _, src, *_ in jobs]
    measured = [(history[os.path.basename(src)], mb) for (_, src, *_), mb in zip(jobs, sizes)
                if os.path.basename(src) in history and mb > 0]
    rate = sum(secs for secs, _ in measured) / sum(mb for _, mb in measured) if measured else None
    out = []
    for (_, src, *_), mb in zip(jobs, sizes):
        name = os.path.basename(src)
        if name in history:
            out.append(max(0.05, history[name]))
        elif rate is not None:
            out.append(max(0.5, rate * mb))
        else:   # (seconds per MB measured on a full build: the eight hot files, the game chunks, the rest)
            out.append(23.5 * mb if name.startswith("code_hot") else 3.4 * mb + 1 if name.startswith("code_") else 1.3)
    return out


def plan_jobs(jobs):
    """Which jobs this run really compiles, and every job's object key. A job whose object is stamped with its key, or
    whose key is in the object cache (the other target's build), is only linked in: not work, and not in the countdown.
    The keys go to compile_one, which then does not hash every source a second time."""
    work, keys = [], {}
    for cc, src, obj, fl, deps in jobs:
        key = object_key(src, fl, deps)
        keys[(src, obj)] = key
        stamp = obj + ".sha1"
        if os.path.exists(obj) and os.path.exists(stamp) and open(stamp).read() == key:
            continue
        if os.path.exists(os.path.join(OBJ_CACHE, key + ".o")):
            continue
        work.append((cc, src, obj, fl, deps))
    return work, keys


def migrate_stamps(gen, obj_dir, fl, deps):
    """Objects stamped with the old key (flags with include folders) are restamped with object_key and cached, so
    a build from before the cache does not have to be compiled again"""
    if not os.path.isdir(obj_dir):
        return
    if not os.path.isdir(gen) or not os.path.isdir(obj_dir):   # (a first build in a fresh folder)
        return
    for f in os.listdir(gen):
        if not (f.startswith("code_") and f.endswith(".c")):
            continue
        obj = os.path.join(obj_dir, f[:-2] + ".o")
        stamp = obj + ".sha1"
        if not (os.path.exists(obj) and os.path.exists(stamp)):
            continue
        src = os.path.join(gen, f)
        content = open(src, "rb").read()
        if open(stamp).read() == hashlib.sha1(content + " ".join(fl).encode() + deps).hexdigest():
            key = object_key(src, fl, deps)
            open(stamp, "w").write(key)
            cache_put(obj, key)


PLANNED = {}   # (src, obj) -> the object key plan_jobs worked out, so compile_one does not hash every source again


# (hookshot) the order of the compile: the longest first (LPT), so no long one starts late and runs alone at the end. The lift's
# estimate (gen/chunk_costs.txt, nx2/chunks.py fn_cost: functions cost with the square of their size) where it is there, else the size.
# HOT_COST: how much longer the hot flags (-O2, the profile, the hot callees inlined) take for the same estimate. Fitted to a cold build
# on a 24-thread desktop; by size alone the hot pieces (9 MB, up to 333 s) started after every 25 MB chunk (45 s)
HOT_COST = 4


def chunk_costs(gen):
    try:
        with open(os.path.join(gen, "chunk_costs.txt")) as f:
            return {name: int(n) for name, _, n in (line.partition(" ") for line in f) if n.strip().isdigit()}
    except OSError:
        return {}


def compile_cost(src, costs):
    """A file's compile time, roughly (see HOT_COST)"""
    name = os.path.basename(src)
    return costs.get(name, os.path.getsize(src)) * (HOT_COST if name.startswith("code_hot") else 1)


def compile_one(job):
    cc, src, obj, fl, deps = job
    key = PLANNED.get((src, obj)) or object_key(src, fl, deps)
    stamp = obj + ".sha1"
    if os.path.exists(obj) and os.path.exists(stamp) and open(stamp).read() == key:
        return src, 0.0, None
    cached = os.path.join(OBJ_CACHE, key + ".o")
    if os.path.exists(cached):   # (the same source and flags built for the other target)
        if os.path.exists(obj):
            os.remove(obj)
        try:
            os.link(cached, obj)
        except OSError:
            import shutil
            shutil.copyfile(cached, obj)
        open(stamp, "w").write(key)
        return src, 0.0, None
    lease = GATE.enter(src) if GATE else None
    t = time.time()
    try:
        with subprocess.Popen([cc] + fl + ["-c", src, "-o", obj], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
            no_throttle(getattr(p, "_handle", None))
            if GATE:
                GATE.started(lease, p)
            out, err = p.communicate()
    finally:
        if GATE:
            GATE.leave(lease)
    r = subprocess.CompletedProcess(p.args, p.returncode, out, err)
    if r.returncode:
        failure = r.stderr or r.stdout or "Compiler exited with status %d without diagnostics." % r.returncode
        return src, time.time() - t, failure[-4000:]
    open(stamp, "w").write(key)
    cache_put(obj, key)
    return src, time.time() - t, None


def main():
    kit.wait_for_quiet('the compile')
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--only", default="")
    p.add_argument("--opt", default="O1")
    p.add_argument("--no-polls", action="store_true")
    p.add_argument("--dll", action="store_true")
    p.add_argument("--lib", action="store_true")
    p.add_argument("--standalone", action="store_true")
    a = p.parse_args()
    sa = os.path.join(kit.WORK, "sa")
    gen = os.path.join(sa if a.standalone else kit.WORK, "gen")
    out = os.path.join(sa if a.standalone else kit.WORK, "build")
    obj_dir = os.path.join(out, "obj")
    # (hookshot) the start-up screen's logo, assets/hookshot.png as RGBA bytes in a header of the build's own (nx2/logo_rgba.py);
    # its bytes are part of vk_menu.cpp's dependencies
    gen_dir = os.path.join(out, "gen")
    logo_text = logo_rgba.write(os.path.join(gen_dir, "hookshot_logo.h")).encode()
    os.makedirs(obj_dir, exist_ok=True)
    if os.environ.get("RX_VK_ONLY"):   # (the Vulkan back end needs no game: built so with no lift, as the AppImage's is)
        os.makedirs(gen, exist_ok=True)
    rt64 = rtgen.write(os.path.join(out, "rt64"))
    cc = toolchain.need(toolchain.find("clang"))
    fl = flags(gen, rt64, a.opt, not a.no_polls)
    # (which Hookshot this is, for the log's first line and the Insert menu: given to those two files only, so a new
    # release name rebuilds them and nothing else)
    stamp_def = version.define()
    print("version: %s" % version.describe(), flush=True)
    # (the lifted chunks include only the guest model: the emulator ABI header is a dependency of the shim alone)
    deps = b"".join(open(os.path.join(d, f), "rb").read() for d in (gen, rt64) for f in sorted(os.listdir(d))
                    if f.endswith(".h"))
    shim_deps = deps + open(os.path.join(NX2_RT, "mkw_host.h"), "rb").read()
    eden_gen, eden_out = os.path.join(kit.WORK, "gen"), os.path.join(kit.WORK, "build")
    migrate_stamps(eden_gen, os.path.join(eden_out, "obj"),
                   flags(eden_gen, os.path.join(eden_out, "rt64"), a.opt, not a.no_polls), deps)
    migrate_stamps(gen, obj_dir, fl, deps)
    chunks = sorted(f for f in os.listdir(gen) if f.startswith("code_") and f.endswith(".c"))
    if a.only:
        want = {"code_%03d.c" % int(n) for n in a.only.split(",")}
        chunks = [c for c in chunks if c in want]
    srcs = [os.path.join(gen, c) for c in chunks]
    host_srcs = []
    if not a.only:
        srcs += [os.path.join(gen, f) for f in ("a64_table.c", "a64_hooks.c", "a64_suyu.c")] + \
                [os.path.join(rt64, "a64_rt.c")]
        if a.standalone:
            srcs += [os.path.join(gen, f) for f in ("a64_stubs.c", "hos_image.c", "nvn2_exports.c")]
            # (EvilOOT) host/rx_nowgpu.c stands in for the WebGPU renderer only when there is no wgpu-native to build it with (Linux has
            # its own stand-ins, host/posix/linux_stubs.c)
            have_wgpu = os.path.isfile(os.path.join(WGPU, "include", "webgpu", "webgpu.h"))
            host_srcs = sorted(os.path.join(HOST, f) for f in os.listdir(HOST) if f.endswith(".c") and (f != "pgo_rt.c" or PGO == "gen")
                               and not (LINUX and f in ("display.c", "mf_av1.c", "rx_nowgpu.c")) and (f != "rx_nowgpu.c" or not have_wgpu))
            if LINUX:   # (the window on SDL2, the Win32 calls on POSIX, no Media Foundation and no WebGPU renderer)
                host_srcs += [os.path.join(POSIX, f) for f in ("win32_posix.c", "display_sdl.c", "linux_stubs.c", "prof.c")]
        else:
            srcs += [os.path.join(NX2_RT, "eden_shim.c"), write_info(out)]
    if os.environ.get("RX_VK_ONLY"):   # (no lift needed: the back end alone)
        srcs = [s for s in srcs if os.path.exists(s)]
    # the longest first, so the long ones do not end up alone at the end (compile_cost)
    costs = chunk_costs(gen)
    srcs.sort(key=lambda s: -compile_cost(s, costs))
    # (hookshot) code_hot.c, the lift's hot functions together (A64_HOT_FUNCS + A64_HOT_CHUNK=1): at RX_HOT_OPT
    # (default O2), so the hot callees inline into their hot callers
    hot_fl = [("-" + os.environ.get("RX_HOT_OPT", "O2")) if f == "-" + a.opt else f for f in fl]
    # (hookshot) RX_PGO=gen: code_hot.c counts its branches (host/pgo_rt.c writes them; -O1 builds much sooner, and the counters are
    # placed before the optimiser, so a profile from it fits the -O2 build). RX_PGO=<merged .prof>: built with that profile
    if PGO == "gen":
        hot_fl = [f if f != "-O2" else "-O1" for f in hot_fl] + ["-fprofile-instr-generate", "-mllvm", "-disable-vp=true"]
    elif PGO:
        # (the profile's contents in the flags the object key hashes: its path alone stays the same across releases, so an update with a
        # new profile kept the hot code built from the old one)
        with open(PGO, "rb") as f:
            hot_fl = hot_fl + ["-fprofile-instr-use=" + PGO, "-DHOOKSHOT_PROFILE_SHA1=" + hashlib.sha1(f.read()).hexdigest()[:16]]
    jobs = [(cc, s, os.path.join(obj_dir, os.path.basename(s)[:-2] + ".o"),
             hot_fl if os.path.basename(s).startswith("code_hot") else fl,
             shim_deps if os.path.dirname(s) == NX2_RT else deps) for s in srcs]
    # (issue 624) what the game code's object keys are made of, as short fingerprints: two builds' logs side by side say which part
    # changed when an update compiled everything again (the headers, the flags, the profile, or the lifted code itself)
    fp = lambda b: hashlib.sha1(b).hexdigest()[:10]
    norm = lambda f: flag_text(f).encode()
    code = hashlib.sha1()
    for code_src in sorted(x for x in srcs if os.path.basename(x).startswith("code_")):
        code.update(os.path.basename(code_src).encode() + open(code_src, "rb").read())
    print("object keys: game-code headers %s, flags %s, hot flags %s, lifted code %s (%d files)" % (
        fp(deps), fp(norm(fl)), fp(norm(hot_fl)), code.hexdigest()[:10],
        sum(1 for x in srcs if os.path.basename(x).startswith("code_"))), flush=True)
    # the runtime: ordinary C at -O2, with its own headers (any host/*.h change rebuilds all of it)
    host_fl = target() + CPU + (["-I" + POSIX, "-D_GNU_SOURCE"] if LINUX else []) + \
        ["-I" + HOST, "-I" + rt64, "-I" + gen, "-Wno-everything", "-O2", "-g0", "-mcx16", "-DA64_POLLING"] + WRITE_TRACK
    host_deps = deps + b"".join(open(os.path.join(HOST, f), "rb").read() for f in sorted(os.listdir(HOST))
                                if f.endswith(".h"))
    if LINUX:
        host_deps += b"".join(open(os.path.join(POSIX, f), "rb").read() for f in sorted(os.listdir(POSIX)) if f.endswith(".h"))
        host_deps += open(os.path.join(GPU, "vk", "vk_api.h"), "rb").read()
    host_deps += ffmpeg_version_key()
    jobs += [(cc, s, os.path.join(obj_dir, "host_" + os.path.basename(s)[:-2] + ".o"),
              host_fl + ["-I" + os.path.join(OPUS, "include")] +
              (["-I" + os.path.join(FFMPEG, "include")] if FFMPEG else []) +
              (FFMPEG_LX[0] if LINUX and FFMPEG_LX else []) +
              ([stamp_def] if os.path.basename(s) == "hos_main.c" else []), host_deps) for s in host_srcs]
    # libopus 1.5.2 (third_party/opus, from TOTKPort): nn::codec's hardware Opus decoder (host/nn_codec.c)
    # (hookshot: built for the same CPU target as the rest (CPU), so its float code can be vectorized where the target has
    # AVX2: the vendored copy has no x86 intrinsics files of its own)
    if host_srcs:
        oflags = target() + CPU + ["-O2", "-g0", "-Wno-everything", "-DOPUS_BUILD", "-DUSE_ALLOCA",
                                              "-DHAVE_LRINTF", "-DHAVE_LRINT"] +             ["-I" + os.path.join(OPUS, d) for d in ("include", "celt", "silk", "silk/float", "src")]
        for rel in open(os.path.join(OPUS, "sources.txt")).read().split():
            jobs.append((cc, os.path.join(OPUS, rel), os.path.join(obj_dir, "opus_" + rel.replace("/", "_")[:-2] + ".o"),
                         oflags, b"opus-1.5.2"))
    # the renderer (C++20, NDEBUG: no asserts in it or suyu), when a wgpu-native release is there
    wgpu = a.standalone and not a.only and not LINUX and os.path.isfile(os.path.join(WGPU, "include", "webgpu", "webgpu.h"))
    if a.standalone and not a.only and not wgpu and not LINUX:
        print("no wgpu-native at %s (WGPU_NATIVE): building without the renderer" % WGPU)
    if wgpu:
        cxx = toolchain.need(toolchain.find("clang++"))
        gflags = target() + ["-std=c++20", "-O2", "-g0", "-DNDEBUG", "-Wno-everything",
                                              "-I" + os.path.join(GPU, "compat"),
                                              "-I" + os.path.join(GPU, "third_party", "suyu"),
                                              "-I" + os.path.join(WGPU, "include"), "-I" + HOST, "-I" + rt64] + WRITE_TRACK +                  ["-include", os.path.join(GPU, "render", "rx_wgpu_names.h")]   # (rx_* -> wgr_*: host/rx_backend.c)
        gdeps = tree_headers(GPU, HOST, skip=("vk",)) + open(os.path.join(WGPU, "include", "webgpu", "webgpu.h"), "rb").read()
        for src in gpu_sources():
            name = os.path.relpath(src, GPU).replace(os.sep, "_")[:-4]
            jobs.append((cxx, src, os.path.join(obj_dir, "gpu_" + name + ".o"), gflags, gdeps))
    # the Vulkan back end, a DLL of its own (RX_BACKEND=vulkan): built when the Vulkan headers are in the CPM cache
    vk_jobs = []
    vk_headers, spirv_headers = cpm_package("vulkanheaders"), cpm_package("spirv-headers")
    vk_prebuilt = os.environ.get("HOOKSHOT_VK_PREBUILT", "")
    if a.standalone and not a.only and vk_prebuilt:
        print("the Vulkan back end: %s, as built" % vk_prebuilt)
    elif a.standalone and not a.only and vk_headers and spirv_headers:
        cxx = toolchain.need(toolchain.find("clang++"))
        vk = os.path.join(GPU, "vk")
        # (what the shader caches are good for: what translates shaders, not the whole DLL, so a build that changes something else
        # keeps them)
        stamp = hashlib.sha1()
        translator = [os.path.join(d, f) for top in ("eden", "sirit", "compat") for d, _, fs in os.walk(os.path.join(vk, top)) for f in fs]
        for f in sorted(translator) + [os.path.join(vk, n) for n in ("vk_shader.cpp", "vk_bank.cpp", "vk_state.h", "vk_api.h")]:
            with open(f, "rb") as src:
                stamp.update(os.path.relpath(f, vk).replace("\\", "/").encode() + src.read().replace(bytes((13, 10)), bytes((10,))))
        shader_stamp = "-DHOOKSHOT_SHADER_STAMP=0x%sull" % stamp.hexdigest()[:16]
        vflags = target() + CPU + (["-fPIC", "-I" + POSIX, "-D_GNU_SOURCE"] if LINUX else []) + [   # (hookshot: the replay is the frame's limit now)
            # (what the code takes for granted, which newer C++ libraries bring in by themselves and older ones do not: issue 48)
            "-include", "algorithm", "-include", "cstdio", "-include", "cstdlib", "-include", "cstring", "-include", "functional",
            "-include", "string", "-include", "vector",
            "-std=c++20", "-O2", "-g0", "-DNDEBUG", "-Wno-everything", "-DVK_NO_PROTOTYPES", "-DMKWVK_BUILDING",
            *(["-DHOOKSHOT_RELEASE"] if os.environ.get("RX_RELEASE") == "1" else []),   # (the shaders precompiled before the game)
            # (Eden's recompiler before suyu's root: suyu has its own shader_recompiler/ headers, and a Shader::Info
            # of the other layout in some objects is what crashed the descriptor walk. MKWVK_INCLUDES changes the
            # object keys, which leave the include folders out, when this order changes)
            "-DMKWVK_INCLUDES=2",
            "-I" + os.path.join(vk, "compat"), "-I" + os.path.join(GPU, "compat"), "-I" + os.path.join(vk, "eden"),
            "-I" + os.path.join(GPU, "third_party", "suyu"),
            "-I" + os.path.join(vk, "sirit", "include"), "-I" + os.path.join(vk, "sirit", "src"),
            "-I" + os.path.join(spirv_headers, "include"), "-I" + os.path.join(spirv_headers, "include", "spirv", "unified1"),
            "-I" + os.path.join(vk_headers, "include"), "-I" + vk, "-I" + gen_dir,
            "-I" + os.path.join(kit.HERE, "..", "third_party", "streamline", "include")] +             (["-I" + os.path.join(cpm_package("vulkanmemoryallocator"), "include")] if cpm_package("vulkanmemoryallocator") else [])
        # (Eden's recompiler and sirit see only their own headers: a change to the back end's vk_*.h does not
        # rebuild their ~500 files; the back end itself depends on everything, suyu's texture decoders included)
        vk_lib_deps = tree_headers(os.path.join(vk, "eden"), os.path.join(vk, "sirit"), os.path.join(vk, "compat"),
                                   os.path.join(GPU, "compat"))
        vdeps = tree_headers(vk, os.path.join(GPU, "render"), os.path.join(GPU, "compat"),
                             os.path.join(GPU, "third_party", "suyu", "video_core", "textures"))
        for src in vk_sources():
            name = os.path.relpath(src, vk).replace(os.sep, "_")[:-4]
            lib = os.path.relpath(src, vk).split(os.sep)[0] in ("eden", "sirit")
            vk_jobs.append((cxx, src, os.path.join(obj_dir, "vk_" + name + ".o"),
                            vflags + ([shader_stamp] if os.path.basename(src) == "vk_shader.cpp" else []) +
                            ([stamp_def] if os.path.basename(src) == "vk_menu.cpp" else []),
                            vk_lib_deps if lib else vdeps + (logo_text if os.path.basename(src) == "vk_menu.cpp" else b"")))
        if LINUX:   # (its own copy of the Win32 calls on POSIX, and the bit of GDI the Insert menu draws text with)
            cflags = target() + CPU + ["-fPIC", "-O2", "-g0", "-Wno-everything", "-D_GNU_SOURCE", "-I" + POSIX,
                                       "-I" + os.path.join(kit.HERE, "..", "third_party", "stb")]
            pdeps = tree_headers(POSIX)
            for f in ("win32_posix.c", "gdi_posix.c"):
                vk_jobs.append((cc, os.path.join(POSIX, f), os.path.join(obj_dir, "vk_posix_" + f[:-2] + ".o"), cflags, pdeps))
        jobs += vk_jobs
    vk_only = bool(os.environ.get("RX_VK_ONLY"))   # (the renderer alone, the game's objects as they are)
    if vk_only:
        jobs = list(vk_jobs)
    # (the builder's compile timer: HOOKSHOT_TIMES is make_game.py's record of the earlier builds on this PC. With it,
    # this run says which files it really rebuilds and how long each took before, and the builder shows the rest of the
    # compile as a countdown that follows this PC's own measurements. Run by hand, it prints none of these lines.)
    global GATE
    GATE = None if os.environ.get("HOOKSHOT_MEM_GATE") == "0" else MemoryGate()
    eta_jobs = GATE.expected(a.jobs) if GATE else a.jobs   # (the countdown with the compilers the memory holds, not --jobs)
    counted, expected, pending = os.environ.get("HOOKSHOT_TIMES"), {}, []
    if counted:
        work_jobs, keys = plan_jobs(jobs)
        PLANNED.update(keys)
        estimate = per_file_seconds(work_jobs, times_from(counted))
        pending = list(estimate)
        for job, seconds in zip(work_jobs, estimate):
            expected.setdefault(job[1], []).append(seconds)
        print("@@plan %d %d" % (len(work_jobs), len(jobs)), flush=True)
        print("@@eta %d" % eta_left(pending, eta_jobs), flush=True)
    print("%d files, %d jobs, %s" % (len(jobs), a.jobs, " ".join(f for f in fl if f.startswith("-D") or f.startswith("-O"))),
          flush=True)
    t0, failed = time.time(), []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        # (as each ends, not in order: the builder app shows progress from these lines, and the biggest file comes first)
        submitted = {ex.submit(compile_one, job): job for job in jobs}
        for future in concurrent.futures.as_completed(submitted):
            src, secs, err = future.result()
            size = os.path.getsize(src) >> 20
            if err:
                failed.append(submitted[future])
                print("not compiled yet: %s (%d MB, %.0fs), tried again alone at the end:\n%s"
                      % (os.path.basename(src), size, secs, err[-1500:]), flush=True)
            else:
                print("  %-16s %4d MB  %6.0fs  (%.0fs)" % (os.path.basename(src), size, secs, time.time() - t0), flush=True)
            if counted:
                print("@@file %s %.1f %s" % (os.path.basename(src), secs,
                                             "failed" if err else "cached" if secs <= 0 else "new"), flush=True)
                if not err and expected.get(src):
                    pending.remove(expected[src].pop())
                    print("@@eta %d" % eta_left(pending, eta_jobs), flush=True)
    # (a compiler that ran out of memory beside the others dies "due to signal": a player's 24 MB code_021.c, 2026-11-08. The
    # files that failed are compiled again one at a time, with all the memory, before the build gives up)
    still = 0
    for job in failed:
        src, secs, err = compile_one(job)
        size = os.path.getsize(src) >> 20
        if err:
            still += 1
            print("FAILED %s (%d MB, %.0fs, alone too):\n%s" % (os.path.basename(src), size, secs, err), flush=True)
        else:
            print("  %-16s %4d MB  %6.0fs  (%.0fs, alone)" % (os.path.basename(src), size, secs, time.time() - t0), flush=True)
        if counted:
            print("@@file %s %.1f %s" % (os.path.basename(src), secs, "failed" if err else "again"), flush=True)
            if not err and expected.get(src):
                pending.remove(expected[src].pop())
                print("@@eta %d" % eta_left(pending, eta_jobs), flush=True)
    print("compiled in %.0fs, %d failed" % (time.time() - t0, still), flush=True)
    if still:
        return 1
    vk_objs = {j[2] for j in vk_jobs}
    objs = [j[2] for j in jobs if j[2] not in vk_objs]
    rsp = os.path.join(out, "objs.rsp")
    open(rsp, "w", encoding="utf-8").write("\n".join('"%s"' % o.replace("\\", "/") for o in objs))
    if a.lib:
        lib = os.path.join(out, "hookshot_game.lib")
        subprocess.run([toolchain.lld_link(), "/lib", "/out:" + lib, "@" + rsp], check=True)
        print("wrote %s (%d objects, %d MB)" % (lib, len(objs), os.path.getsize(lib) >> 20))
    if a.standalone and not a.only and not vk_only and LINUX:
        exe = os.path.join(out, EXE)   # (+ hookshot.map: crash offsets -> f_<guest address>)
        t = time.time()
        # (--export-dynamic-symbol: the window's state, which the Vulkan back end's own Win32 calls read by name)
        r = subprocess.run([cc] + target() + ["-fuse-ld=lld", "-o", exe, "-Wl,-Map=" + exe + ".map", "-Wl,--gc-sections",
                                              "-Wl,--export-dynamic-symbol=hos_posix_*", "@" + rsp] + HOT_ORDER() +
                           (["-fprofile-instr-generate"] if PGO == "gen" else []) +
                           (FFMPEG_LX[1] if FFMPEG_LX else []) + HOST_LIBS,
                           capture_output=True, text=True)
        undefined = sorted({l.strip() for l in (r.stdout + r.stderr).splitlines() if "undefined symbol" in l})
        if r.returncode:
            print("link FAILED (%.0fs), %d undefined symbols" % (time.time() - t, len(undefined)))
            print("\n".join(undefined[:60]) if undefined else (r.stdout + r.stderr)[-3000:])
            return 1
        print("linked %s: %d MB (%.0fs)" % (exe, os.path.getsize(exe) >> 20, time.time() - t))
        sdl = os.environ.get("HOOKSHOT_SDL2", "")   # (libSDL2-2.0.so.0 beside it, when the AppImage brings one)
        if sdl and os.path.exists(sdl):
            import shutil
            shutil.copyfile(sdl, os.path.join(out, "libSDL2-2.0.so.0"))
    elif a.standalone and not a.only and not vk_only:
        exe = os.path.join(out, "hookshot.exe")   # (+ hookshot.map: crash offsets -> f_<guest address>)
        t = time.time()
        r = subprocess.run([cc] + toolchain.native_target() + toolchain.native_link() +
                           ["-o", exe, toolchain.link_opt("/map:" + exe[:-4] + ".map"), toolchain.link_opt("/stack:8388608"), "@" + rsp] + HOT_ORDER() + HOST_LIBS +
                           ([toolchain.link_opt("/nodefaultlib:clang_rt.profile.lib")] if PGO == "gen" else []) +
                           ([os.path.join(kit.WORK, "hookshot.res")] if os.path.exists(os.path.join(kit.WORK, "hookshot.res")) else []) +   # (nx2/make_icon.py)
                           [win_manifest.write(os.path.join(out, "hookshot_manifest.res"))] +   # (UTF-8 paths: nx2/win_manifest.py)
                           ([toolchain.link_opt("/subsystem:windows"), toolchain.link_opt("/entry:mainCRTStartup")] if os.environ.get("RX_RELEASE") == "1" else []) +   # (no console window)
                           ([os.path.join(WGPU, "lib", "wgpu_native.dll.lib")] if wgpu else []) +
                           ([os.path.join(FFMPEG, "lib", l + ".lib") for l in ("avcodec", "swscale", "swresample", "avutil")] +
                            ["-lbcrypt", "-lmfplat", "-lmfuuid", "-lstrmiids", "-lole32"] if FFMPEG else []) +
                           (["-lc++", "-lunwind"] if toolchain.mingw() else []),   # (llvm-mingw: the C++ parts' runtime, linked in; MSVC's comes by itself)
                           capture_output=True, text=True)
        undefined = sorted({l.strip() for l in (r.stdout + r.stderr).splitlines() if "undefined symbol" in l})
        if r.returncode:
            print("link FAILED (%.0fs), %d undefined symbols" % (time.time() - t, len(undefined)))
            print("\n".join(undefined[:60]) if undefined else (r.stdout + r.stderr)[-3000:])
            return 1
        print("linked %s: %d MB (%.0fs)" % (exe, os.path.getsize(exe) >> 20, time.time() - t))
        if wgpu:   # (wgpu_native.dll beside it; in use while the game runs)
            import shutil
            dll, dst = os.path.join(WGPU, "lib", "wgpu_native.dll"), os.path.join(out, "wgpu_native.dll")
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(dll):
                shutil.copyfile(dll, dst)
        # (hookshot) a shared FFmpeg's DLLs beside it (an LGPL shared build, as cross builds find: the static one needs none)
        for l in FFMPEG_LIBS if FFMPEG and os.path.isdir(os.path.join(FFMPEG, "bin")) else ():
            import shutil
            for f in os.listdir(os.path.join(FFMPEG, "bin")):
                if f.startswith(l + "-") and f.endswith(".dll"):
                    shutil.copyfile(os.path.join(FFMPEG, "bin", f), os.path.join(out, f))
        # (hookshot) SDL2.dll beside it: host/pad_sdl.c's controllers (DualSense, DualShock 4, Switch Pro ...)
        sdl = os.path.join(kit.HERE, "..", "third_party", "sdl2", "SDL2.dll")
        sdl_dst = os.path.join(out, "SDL2.dll")
        if os.path.exists(sdl) and (not os.path.exists(sdl_dst) or os.path.getsize(sdl_dst) != os.path.getsize(sdl)):
            import shutil
            shutil.copyfile(sdl, sdl_dst)
        # (SDL2.dll is sdl2-compat: the SDL2 calls on SDL3, whose controller support is current (a Steam Controller user's
        # report). It loads SDL3.dll from beside it)
        sdl3 = os.path.join(kit.HERE, "..", "third_party", "sdl2", "SDL3.dll")
        sdl3_dst = os.path.join(out, "SDL3.dll")
        if os.path.exists(sdl3) and (not os.path.exists(sdl3_dst) or os.path.getsize(sdl3_dst) != os.path.getsize(sdl3)):
            import shutil
            shutil.copyfile(sdl3, sdl3_dst)
    if vk_prebuilt and a.standalone and not a.only:
        import shutil
        shutil.copyfile(vk_prebuilt, os.path.join(out, VK_LIB))
    if vk_jobs:   # (hookshot_vk.dll beside the exe: loaded by host/rx_backend.c with RX_BACKEND=vulkan)
        vdll = os.path.join(out, VK_LIB)
        vrsp = os.path.join(out, "vk_objs.rsp")
        open(vrsp, "w", encoding="utf-8").write("\n".join('"%s"' % j[2].replace("\\", "/") for j in vk_jobs))
        t = time.time()
        if LINUX:   # (its C++ runtime inside: the game folder runs where the AppImage's sysroot is older than the system)
            link = [toolchain.need(toolchain.find("clang++"))] + target() + [
                "-fuse-ld=lld", "-shared", "-o", vdll, "-Wl,-Map=" + vdll + ".map", "-Wl,-Bsymbolic", "-Wl,--gc-sections",
                "-static-libstdc++", "-static-libgcc", "@" + vrsp, "-ldl", "-lpthread", "-lm"]
        else:
            link = [cc] + toolchain.native_target() + toolchain.native_link() + \
                ["-shared", "-o", vdll, toolchain.link_opt("/map:" + vdll[:-4] + ".map"), "@" + vrsp, "-luser32", "-lgdi32",
                 "-lwintrust", "-lcrypt32", "-lversion"] + (["-lc++", "-lunwind"] if toolchain.mingw() else [])
        r = subprocess.run(link, capture_output=True, text=True)
        if r.returncode:
            undefined = sorted({l.strip() for l in (r.stdout + r.stderr).splitlines() if "undefined symbol" in l})
            print("%s link FAILED, %d undefined symbols" % (VK_LIB, len(undefined)))
            print("\n".join(undefined[:60]) if undefined else (r.stdout + r.stderr)[-3000:])
            return 1
        print("linked %s: %d MB (%.0fs)" % (vdll, os.path.getsize(vdll) >> 20, time.time() - t))
    if a.dll and not a.only:
        dll = os.path.join(out, "hookshot_game.dll")   # (+ hookshot_game.map: crash offsets -> f_<guest address>)
        t = time.time()
        r = subprocess.run([cc] + toolchain.native_target() + toolchain.native_link() +
                           ["-shared", "-o", dll, toolchain.link_opt("/map:" + dll[:-4] + ".map"), "@" + rsp],
                           capture_output=True, text=True)
        undefined = sorted({l.strip() for l in (r.stdout + r.stderr).splitlines() if "undefined symbol" in l})
        if r.returncode:
            print("link FAILED (%.0fs), %d undefined symbols" % (time.time() - t, len(undefined)))
            print("\n".join(undefined[:40]) if undefined else (r.stdout + r.stderr)[-3000:])
            return 1
        print("linked %s: %d MB, no undefined symbols (%.0fs)" % (dll, os.path.getsize(dll) >> 20, time.time() - t))
    return 0


if __name__ == "__main__":
    sys.exit(main())

