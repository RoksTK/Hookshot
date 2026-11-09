"""Compile and run the rule for which pages a remade pool's mirror takes from a retired one (gpu/vk/mirror_inherit.h; no game/GPU)."""
import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_source():
    """mirror_make still copies the retired mirrors through the rule, and the switch back is still read"""
    src = (ROOT / "gpu/vk/vk_memory.cpp").read_text(encoding="utf-8")
    for needle in ('#include "mirror_inherit.h"', "mirror_inherit::runs(a, b, r.lo, p->lo, r.synced_gen",
                   'getenv("RX_VK_INHERIT_ALL")', "region_gen[region] <= r.synced_gen"):
        if needle not in src:
            raise SystemExit("vk_memory.cpp no longer has: " + needle)
    make = src[src.index("bool mirror_make(Pool& pool) {"):]
    make = make[:make.index("\n}\n")]
    if "const VkBufferCopy part{a - r.lo, a - p->lo, b - a};" in make:
        raise SystemExit("mirror_make copies a retired mirror's whole part again")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    check_source()
    with tempfile.TemporaryDirectory(prefix="mirror-inherit-") as tmp:
        exe = Path(tmp) / "mirror-inherit"
        subprocess.run([args.cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                        str(ROOT / "tests/test_mirror_inherit.cpp"), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
