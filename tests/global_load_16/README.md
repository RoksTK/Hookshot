# SPIR-V global narrow load/store regression

This fixture builds a small `IR::Program`, runs the real shader-info pass and
`EmitSPIRV`, then checks both the emitted instructions and the generated module
with `spirv-val`. It covers signed and unsigned halfword loads at naturally
aligned offsets, preservation of high guest-address bits, and a page-end access
at byte offset `0xffe` whose helper word base is `0xffc`. Narrow 8- and 16-bit
stores use an aligned-word compare-exchange loop and bit-field insert. A
mesh-stage load and store preserve the upstream regression cases. The existing
zero-result behavior without 64-bit integer support is checked too.

Eight single-storage-buffer scenarios trace the actual called U32 helper's
range branch. Its logical end must use the original descriptor address plus
size, then be capped by `OpArrayLength`. Concrete cbuf/descriptor inputs cover
a hit at `0x1004` with size 2 and alignment 16, a miss at `0x1008`, an aligned-base
control, short and zero-length descriptor caps, signed/unsigned loads and a
high 64-bit base. These eight scenarios and the eleven upstream cases emit
nineteen modules. Two more vertex fixtures check that the 32-bit and
128-bit load helpers default to `DontInline` and that the runtime workaround
switches only the 128-bit helper to `Inline`.

The additional `--cas-loops OUTPUT_DIRECTORY` case emits one real compute IR
program with shared/storage byte writes, a storage/global float atomic add and
a global compare-and-swap. With `RX_VK_FOLLOW_STORES=1`, it checks seven emitted
retry loops across the SSBO and page-table routes. Each counter must be a leading
entry-block variable, reset before its loop, incremented once per failed or
successful exchange, and tested against 1024 alongside the actual exchange
success. The failed-attempt path must feed back the returned word and retry.
The two alternative single-exchange global CAS routes must return directly,
without being counted as retry loops. CI validates all twenty-two modules.

The fixture checks the emitted SPIR-V dependency chain; it does not execute a
shader or driver, and does not validate odd guest addresses or AMD hardware. It
tests the SPIR-V backend's lowering contract only.
The workflow checks out the pinned SPIR-V-Headers revision and uses the system
SPIR-V Tools validator with `--uniform-buffer-standard-layout`: the single-SSBO
fixture profiles use four-byte array strides in a uniform block. This validation
option describes those fixture profiles; it does not test device feature enablement.

Reproduce it with Clang in C++20 mode, as in the Linux workflow. The focused
target includes project compatibility headers with a function-local static
inside a `constexpr` assertion macro. GCC rejects that construct in C++20,
while the project's production Linux compiler is Clang in C++20 mode. The
fixture does not establish an unmodified MSVC build.
To include the CAS case when reproducing manually, run the built fixture again
with `RX_VK_FOLLOW_STORES=1` and `--cas-loops` before validating the output modules.
