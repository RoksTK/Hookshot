// The Insert menu. Drawn on the CPU (GDI text into a bitmap) and copied over the window's image at the last step of a present,
// so it costs nothing while closed and it never touches the game's picture, nor what frame interpolation sees. Keys: Insert
// opens and closes it, Up and Down choose, Left and Right change, Esc closes; the game gets no keyboard input while it is open.
// (hookshot) A gamepad's Home button, or Select and Start together, opens it too; the D-pad chooses and changes, the bottom
// face button changes too, the right one closes. The game gets no gamepad input while it is open either.
// Its Controls page binds the Switch's buttons to gamepad inputs and keys.
// (hookshot) The mouse is free while it is open (the window lets it go: WM_APP + 8, host/display.c and
// host/posix/display_sdl.c): the row under the cursor is selected, a click on a tab shows it, on a value's left or right
// half changes it down or up, on a toggle flips it, on a slider's track sets it (or drag it); the wheel changes the value
// under the cursor, else moves the selection; a right click is Esc. On the Cheats page and its lists a click is Enter (on a
// choice's or an amount's left or right half, Left or Right) and a right click goes back. vk_menu_hit.h has the layout.
// (hookshot) The rows that take a restart (Row::restart) say "(after restart)" while their value is not the one the game
// started with, and the Graphics tab lists them. Closing the menu then asks whether to restart now (menu_restart.h):
// Later keeps the saved settings for the next start and is not asked again until they change; Restart now saves and
// closes the window as the player would, and the game starts again (WM_APP + 9: host/hos_relaunch.c).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

#include "vk_paths.h"
#include "vk_menu.h"
#include "vk_menu_hit.h"
#include "vk_draw.h"   // (pipeline_cache_save: Restart now keeps what was built)
#include "oot_cheat_items.h"
#include "vk_startup_paint.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)
#include "vk_dlss.h"
#include "vk_reflex.h"
#include "upscaler_rows.h"
#include "menu_restart.h"
#ifndef HOOKSHOT_VERSION   // (nx2/build.py gives it to this file: nx2/version.py)
#define HOOKSHOT_VERSION "unknown"
#endif

namespace mkwvk {
HWND app_window();

Options opts;

namespace {

enum MenuTab {
    TAB_GRAPHICS = 0,
    TAB_GAMEPLAY = 1,
    TAB_CONTROLS = 2,
    TAB_COUNT = 3,
    TAB_MOTION = 3   // (hookshot: not a tab of the strip, the Motion page that the Controls tab's Motion row opens)
};

struct Row {
    MenuTab tab;
    const char* name;
    int* value;
    int lo, hi, step;
    const char* (*text)(int v);
    const char* help;   // (shown under the list while the row is selected: what the row is, the same whatever its value)
    // (a line about the value chosen, under the help, or nullptr: hint(v, -1) is the one to show now for v, hint(v, 0),
    // hint(v, 1), ... every one v can have (those that depend on the other rows or on the GPU), until nullptr, so the
    // menu keeps the height of the tallest: measure_help)
    const char* (*hint)(int v, int variant) = nullptr;
    bool restart = false;   // (read once, at the start: a change applies after a restart)
    bool preset_exempt = false;   // (a change here leaves the Quality preset alone: the controls, the window, the counter)
};
/// A row's hints from a table, the value lo first (nullptr: none for that value)
const char* hint_in(const char* const* t, int n, int lo, int v, int variant) {
    return variant <= 0 && v - lo >= 0 && v - lo < n ? t[v - lo] : nullptr;
}
#define MENU_HINTS(fn, lo, ...)                                                                                              \
    const char* fn(int v, int variant) {                                                                                     \
        static const char* const t[] = {__VA_ARGS__};                                                                        \
        return hint_in(t, (int)(sizeof t / sizeof t[0]), lo, v, variant);                                                    \
    }
const char* tx_framegen(int v) {
    static const char* t[] = {"Off", "", "2x", "3x", "4x", "5x", "6x", "7x", "8x"};
    return t[v];
}
const char* tx_onoff(int v) { return v ? "On" : "Off"; }
// (hookshot) Motion controls: the choice, and whether a connected controller has a gyroscope (host/pad_sdl.c). Either can be
// chosen whatever is connected: a pad plugged in later is read when it comes
const char* tx_gyro(int v) {
    const int status = host && host->gyro_status ? host->gyro_status() : -1;
    if (status == 2) return v ? "On (gyro found)" : "Off (gyro found)";
    if (status == 1) return v ? "On (no gyro on this pad)" : "Off (no gyro on this pad)";
    if (status == 0) return v ? "On (no controller)" : "Off (no controller)";
    return v ? "On" : "Off";
}
const char* tx_mousecam(int v) { return v ? "Native" : "As the stick"; }
const char* tx_sens(int v) {
    static char b[12];
    snprintf(b, sizeof b, "%.2f", v / 100.0);
    return b;
}
// (the game's picture and the window it is shown in, as the last present had them: menu_note_present_sizes)
std::atomic<uint64_t> present_sizes{0};
bool filter_needed() {
    const uint64_t v = present_sizes.load(std::memory_order_relaxed);
    return upscaler_rows::filter_needed((unsigned)(v >> 48), (unsigned)(v >> 32) & 0xffff, (unsigned)(v >> 16) & 0xffff,
                                        (unsigned)v & 0xffff);
}
const char* tx_upscale(int v) {
    static const char* t[] = {"Off", "Catmull-Rom", "Lanczos"};
    if ((unsigned)v >= sizeof t / sizeof t[0]) return t[0];   // (a setting past the row: the label, not the next bytes)
    return t[v];
}
/// The Window scaling filter row's value: its filter, or that none is needed (the saved value stays, for another window size)
const char* tx_filter(int v) { return filter_needed() ? tx_upscale(v) : "Not needed (window = output)"; }
const char* tx_reflex(int v) {
    static const char* t[] = {"Off", "On", "On + Boost"};
    return t[reflex_available() ? std::clamp(v, 0, 2) : 0];
}
/// Reflex: on the value, or that it is not available here (the row shows Off)
const char* hn_reflex(int v, int variant) {
    static const char* const t[] = {"The game as without it.", "The driver paces the frames for less latency.",
                                    "As On, and the GPU's clocks are kept up as well."};
    static const char* const none = "Not available here: it needs an NVIDIA GPU whose driver has VK_NV_low_latency2.";
    if (variant < 0) return reflex_available() ? t[std::clamp(v, 0, 2)] : none;
    return variant == 0 ? t[std::clamp(v, 0, 2)] : variant == 1 ? none : nullptr;
}
// The Upscaler (opts.dlss): what scales the 3D up to the output. DLSS and DLAA only where dlss_why() lets them run; the menu
// shows (and steps from) what runs, dlss_mode(), and keeps the saved choice for a PC where it can
namespace ur = upscaler_rows;
int upscaler() { return dlss_mode(); }
const char* tx_upscaler(int) {
    static const char* t[] = {"Game's own", "NVIDIA DLSS", "NVIDIA DLAA"};
    return t[std::clamp(upscaler(), 0, 2)];
}
/// The Upscaler: the one shown (what runs, upscaler()), and with the game's own, why DLSS and DLAA are not offered
const char* hn_upscaler(int v, int variant) {
    static const char* const t[] = {
        "The game's own network.",
        "NVIDIA's, from a smaller 3D. Softer than the game's own: set Sharpness to around 80%. Things that move by "
        "themselves can trail for now.",
        "NVIDIA's, the 3D at full size with its edges smoothed; the slowest. Softer than the game's own: set Sharpness to "
        "around 80%. Things that move by themselves can trail for now.",
    };
    static const char* const why[DLSS_WHY_COUNT] = {
        nullptr,
        "DLSS and DLAA: not in this build (built without NVIDIA's DLSS SDK, see the readme).",
        "DLSS and DLAA: off for this run (RX_VK_DLSS=0).",
        "DLSS and DLAA: need an NVIDIA RTX GPU.",
        "DLSS and DLAA: need a newer NVIDIA driver.",
        "DLSS and DLAA: DLSS's runtime was not found (the game folder's ngx folder, see the readme).",
    };
    // (now: the value shown and what runs here; every one: each value, and the game's own with each reason)
    int w = 0;
    if (variant < 0) v = std::clamp(upscaler(), 0, 2), w = (int)dlss_why();
    else if (v == ur::GAME) w = variant;
    else if (variant > 0) return nullptr;
    v = std::clamp(v, 0, 2);
    if (w >= DLSS_WHY_COUNT) return nullptr;
    if (v != ur::GAME || !why[w]) return t[v];
    static char b[DLSS_WHY_COUNT][256];
    snprintf(b[w], sizeof b[w], "%s %s", t[0], why[w]);
    return b[w];
}
/// The DLSS model row (opts.dlss_model): NVIDIA's network by its preset, or RX_VK_DLSS_PRESET's for this run
const char* dlss_preset_env() {
    static const char* const e = getenv("RX_VK_DLSS_PRESET");
    return ur::preset_env(e) ? e : nullptr;
}
const char* tx_dlss_model(int v) {
    static const char* const t[] = {"Transformer (K)", "Transformer 2 (M)", "Transformer 2 (L)", "NVIDIA default"};
    if (const char* e = dlss_preset_env()) {
        static char b[48];
        const unsigned p = ur::dlss_preset(v, e);
        if (p) snprintf(b, sizeof b, "Preset %c (RX_VK_DLSS_PRESET)", ur::preset_letter(p));
        else snprintf(b, sizeof b, "NVIDIA default (RX_VK_DLSS_PRESET)");
        return b;
    }
    return t[std::clamp(v, 0, ur::kModels - 1)];
}
/// The DLSS model: the value chosen; variant 1: the game's own upscales (the row waits for DLSS or DLAA); variant 2: set for
/// this run by RX_VK_DLSS_PRESET
const char* hn_dlss_model(int v, int variant) {
    static const char* const t[] = {
        "NVIDIA's transformer model: the one tested most here.",
        "NVIDIA's second-generation transformer, its pick for Performance: sharper, at a little more GPU time.",
        "The second-generation transformer as NVIDIA tuned it for Ultra Performance, the smallest Render resolutions.",
        "Lets DLSS's runtime pick for each mode: what it picks can change with a runtime update.",
    };
    static const char* const unused = "Used by NVIDIA DLSS and DLAA: choose one in the Upscaler row.";
    static const char* const env = "Set for this run by RX_VK_DLSS_PRESET.";
    if (variant < 0) variant = dlss_preset_env() ? 2 : upscaler() == ur::GAME ? 1 : 0;
    if (variant == 1) return unused;
    if (variant == 2) return env;
    return variant == 0 ? t[std::clamp(v, 0, ur::kModels - 1)] : nullptr;
}
const char* tx_percent(int v) {
    static char b[8];
    snprintf(b, sizeof b, "%d%%", v);
    return b;
}
// The output the menu's sizes are shown for: the Output resolution row's (applied at the next start), so the rows below follow a change
// at once; the handheld profile runs the game's own (1536x864)
struct Out {
    uint32_t w, h;
};
constexpr Out kOutputs[] = {{2048, 1152}, {2560, 1440}, {3840, 2160}, {1920, 1080}};
constexpr int kLastOutput = std::size(kOutputs) - 1;
int output_choice(int v) { return v >= 0 && v <= kLastOutput ? v : 0; }
// Display the sizes from smallest to largest while keeping the IDs used by saved settings.
constexpr int kOutputOrder[] = {3, 0, 1, 2};
static_assert(std::size(kOutputOrder) == std::size(kOutputs));
int step_output_choice(int id, int dir) {
    const int* pos = std::find(std::begin(kOutputOrder), std::end(kOutputOrder), output_choice(id));
    const int next = std::clamp((int)(pos - std::begin(kOutputOrder)) + dir, 0, kLastOutput);
    return kOutputOrder[next];
}
Out shown_output() {
    return opts.handheld ? Out{1536, 864} : kOutputs[output_choice(opts.output_res)];
}
// The size the game draws the 3D at for a fixed scale of its output, a twelfth of it: every twelfth asked came out exact at the three
// outputs (2048x1152: 0.58 drew 1195x672, 0.6667 1366x768). So the scale the game is given is the twelfth too: 0.67 drew 2615x1471 at
// 3840x2160, not the 2560x1440 of 8/12.
int twelfths(float scale) { return std::min(std::max((int)(scale * 12.0f + 0.5f), 1), 12); }
void game_size(float scale, uint32_t& w, uint32_t& h) {
    const Out o = shown_output();
    const int k = twelfths(scale);
    w = (o.w * k + 11) / 12, h = o.h * k / 12;
}
const char* tx_resolution(int v) {
    static char b[48];
    const int up = upscaler();
    if (up == ur::DLAA) return "Native (DLAA)";
    if (!ur::percent(v)) return "Auto (dynamic)";
    if (const char* mode = up == ur::DLSS ? ur::dlss_mode_name(v) : nullptr) {
        snprintf(b, sizeof b, "%s (%d%%)", mode, ur::percent(v));
        return b;
    }
    uint32_t w, h;
    game_size(ur::percent(v) / 100.0f, w, h);
    snprintf(b, sizeof b, "%d%% (%ux%u)", ur::percent(v), w, h);
    return b;
}
/// The Render resolution: the value chosen, for the Upscaler shown (every one: each value with each Upscaler)
const char* hn_resolution(int v, int variant) {
    const int up = variant < 0 ? std::clamp(upscaler(), 0, 2) : variant;
    if (up > 2) return nullptr;
    if (up == ur::DLAA)
        return "DLAA draws the 3D at full size: nothing to set here. Choose DLSS or the game's own upscaler to draw it "
               "smaller.";
    const int p = ur::percent(v);
    if (up == ur::DLSS) {
        switch (p) {
        case 0: return "The game lowers it when the GPU is busy, and DLSS follows. Set Sharpness to around 80%.";
        case 67: return "DLSS's sharpest mode. Set Sharpness to around 80%.";
        case 58: return "Between Quality and Performance. Set Sharpness to around 80%.";
        case 50: return "Faster, and softer. Set Sharpness to around 80%.";
        case 33: return "DLSS's fastest mode, and its softest. Set Sharpness to around 80%.";
        default: return nullptr;   // (not one of DLSS's steps: the row moves to one)
        }
    }
    if (!p) return "The game lowers it when the GPU is busy.";
    if (p == 100) return "The 3D at the output's full size: the sharpest, and the slowest.";
    return "A steady size: lower is faster, and softer.";
}
/// A row with nothing else to choose here: shown without its arrows, Left and Right leave it (and the saved value) alone
bool row_fixed(const int* value) {
    return (value == &opts.reflex && !reflex_available()) || (value == &opts.dlss && ur::upscaler_fixed(dlss_why() == DLSS_OK)) ||
           (value == &opts.resolution && ur::resolution_fixed(upscaler())) || (value == &opts.upscale && !filter_needed()) ||
           (value == &opts.dlss_model && ur::model_fixed(upscaler(), dlss_preset_env() != nullptr));
}
const char* tx_output_res(int v) {
    static char b[32];
    v = output_choice(v);
    const Out o = kOutputs[v];
    snprintf(b, sizeof b, "%ux%u%s", o.w, o.h, v ? "" : " (game's)");
    return b;
}
const char* tx_speed(int v) { return v ? "Real time" : "Per frame"; }
const char* tx_text_speed(int v) {
    static const char* t[] = {"Normal", "2x", "4x", "Instant"};
    if ((unsigned)v >= sizeof t / sizeof t[0]) return t[0];
    return t[v];
}
const char* tx_ultrawide(int v) { return v ? "Monitor's shape" : "16:9"; }
const char* tx_gpu(int v) {
    static char buf[300];
    const int idx = v - 1;
    if (v <= 0) snprintf(buf, sizeof buf, "Auto");
    else if (idx < (int)gpu_device_names.size()) snprintf(buf, sizeof buf, "%d: %s", idx, gpu_device_names[idx].c_str());
    else snprintf(buf, sizeof buf, "%d: Device %d", idx, idx);
    // (the whole name shows: the selected row slides through it, see marquee_pos)
    return buf;
}
const char* tx_pad_layout(int v);
// (hookshot) the Microphone row: Off, the system's default, or one of the host's microphones (host/nn_audio_in.c: the
// ocarina's input streams), told to the host by its name (host->mic_select) whenever it changes. The chosen one is kept by
// its name, mic_want, also while it is unplugged: the row then has one more entry for it, "<name> (not connected)", the
// host lets the default stand in and opens it again once it is back, and the name stays in the ini. In the ini the name is
// cut to kMicNameKept bytes (at a character's start), so the line fits the 80-byte readers of the file (host/nn_hid.c,
// nn_applet.c, hos_main.c), and a name is matched by that much of it
constexpr int kMaxMics = 16;
constexpr size_t kMicNameKept = 60;
std::string mic_want;      // (the chosen microphone's name, cut to kMicNameKept; empty: Off or the default chosen)
bool mic_missing = false;  // (mic_want is not plugged in: the row's last entry, 2 + mic_listed, stands for it)
int mic_listed = 0;        // (the microphones as the host last listed them)
bool can_mic() { return host && host->mic_count && host->mic_name && host->mic_select; }
std::string mic_key(const char* name) {
    std::string k(name ? name : "");
    if (k.size() <= kMicNameKept) return k;
    size_t cut = kMicNameKept;
    while (cut > 0 && ((unsigned char)k[cut] & 0xC0) == 0x80) cut--;   // (not in the middle of a UTF-8 character)
    return k.substr(0, cut);
}
const char* tx_mic(int v) {
    static char buf[160];
    if (v <= 0) return "Off";
    if (v == 1) return "Default";
    if (mic_missing && v == 2 + mic_listed) {
        snprintf(buf, sizeof buf, "%s (not connected)", mic_want.c_str());
        return buf;
    }
    const char* n = can_mic() ? host->mic_name(v - 2) : "";
    snprintf(buf, sizeof buf, "%s", n && *n ? n : "Microphone");
    return buf;
}
/// The Microphone row: Off, the default, a microphone of the list, or the chosen one while it is unplugged (variant 1)
const char* hn_mic(int v, int variant) {
    if (v <= 0) return variant <= 0 ? "The game hears silence." : nullptr;
    if (v == 1)
        return variant <= 0 ? "The system's default recording device. On Windows, desktop apps must be allowed to use the "
                              "microphone (Settings, Privacy & security)."
                            : nullptr;
    const bool missing = variant < 0 ? mic_missing && v == 2 + mic_listed : variant == 1;
    if (variant > 1) return nullptr;
    return missing ? "Not plugged in: Default stands in until it is back." : "This microphone, by its name. A change applies at once.";
}
const char* tx_rumble(int v) { return v ? tx_percent(v) : "Off"; }
// (hookshot) Audio output: the speakers the game mixes for, read once at its start (host/nn_audio.c au_menu_output)
const char* tx_audio_output(int v) {
    static const char* t[] = {"Auto", "Stereo", "5.1"};
    return (unsigned)v < 3 ? t[v] : t[0];
}
// (hookshot) the motion's deadzone: the console's three zero-drift modes, the rate under which the pad counts as still
const char* tx_deadzone(int v) {
    static const char* t[] = {"Low", "Standard", "High"};
    return t[v < 0 ? 0 : v > 2 ? 2 : v];
}
const char* tx_robust(int v) {
    static const char* t[] = {"Auto", "Off", "On"};
    return (unsigned)v < 3 ? t[v] : t[0];
}
const char* tx_preset(int v) {
    static const char* t[] = {"Custom", "Minimum (4GB)", "Low (6GB)", "Balanced (8GB)", "High (12GB)", "Ultra (16GB+)"};
    return t[std::min(std::max(v, 0), 5)];
}
void sync_preset_budgets(int p) {
    switch (p) {
    case 1: opts.vram_mb = 2200;  opts.mirror_mb = 256;  break;
    case 2: opts.vram_mb = 3600;  opts.mirror_mb = 512;  break;
    case 3: opts.vram_mb = 4800;  opts.mirror_mb = 1024; break;
    case 4: opts.vram_mb = 7500;  opts.mirror_mb = 2048; break;
    case 5: opts.vram_mb = 11000; opts.mirror_mb = 3072; break;
    default: opts.vram_mb = 0;    opts.mirror_mb = 0;    break;
    }
}
// (Sharpness: tuned for the window pass's neighbours one picture pixel apart, each about the old setting's sharpening on a
// 3840 x 2160 window at that preset's picture, 2048 x 1152 or the handheld profile's 1536 x 864: same frame, mean |Laplacian|
// of the window and of the picture, on the slider's steps)
void apply_preset(int p) {
    switch (p) {
    case 1: // Minimum (4GB VRAM / iGPU)
        opts.resolution = 4; opts.upscale = 2; opts.sharpness = 30;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 1;
        break;
    case 2: // Low (6GB VRAM)
        opts.resolution = 2; opts.upscale = 2; opts.sharpness = 30;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 1;
        break;
    case 3: // Balanced (8GB VRAM)
        opts.resolution = 1; opts.upscale = 1; opts.sharpness = 30;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    case 4: // High (12GB VRAM)
        opts.resolution = 1; opts.upscale = 2; opts.sharpness = 40;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    case 5: // Ultra (16GB+ VRAM)
        opts.resolution = 1; opts.upscale = 2; opts.sharpness = 50;
        opts.framegen = 3;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    default:
        break;
    }
    // (the Upscaler is the player's: a preset never turns DLSS on or off. With DLSS its resolution is DLSS's nearest mode)
    if (p > 0) opts.resolution = ur::snap(upscaler(), opts.resolution);
    sync_preset_budgets(p);
}

MENU_HINTS(hn_preset, 0, "Each row as you set it. Changing most rows yourself comes back to Custom.",
           "Resolution 50% (with DLSS its nearest mode), Lanczos, Sharpness 50%, 2x frames, Handheld profile.",
           "Resolution 75% (with DLSS its nearest mode), Lanczos, Sharpness 35%, 2x frames, Handheld profile.",
           "Resolution 100% (with DLSS its nearest mode), Catmull-Rom, Sharpness 25%, 2x frames.",
           "Resolution 100% (with DLSS its nearest mode), Lanczos, Sharpness 30%, 2x frames.",
           "Resolution 100% (with DLSS its nearest mode), Lanczos, Sharpness 40%, 3x frames.")
MENU_HINTS(hn_gpu, 0, "The best dedicated GPU.")
/// Frame generation: how many frames are made for each of the game's
const char* hn_framegen(int v, int variant) {
    if (variant > 0 || v == 1) return nullptr;
    if (v <= 0) return "Only the game's own frames: no added delay.";
    static char b[9][96];
    snprintf(b[v], sizeof b[v], "%s between each two of the game's.", v == 2 ? "One made frame" : v == 3 ? "Two made frames" : v == 4 ? "Three made frames"
             : v == 5 ? "Four made frames" : v == 6 ? "Five made frames" : v == 7 ? "Six made frames" : "Seven made frames");
    return b[std::clamp(v, 2, 8)];
}
MENU_HINTS(hn_pace, 0, "As soon as it is ready.", "At an even interval: less stutter, at up to one frame of delay.")
MENU_HINTS(hn_output_res, 0, "The game's own size.",
           "The sharpest on a 2560x1440 screen: nothing to scale to the window.",
           "The sharpest on a 3840x2160 screen. The game's render targets get a bigger GPU pool, so a full-size 3D fits.",
           "The sharpest on a 1920x1080 screen: the HUD and text pixel for pixel, nothing to scale to the window.")
/// The Window scaling filter: its filter, or that none is needed at this window's size
const char* hn_filter(int v, int variant) {
    static const char* const t[] = {"The plain copy.", "Sharper.", "Sharper still; can ring on edges."};
    static const char* const none = "The window is the Output resolution's size: nothing to scale. The choice is kept for "
                                    "another size.";
    if (variant < 0) return filter_needed() ? t[std::clamp(v, 0, 2)] : none;
    return variant == 0 ? t[std::clamp(v, 0, 2)] : variant == 1 ? none : nullptr;
}
/// Sharpness: with DLSS or DLAA, the amount that suits them
const char* hn_sharpness(int v, int variant) {
    static const char* const dlss = "DLSS and DLAA look softer than the game's own upscaler: around 80% suits them.";
    const bool with_dlss = variant < 0 ? upscaler() != ur::GAME : variant == 1;
    if (variant > 1) return nullptr;
    return with_dlss ? dlss : v == 0 ? "Off: the picture as it is." : nullptr;
}
MENU_HINTS(hn_handheld, 0, "Docked: the Output resolution's size.", "Lower resolutions, lighter on the GPU.")
MENU_HINTS(hn_ultrawide, 0, "The game's own shape.", "The camera takes the screen's shape, wider or narrower. The HUD is still stretched.")
MENU_HINTS(hn_robust, 0, "On, on every card.", "Can be faster on some NVIDIA cards.", "Always on.")
MENU_HINTS(hn_fps, 0, nullptr, "With frame generation on, the game's own rate too.")
MENU_HINTS(hn_speed, 0, "A step each frame: the game slows down whenever it runs below 60 fps.",
           "Normal speed down to 15 fps, by taking longer steps.")
MENU_HINTS(hn_text_speed, 0, "The game's own pace, with its pauses.", "Twice as fast.", "Four times as fast.",
           "Each page at once, as pressing A does.")
MENU_HINTS(hn_audio_output, 0, "5.1 when the default output device has 5.1 or more speakers, else stereo.",
           "Stereo, whatever the device has.", "5.1, whatever the device has. Mixed down on stereo speakers.")
MENU_HINTS(hn_background, 0, "Only while the game's window has the focus.",
           "Also behind other windows, for streaming or other apps. The keyboard and mouse still need the game's window.")
MENU_HINTS(hn_mousecam, 0, "The mouse stands in for the right stick, with its curve and top speed.",
           "The mouse turns the camera itself, one to one, as in a PC game.")
MENU_HINTS(hn_mouse_invert, 0, nullptr, "Moving the mouse forward tilts the camera down.")
MENU_HINTS(hn_gyro, 0, "The sticks alone aim.",
           "Switch Pro controllers and pads in Switch mode, DualSense and DualShock 4. The value says whether the "
           "connected pad has a gyroscope.")
MENU_HINTS(hn_pad_layout, 0, "The Switch's order: A on the right, B at the bottom, X at the top, Y on the left.",
           "A at the bottom (action and talk), B on the right, X on the left, Y at the top.",
           "A at the bottom, B on the right; X and Y as on the Switch.",
           "Bindings of your own, from the Controls page.")

Row rows[] = {
    // Graphics (the Upscaler only in a build with DLSS: row_shown)
    {TAB_GRAPHICS, "Quality preset", &opts.preset, 0, 5, 1, tx_preset,
     "Sets the scaling rows, frame generation and memory limits for a GPU with that much memory. The Upscaler stays yours.",
     hn_preset},
    {TAB_GRAPHICS, "Fullscreen", &opts.fullscreen, 0, 1, 1, tx_onoff,
     "Fills the whole screen with the game, without the window's border. The same as F11.", nullptr, false, true},
    {TAB_GRAPHICS, "GPU device", &opts.gpu_device, 0, 8, 1, tx_gpu,
     "Which graphics card the game runs on. Applies after a restart.", hn_gpu, true, true},
    {TAB_GRAPHICS, "Frame generation", &opts.framegen, 0, 8, 1, tx_framegen,
     "Makes frames between the game's own, for smoother motion, at a little input delay. Fast motion or camera cuts can "
     "show artifacts.", hn_framegen},
    {TAB_GRAPHICS, "Frame pacing", &opts.pace, 0, 1, 1, tx_onoff, "When each frame is shown.", hn_pace},
    {TAB_GRAPHICS, "NVIDIA Reflex", &opts.reflex, 0, 2, 1, tx_reflex,
     "NVIDIA GPUs: each frame starts as late as it can, for less input delay. Not used with frame generation.", hn_reflex},
    // (the picture's way, top to bottom: the output's size, what fills it from the 3D, the 3D's size, then the window)
    {TAB_GRAPHICS, "Output resolution", &opts.output_res, 0, kLastOutput, 1, tx_output_res,
     "The game's whole frame, HUD included, before it is scaled to the window. Docked only. Applies after a restart.",
     hn_output_res, true},
    {TAB_GRAPHICS, "Upscaler", &opts.dlss, 0, 2, 1, tx_upscaler, "What scales the 3D up to the Output resolution.",
     hn_upscaler},
    {TAB_GRAPHICS, "DLSS model", &opts.dlss_model, 0, ur::kModels - 1, 1, tx_dlss_model,
     "Which of NVIDIA's DLSS networks draws the picture.", hn_dlss_model, false, true},
    {TAB_GRAPHICS, "Render resolution", &opts.resolution, 0, ur::kValues - 1, 1, tx_resolution,
     "The share of the output the 3D is drawn at; the Upscaler fills in the rest.", hn_resolution},
    {TAB_GRAPHICS, "Window scaling filter", &opts.upscale, 0, 2, 1, tx_filter,
     "How the finished picture is scaled to the window, when its size differs from the Output resolution.", hn_filter},
    {TAB_GRAPHICS, "Sharpness", &opts.sharpness, 0, 100, 10, tx_percent,
     "Sharpens the picture as it goes to the window, with any filter. Too much shows halos around edges.", hn_sharpness},
    {TAB_GRAPHICS, "Handheld profile", &opts.handheld, 0, 1, 1, tx_onoff,
     "Runs the game as if the console were undocked. Applies after a restart.", hn_handheld, true},
    {TAB_GRAPHICS, "Ultrawide", &opts.ultrawide, 0, 1, 1, tx_ultrawide,
     "The camera's shape on a screen wider or narrower than 16:9 (4:3, 16:10). Applies after a restart.", hn_ultrawide, true},
    {TAB_GRAPHICS, "Robust buffers", &opts.robust, 0, 2, 1, tx_robust,
     "Bounds-checked buffer reads: a read past a buffer's end is dropped instead of stopping the GPU. Applies after a "
     "restart.", hn_robust, true},

    // Gameplay
    {TAB_GAMEPLAY, "FPS counter", &opts.show_fps, 0, 1, 1, tx_onoff, "Frames per second in the top right corner.", hn_fps,
     false, true},
    {TAB_GAMEPLAY, "Game speed", &opts.realtime, 0, 1, 1, tx_speed, "How the game keeps time when it runs below 60 fps.",
     hn_speed},
    {TAB_GAMEPLAY, "Text speed", &opts.text_speed, 0, 3, 1, tx_text_speed,
     "How fast the letters of dialogue and other messages appear.", hn_text_speed},
    {TAB_GAMEPLAY, "Audio output", &opts.audio_output, 0, 2, 1, tx_audio_output,
     "The speakers the game mixes its sound for. Applies after a restart.", hn_audio_output, true, true},
    {TAB_GAMEPLAY, "Microphone", &opts.mic, 0, 2 + kMaxMics, 1, tx_mic,
     "What the ocarina hears when the game's microphone option is on (playing an instrument, or humming).", hn_mic, false,
     true},

    // Controls (10 rows; the Motion row opens the 5 TAB_MOTION rows)
    {TAB_CONTROLS, "Controller in background", &opts.pad_background, 0, 1, 1, tx_onoff,
     "Whether the controller plays the game while another window has the focus.", hn_background},
    {TAB_CONTROLS, "Mouse camera", &opts.mouse_native, 0, 1, 1, tx_mousecam, "How the captured mouse turns the camera.",
     hn_mousecam, false, true},
    {TAB_CONTROLS, "Mouse sensitivity", &opts.mouse_sens, 10, 500, 10, tx_sens,
     "How far the camera turns for a mouse movement, with either Mouse camera: at 1.00 native look turns 0.1 degree a "
     "count. The - and = keys change it in game too.", nullptr, false, true},
    {TAB_CONTROLS, "Invert mouse vertical", &opts.mouse_invert, 0, 1, 1, tx_onoff, "Swaps up and down for the mouse.",
     hn_mouse_invert, false, true},
    {TAB_CONTROLS, "Invert camera X", &opts.invert_cam_x, 0, 1, 1, tx_onoff,
     "Inverts horizontal look (left/right) on the right analog stick.", nullptr, false, true},
    {TAB_CONTROLS, "Invert camera Y", &opts.invert_cam_y, 0, 1, 1, tx_onoff,
     "Inverts vertical look (up/down) on the right analog stick.", nullptr, false, true},
    {TAB_CONTROLS, "Rumble strength", &opts.rumble, 0, 150, 25, tx_rumble,
     "How strongly the controller rumbles, in percent of the game's rumble. Off turns it off.", nullptr, false, true},
    {TAB_MOTION, "Motion controls", &opts.gyro, 0, 1, 1, tx_gyro,
     "Aiming and looking with the controller's gyroscope where the game uses motion, as on the Switch.", hn_gyro, false, true},
    {TAB_MOTION, "Motion sensitivity X", &opts.gyro_sens_x, 10, 300, 10, tx_percent,
     "How far the aim moves left and right for a given turn of the controller. 100% is the console's own.", nullptr, false,
     true},
    {TAB_MOTION, "Motion sensitivity Y", &opts.gyro_sens_y, 10, 300, 10, tx_percent,
     "How far the aim moves up and down for a given tilt of the controller. 100% is the console's own.", nullptr, false,
     true},
    {TAB_MOTION, "Invert motion X", &opts.gyro_inv_x, 0, 1, 1, tx_onoff, "Turns the motion aim's left and right around.",
     nullptr, false, true},
    {TAB_MOTION, "Invert motion Y", &opts.gyro_inv_y, 0, 1, 1, tx_onoff, "Turns the motion aim's up and down around.",
     nullptr, false, true},
    {TAB_MOTION, "Motion deadzone", &opts.gyro_deadzone, 0, 2, 1, tx_deadzone,
     "How slow a turn counts as none, the console's three settings: Standard is its own. High steadies a pad that jitters "
     "when held still; Low keeps the smallest movements.", nullptr, false, true},
    {TAB_CONTROLS, "Layout preset", &opts.pad_layout, 0, 3, 1, tx_pad_layout,
     "Which face button is which. Binding one yourself on the Controls page makes it Custom.", hn_pad_layout},
    {TAB_CONTROLS, "Motion", nullptr, 0, 0, 0, nullptr,   // (hookshot: opens the Motion page, the TAB_MOTION rows)
     "Motion controls: aiming with the controller's gyroscope, its sensitivity and its inverts."},
    {TAB_CONTROLS, "Controls", nullptr, 0, 0, 0, nullptr,   // (hookshot: opens its page)
     "Which gamepad button and which keys press each of the Switch's buttons."},
    // (Restart, Quit and Cheats stay the table's last three rows: kRestartRow, kQuitRow and kCheatsRow count back from its end)
    {TAB_GAMEPLAY, "Restart", nullptr, 0, 0, 0, nullptr,
     "Starts the game again. Save your progress first. Press Enter or the bottom face button twice to restart. Esc or the "
     "right face button cancels."},
    {TAB_GAMEPLAY, "Quit", nullptr, 0, 0, 0, nullptr,
     "Save your progress first. Press Enter or the bottom face button twice to quit. Esc or the right face button cancels."},
    {TAB_GAMEPLAY, "Cheats", nullptr, 0, 0, 0, nullptr,
     "Open the private gameplay tools."},
};
constexpr int kCheatsRow = sizeof rows / sizeof rows[0] - 1;
constexpr int kQuitRow = kCheatsRow - 1;
constexpr int kRestartRow = kCheatsRow - 2;
const Row* pending = nullptr;   // (Restart or Quit: shows Confirm until its second press)
bool cheats_enabled = false;
std::atomic<int> protect_hearts{-1};
// -1 disables the shortcut, 0 enables it on the ground, and 1 means flying.
std::atomic<int> player_flight{-1};
std::atomic<int> free_camera{0};
std::atomic<int> cheat_action{0};
int clock_hour = 12;
int pouch_item = 0;
int pouch_delta = 1;
int cheat_selected = 0;
constexpr const char* pouch_items[] = {"Deku Seeds", "Deku Nuts", "Deku Sticks", "Bombs", "Arrows"};
constexpr int kPouchItems = sizeof pouch_items / sizeof pouch_items[0];
// (teleport list: generated by work/oot-teleport-tools/write_destinations.py)
// (the order of oot_destinations in host/hos_main.c; Enter sends 1000 + the index)
constexpr const char* teleport_names[] = {
    "Kokiri Forest",
    "Hyrule Field",
    "Kakariko Village",
    "Market",
    "Hyrule Castle",
    "Lost Woods",
    "Sacred Forest Meadow",
    "Lake Hylia",
    "Graveyard",
    "Death Mountain Trail",
    "Death Mountain Crater",
    "Goron City",
    "Zora's River",
    "Zora's Domain",
    "Zora's Fountain",
    "Gerudo Valley",
    "Gerudo Fortress",
    "Haunted Wasteland",
    "Desert Colossus",
    "Inside the Deku Tree",
    "Dodongo's Cavern",
    "Jabu-Jabu's Belly",
    "Forest Temple",
    "Fire Temple",
    "Water Temple",
    "Shadow Temple",
    "Spirit Temple",
    "Bottom of the Well",
    "Ice Cavern",
    "Gerudo Training Ground",
    "Ganon's Castle"};
constexpr const char* teleport_notes[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "adult only", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
struct TeleportSection { const char* name; int first; };
constexpr TeleportSection teleport_sections[] = {{"Overworld", 0}, {"Dungeons", 19}};
constexpr int kTeleportSections = sizeof teleport_sections / sizeof teleport_sections[0];
// (end of the generated teleport list)
constexpr int kTeleports = sizeof teleport_names / sizeof teleport_names[0];
int teleport_choice = 0;
int teleport_scroll = 0;   // (the first list line shown on the Teleport page)
constexpr int kTeleportLines = 20;   // (list lines the Teleport page shows at once)
// the Inventory page (gpu/vk/oot_cheat_items.h): items and their section headings, counts from the game side
#define OOT_MENU_ITEM_LABEL(label, name, kind) label,
#define OOT_MENU_ITEM_KIND(label, name, kind) kind,
#define OOT_MENU_NO_SECTION(name)
constexpr const char* item_names[] = {OOT_CHEAT_ITEMS(OOT_MENU_NO_SECTION, OOT_MENU_ITEM_LABEL)};
constexpr unsigned char item_kinds[] = {OOT_CHEAT_ITEMS(OOT_MENU_NO_SECTION, OOT_MENU_ITEM_KIND)};
constexpr int kItems = sizeof item_names / sizeof item_names[0];
// (each section's first item: a counter that a section heading reads and an item bumps)
constexpr int item_section_count() {
    int n = 0;
#define OOT_MENU_COUNT_SECTION(title) n++;
#define OOT_MENU_SKIP_ITEM(label, name, kind)
    OOT_CHEAT_ITEMS(OOT_MENU_COUNT_SECTION, OOT_MENU_SKIP_ITEM)
    return n;
}
constexpr int kItemSections = item_section_count();
struct ItemSections { TeleportSection s[kItemSections]; };
constexpr ItemSections make_item_sections() {
    ItemSections r{};
    int section = 0, item = 0;
#define OOT_MENU_ADD_SECTION(title) r.s[section].name = title, r.s[section].first = item, section++;
#define OOT_MENU_ADD_ITEM(label, name, kind) item++;
    OOT_CHEAT_ITEMS(OOT_MENU_ADD_SECTION, OOT_MENU_ADD_ITEM)
    return r;
}
constexpr ItemSections item_sections = make_item_sections();
std::atomic<int> item_counts[kItems];   // (-1 until the game side reports: OOT_CHEAT_ITEM_BASE + i)
int item_choice = 0, item_scroll = 0;
constexpr int kCheatRows = 11;
constexpr int kRows = sizeof rows / sizeof rows[0];
constexpr int kMenuWidth = menu_hit::kWidth;   // (at a scale of 1: 720 lines)

int cur_tab = TAB_GRAPHICS;
int tab_selected[TAB_COUNT + 1] = {0, 0, 0, 0};   // (and the Motion page's)
bool is_motion_row(const Row& r) { return r.tab == TAB_CONTROLS && !r.value && !strcmp(r.name, "Motion"); }

// ---- the rows that take a restart: their values at the start (menu_init, the GPU's at menu_set_gpu_count) ----
namespace mr = menu_restart;
mr::State restart_state;
int restart_slot[kRows];             // (a row's slot in restart_state, -1 none)
const char* restart_names[mr::kMax];
void restart_init() {
    static bool done = false;
    if (done) return;
    done = true;
    for (int i = 0; i < kRows; i++) {
        restart_slot[i] = rows[i].restart && rows[i].value ? restart_state.add(*rows[i].value) : -1;
        if (restart_slot[i] >= 0) restart_names[restart_slot[i]] = rows[i].name;
    }
}
int restart_slot_of(const Row& r) { return restart_slot[&r - rows]; }
void restart_values(int* now) {
    for (int i = 0; i < kRows; i++)
        if (restart_slot[i] >= 0) now[restart_slot[i]] = *rows[i].value;
}
/// The rows whose value is not the one the game runs with (a bit a slot)
unsigned restart_pending() {
    int now[mr::kMax];
    restart_values(now);
    return restart_state.pending(now);
}
/// Whether closing the menu now asks to restart: something pending, not declined with Later, and a window to close
bool restart_ask() {
    int now[mr::kMax];
    restart_values(now);
    return app_window() && restart_state.ask(now);
}
std::string restart_list(const char* last) {
    return mr::names(restart_names, restart_state.n, restart_pending(), ", ", last);
}
int ask_sel = 0;   // (the restart question's choice: 0 Later, 1 Restart now)
/// "Restart needed: " and the pending rows, on one line of width: the last ones as "and N more" when they do not fit
std::string restart_line(HDC dc, int width) {
    const unsigned all = restart_pending();
    std::string line;
    for (int keep = restart_state.n; keep >= 0; keep--) {   // (the first `keep` pending rows by name)
        unsigned shown = 0;
        int named = 0, more = 0;
        for (int i = 0; i < restart_state.n; i++)
            if ((all >> i) & 1) (named < keep ? (shown |= 1u << i, named++) : more++);
        line = "Restart needed: " + mr::names(restart_names, restart_state.n, shown);
        if (more) line += (named ? " and " : "") + std::to_string(more) + (named ? " more" : more > 1 ? " settings" : " setting");
        SIZE sz{};
        GetTextExtentPoint32A(dc, line.c_str(), (int)line.size(), &sz);
        if (sz.cx <= width) break;
    }
    return line;
}

/// A row on its tab: the Cheats one only with cheats on, the Upscaler only in a build with DLSS
bool row_shown(int i) {
    return (i != kCheatsRow || cheats_enabled) && (rows[i].value != &opts.dlss || ur::upscaler_shown(dlss_built())) &&
           (rows[i].value != &opts.dlss_model || ur::model_shown(dlss_built()));
}
int tab_row_count(int t) {
    int c = 0;
    for (int i = 0; i < kRows; i++) if (rows[i].tab == (MenuTab)t && row_shown(i)) c++;
    return c;
}
/// The most rows a tab has: the menu keeps that height whichever tab is shown
int max_tab_rows() {
    int most = 0;
    for (int t = 0; t < TAB_COUNT; t++) most = std::max(most, tab_row_count(t));
    return most;
}

int tab_row_to_global(int t, int local_idx) {
    int count = 0;
    for (int i = 0; i < kRows; i++) {
        if (rows[i].tab == (MenuTab)t && row_shown(i)) {
            if (count == local_idx) return i;
            count++;
        }
    }
    return 0;
}

std::atomic<bool> open_now{false};
bool menu_refresh = false;
// (a value too long for its row, a GPU's name: the selected row's slides through it, a character a step, resting at
// each end; the other rows end it with "...". paint_menu says how far it can slide, menu_tick moves it)
int marquee_pos = 0, marquee_max = 0;
int64_t marquee_next = 0;
constexpr int64_t kMarqueeStep = 150000000LL, kMarqueeRest = 1200000000LL;   // (ns)
// (hookshot) the mouse (menu_tick): the tab strip's part under the cursor (a tab 0 to 2, 3 the previous, 4 the next; -1
// none); paint_menu's record of each shown row's value's left edge (as with its arrows, whether shown or not) and of the
// rows a slider's track was drawn on; the slider held with the left button (its row in the tab, -1 none)
int hover_tab = -1;
int row_value_l[kRows];
bool row_slider[kRows];
int drag_row = -1;
bool mouse_read = false;   // (the menu has read the mouse since it opened)
// (the Cheats page's and the list pages' record of each shown row's or line's value's left edge: the menu's width, none)
int cheat_value_l[kCheatRows];
int list_value_l[kTeleportLines];
bool is_slider(const Row& r) { return r.value == &opts.sharpness || r.value == &opts.mouse_sens || r.value == &opts.gyro_sens_x || r.value == &opts.gyro_sens_y; }
/// (a scale: the arrows, a click and the wheel stop at its ends; Rumble strength would go from Off to its strongest)
bool stops_at_ends(const Row& r) { return r.value == &opts.mouse_sens || r.value == &opts.rumble || r.value == &opts.gyro_sens_x || r.value == &opts.gyro_sens_y; }
/// (the ini's Rumble strength on the menu's steps: 0 to 150, the nearest 25)
int rumble_step(int v) { return (std::min(std::max(v, 0), 150) + 12) / 25 * 25; }
std::string ini_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string dir(exe, n);
    return dir.substr(0, dir.find_last_of("\\/") + 1) + "hookshot_options.ini";
}
const char* ini_key(const Row& r) {
    if (r.value == &opts.output_res) return "output_res";   // (host/nn_applet.c reads it at the start)
    // (rows renamed since their values were first saved keep their keys: an options file from before still loads)
    if (r.value == &opts.resolution) return "resolution";
    if (r.value == &opts.upscale) return "upscaling";
    if (r.value == &opts.dlss) return "dlss";   // (as opts.dlss and RX_VK_DLSS)
    static char k[32];
    size_t n = 0;
    for (const char* p = r.name; *p && n + 1 < sizeof k; p++) k[n++] = *p == ' ' ? '_' : (char)tolower(*p);
    k[n] = 0;
    return k;
}
// (hookshot) the controls, kept by the host (host/nn_hid.c reads them back from here): pad_<button>, key_<button>
const char* const bind_names[HOS_KEY_N] = {"a", "b", "x", "y", "ls", "rs", "l", "r", "zl", "zr", "plus", "minus",
                                           "left", "up", "right", "down", "move_up", "move_down", "move_left",
                                           "move_right", "cam_up", "cam_down", "cam_left", "cam_right"};
bool can_bind() { return host && host->binding && host->bind && host->pad_raw; }
const char* tx_pad_layout(int v) {
    static const char* t[] = {"Nintendo (Def)", "Xbox / PC", "Swap Face", "Custom"};
    return t[std::min(std::max(v, 0), 3)];
}
void apply_pad_layout(int layout) {
    if (!can_bind()) return;
    if (layout == 0) { // Nintendo (Default: Bottom=B, Right=A, Top=X, Left=Y)
        host->bind(HOS_BIND_PAD, HOS_SW_A, 0, HOS_PAD_EAST);
        host->bind(HOS_BIND_PAD, HOS_SW_B, 0, HOS_PAD_SOUTH);
        host->bind(HOS_BIND_PAD, HOS_SW_X, 0, HOS_PAD_NORTH);
        host->bind(HOS_BIND_PAD, HOS_SW_Y, 0, HOS_PAD_WEST);
    } else if (layout == 1) { // Xbox / PC Standard (Bottom=A, Right=B, Left=X, Top=Y)
        host->bind(HOS_BIND_PAD, HOS_SW_A, 0, HOS_PAD_SOUTH);
        host->bind(HOS_BIND_PAD, HOS_SW_B, 0, HOS_PAD_EAST);
        host->bind(HOS_BIND_PAD, HOS_SW_X, 0, HOS_PAD_WEST);
        host->bind(HOS_BIND_PAD, HOS_SW_Y, 0, HOS_PAD_NORTH);
    } else if (layout == 2) { // Swap Face Buttons (Bottom=A, Right=B, Top=X, Left=Y)
        host->bind(HOS_BIND_PAD, HOS_SW_A, 0, HOS_PAD_SOUTH);
        host->bind(HOS_BIND_PAD, HOS_SW_B, 0, HOS_PAD_EAST);
        host->bind(HOS_BIND_PAD, HOS_SW_X, 0, HOS_PAD_NORTH);
        host->bind(HOS_BIND_PAD, HOS_SW_Y, 0, HOS_PAD_WEST);
    }
}
int detect_pad_layout() {
    if (!can_bind()) return opts.pad_layout;
    const int a = host->binding(HOS_BIND_PAD, HOS_SW_A, 0);
    const int b = host->binding(HOS_BIND_PAD, HOS_SW_B, 0);
    const int x = host->binding(HOS_BIND_PAD, HOS_SW_X, 0);
    const int y = host->binding(HOS_BIND_PAD, HOS_SW_Y, 0);
    if (a == HOS_PAD_EAST && b == HOS_PAD_SOUTH && x == HOS_PAD_NORTH && y == HOS_PAD_WEST) return 0;
    if (a == HOS_PAD_SOUTH && b == HOS_PAD_EAST && x == HOS_PAD_WEST && y == HOS_PAD_NORTH) return 1;
    if (a == HOS_PAD_SOUTH && b == HOS_PAD_EAST && x == HOS_PAD_NORTH && y == HOS_PAD_WEST) return 2;
    return 3;
}
// (hookshot) the row's choice to the host: off, the default, or a microphone by its name (and its place in the host's list,
// -1 while it is unplugged: of two with one name, the one chosen). Unplugged, the name is the kept one, cut: the host takes
// a name that starts with it
void mic_tell_host() {
    if (!can_mic()) return;
    if (opts.mic <= 1) host->mic_select(opts.mic <= 0 ? 0 : 1, "", -1);
    else if (mic_missing && opts.mic == 2 + mic_listed) host->mic_select(2, mic_want.c_str(), -1);   // (the kept name, cut)
    else host->mic_select(2, host->mic_name(opts.mic - 2), opts.mic - 2);   // (its whole name: the ini's cut is only the ini's)
}
void mic_row_ends() {
    for (Row& r : rows)
        if (r.value == &opts.mic) r.hi = 1 + mic_listed + (mic_missing ? 1 : 0);
}
// (hookshot) the microphones listed again (one plugged in since shows up), the row on mic_want's: the one at the row's place
// when it still has that name (two microphones of one name: the second stays chosen), else the first of that name, else
// its "(not connected)" entry while it is unplugged
void refresh_mics() {
    if (!can_mic()) {
        if (opts.mic >= 2) opts.mic = 1;
        return;
    }
    mic_listed = std::min(host->mic_count(), kMaxMics);
    int at = -1;
    if (!mic_want.empty()) {
        if (opts.mic >= 2 && opts.mic - 2 < mic_listed && mic_key(host->mic_name(opts.mic - 2)) == mic_want) at = opts.mic - 2;
        for (int i = 0; i < mic_listed && at < 0; i++)
            if (mic_key(host->mic_name(i)) == mic_want) at = i;
        opts.mic = at >= 0 ? at + 2 : 2 + mic_listed;
    } else if (opts.mic >= 2) {
        opts.mic = 1;
    }
    mic_missing = !mic_want.empty() && at < 0;
    mic_row_ends();
    mic_tell_host();
}
// (the row changed by the player: the choice is now this one, by its name; the "(not connected)" entry is kept only while
// it is the one chosen)
void mic_chosen() {
    if (opts.mic >= 2 && !(mic_missing && opts.mic == 2 + mic_listed)) {
        mic_want = mic_key(host->mic_name(opts.mic - 2)), mic_missing = false;
    } else if (opts.mic < 2) {
        mic_want.clear(), mic_missing = false;
    }
    mic_row_ends();
    mic_tell_host();
}
void save() {
    // (hookshot) the lines that are not the menu's are kept: the controls, pad_ and key_ (host/nn_hid.c)
    std::string kept;
    if (FILE* f = fopen(ini_path().c_str(), "r")) {
        char line[256];
        while (fgets(line, sizeof line, f))
            if (!strncmp(line, "pad_", 4) || !strncmp(line, "key_", 4)) kept += line;
        fclose(f);
    }
    if (FILE* f = fopen(ini_path().c_str(), "w")) {
        for (const Row& r : rows)
            if (r.value) fprintf(f, "%s=%d\n", ini_key(r), *r.value);
        if (!mic_want.empty()) fprintf(f, "microphone_device=%s\n", mic_want.c_str());   // (hookshot: also while it is unplugged)
        // (2: game_speed below 2 was only ever the old default, see menu_init; 3: the mouse's buttons are bindings, see
        // host/nn_hid.c binds_load)
        fprintf(f, "options_version=3\n");
        if (!can_bind()) fputs(kept.c_str(), f);
        for (int i = 0; i < HOS_KEY_N && can_bind(); i++) {   // (the keys-only controls, HOS_KEY_*, have no pad_ line)
            if (i < HOS_SW_N) fprintf(f, "pad_%s=%d\n", bind_names[i], host->binding(HOS_BIND_PAD, i, 0));
            fprintf(f, "key_%s=%d,%d\n", bind_names[i], host->binding(HOS_BIND_KEY, i, 0), host->binding(HOS_BIND_KEY, i, 1));
        }
        fclose(f);
    }
}

/// A row's new value v, as the arrows set it (dir: 1 up, -1 down, for the rows that skip or wrap), saved unless !keep
/// (a slider's drag: saved once it is let go)
void set_row(Row& r, int v, int dir, bool keep = true) {
    // (nothing else to choose: no change, no save. A saved choice that cannot run here is kept)
    if (row_fixed(r.value)) return;
    // (the scaling rows: only what can run here; the Render resolution row through the Upscaler's own steps)
    if (r.value == &opts.dlss) v = ur::step_upscaler(upscaler(), dir, true);
    if (r.value == &opts.resolution) v = ur::step(upscaler(), *r.value, dir);
    if (r.value == &opts.output_res) {   // (smallest to largest: kOutputOrder)
        v = step_output_choice(*r.value, dir);
        if (v == *r.value) return;
    }
    if (r.value == &opts.framegen && v == 1) v = dir > 0 ? 2 : 0;   // (off, 2x ... 8x)
    if (stops_at_ends(r)) v = std::min(std::max(v, r.lo), r.hi);   // (a scale: it stops at its ends)
    if (r.value == &opts.pad_layout) {
        if (v > 2) v = dir > 0 ? 0 : 2;
        if (v < 0) v = 2;
        *r.value = v;
        apply_pad_layout(*r.value);
    } else if (r.value == &opts.preset) {
        *r.value = v > r.hi ? r.lo : v < r.lo ? r.hi : v;
        apply_preset(*r.value);
    } else {
        *r.value = v > r.hi ? r.lo : v < r.lo ? r.hi : v;
        // (another Upscaler: the Render resolution moves to the nearest of its steps, DLSS's modes or the game's)
        if (r.value == &opts.dlss) opts.resolution = ur::snap(upscaler(), opts.resolution);
        if (!r.preset_exempt) {   // (the controls, the window and the counter are not the preset's)
            opts.preset = 0; // Custom
            sync_preset_budgets(0);
        }
    }
    if (r.value == &opts.mic && can_mic()) mic_chosen();   // (hookshot)
    {   // (a row that takes a restart, or a preset that sets one: asked again at the next close)
        int now[mr::kMax];
        restart_values(now);
        restart_state.note(now);
    }
    if (keep) save();
}

// ---- (hookshot) the Controls page: what presses each of the Switch's buttons, a gamepad input and up to two keys ----
struct Control {
    int button;   // HOS_SW_*, or HOS_KEY_* (keys only)
    const char* name;
};
const Control controls[] = {{HOS_SW_A, "A"}, {HOS_SW_B, "B"}, {HOS_SW_X, "X"}, {HOS_SW_Y, "Y"}, {HOS_SW_L, "L"},
                            {HOS_SW_R, "R"}, {HOS_SW_ZL, "ZL"}, {HOS_SW_ZR, "ZR"}, {HOS_SW_PLUS, "Plus"},
                            {HOS_SW_MINUS, "Minus"}, {HOS_SW_LS, "Left stick press"}, {HOS_SW_RS, "Right stick press"},
                            {HOS_SW_UP, "D-pad up"}, {HOS_SW_DOWN, "D-pad down"}, {HOS_SW_LEFT, "D-pad left"},
                            {HOS_SW_RIGHT, "D-pad right"}, {HOS_KEY_MOVE_UP, "Move forward"},
                            {HOS_KEY_MOVE_DOWN, "Move back"}, {HOS_KEY_MOVE_LEFT, "Move left"},
                            {HOS_KEY_MOVE_RIGHT, "Move right"}, {HOS_KEY_CAM_UP, "Camera up"},
                            {HOS_KEY_CAM_DOWN, "Camera down"}, {HOS_KEY_CAM_LEFT, "Camera left"},
                            {HOS_KEY_CAM_RIGHT, "Camera right"}};
constexpr int kControls = sizeof controls / sizeof controls[0];   // (then "Reset to defaults")
bool keys_only(int i) { return i < kControls && controls[i].button >= HOS_SW_N; }   // (no gamepad input for those)
const char* pad_name(int p) {
    static const char* t[HOS_PAD_N] = {"Bottom button", "Right button", "Left button", "Top button", "Select / Back",
                                       "Home", "Start", "Left stick press", "Right stick press", "L1 / LB", "R1 / RB",
                                       "D-pad up", "D-pad down", "D-pad left", "D-pad right", "L2 / LT", "R2 / RT"};
    return p >= 0 && p < HOS_PAD_N ? t[p] : "None";
}
const char* key_name(int vk) {
    static char b[16];
    if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9')) return b[0] = (char)vk, b[1] = 0, b;
    if (vk >= VK_F1 && vk <= VK_F12) return snprintf(b, sizeof b, "F%d", vk - VK_F1 + 1), b;
    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD0 + 9) return snprintf(b, sizeof b, "Num %d", vk - VK_NUMPAD0), b;
    switch (vk) {
    case VK_SPACE: return "Space";
    case VK_SHIFT: return "Shift";
    case VK_CONTROL: return "Ctrl";
    case VK_MENU: return "Alt";
    case VK_TAB: return "Tab";
    case VK_RETURN: return "Enter";
    case VK_BACK: return "Backspace";
    case VK_CAPITAL: return "Caps Lock";
    case VK_UP: return "Up";
    case VK_DOWN: return "Down";
    case VK_LEFT: return "Left";
    case VK_RIGHT: return "Right";
    case VK_HOME: return "Home";
    case VK_END: return "End";
    case VK_PRIOR: return "Page Up";
    case VK_NEXT: return "Page Down";
    case VK_ADD: return "Num +";
    case VK_SUBTRACT: return "Num -";
    case VK_MULTIPLY: return "Num *";
    case VK_DIVIDE: return "Num /";
    case VK_DECIMAL: return "Num .";
    case VK_OEM_1: return ";";
    case VK_OEM_PLUS: return "=";
    case VK_OEM_COMMA: return ",";
    case VK_OEM_MINUS: return "-";
    case VK_OEM_PERIOD: return ".";
    case VK_OEM_2: return "/";
    case VK_OEM_3: return "`";
    case VK_OEM_4: return "[";
    case VK_OEM_5: return "\\";
    case VK_OEM_6: return "]";
    case VK_OEM_7: return "'";
    case VK_LBUTTON: return "Left click";
    case VK_RBUTTON: return "Right click";
    case VK_MBUTTON: return "Middle click";
    case 0x05: return "Mouse 4";   // (VK_XBUTTON1)
    case 0x06: return "Mouse 5";
    }
    return snprintf(b, sizeof b, "Key %d", vk), b;
}
// the keys a binding can take (Esc cancels, Insert and F1 are the menu's, Delete clears)
const std::vector<int>& bindable_keys() {
    static std::vector<int> k;
    if (k.empty()) {
        for (int c = 'A'; c <= 'Z'; c++) k.push_back(c);
        for (int c = '0'; c <= '9'; c++) k.push_back(c);
        for (int c = VK_F2; c <= VK_F12; c++) k.push_back(c);
        for (int c = VK_NUMPAD0; c <= VK_NUMPAD0 + 9; c++) k.push_back(c);
        for (int c : {VK_SPACE, VK_SHIFT, VK_CONTROL, VK_MENU, VK_TAB, VK_RETURN, VK_BACK, VK_CAPITAL, VK_UP, VK_DOWN,
                      VK_LEFT, VK_RIGHT, VK_HOME, VK_END, VK_PRIOR, VK_NEXT, VK_ADD, VK_SUBTRACT, VK_MULTIPLY,
                      VK_DIVIDE, VK_DECIMAL, VK_OEM_1, VK_OEM_PLUS, VK_OEM_COMMA, VK_OEM_MINUS, VK_OEM_PERIOD,
                      VK_OEM_2, VK_OEM_3, VK_OEM_4, VK_OEM_5, VK_OEM_6, VK_OEM_7, VK_LBUTTON, VK_RBUTTON, VK_MBUTTON,
                      0x05, 0x06})   // (the mouse's buttons too: they count while the mouse is captured)
            k.push_back(c);
    }
    return k;
}
int page = 0;          // 0 the options, 1 the controls, 2 the cheats, 3 the teleport list, 4 the inventory, 5 the restart question
int csel = 0, ccol = 0;   // the controls' row (kControls: reset) and column (0 the gamepad, 1 the keyboard)
int listening = 0;     // waiting for the new input: 1 a gamepad's, 2 a key
bool listen_armed = false;   // (once what started it was let go)
int64_t listen_end = 0;
std::string keys_text(int button) {
    const int a = host->binding(HOS_BIND_KEY, button, 0), b = host->binding(HOS_BIND_KEY, button, 1);
    std::string t = a ? key_name(a) : "";
    if (b) t += std::string(a ? ", " : "") + key_name(b);
    return t.empty() ? "None" : t;
}

// ---- what is drawn: bitmaps of the menu and the counter, in the window's byte order ----
struct Bitmap {
    uint32_t w = 0, h = 0;
    std::vector<uint8_t> px;
    uint64_t version = 0;
    bool keyed = false;                   // (black is see-through: copied as the runs of pixels in between)
    std::vector<VkBufferImageCopy> runs;
};
Bitmap menu_bmp, fps_bmp;
std::mutex bmp_mu;
int scale_of(uint32_t h) { return std::max(1u, h / 720); }

thread_local uint32_t* paint_bits = nullptr;   // (the bitmap being drawn, for a paint that writes pixels itself)
void gdi_draw(Bitmap& out, uint32_t w, uint32_t h, void (*paint)(HDC, uint32_t, uint32_t, int), int scale) {
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth = (LONG)w;
    bi.bmiHeader.biHeight = -(LONG)h;
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    HDC dc = CreateCompatibleDC(nullptr);
    void* bits = nullptr;
    HBITMAP bmp = CreateDIBSection(dc, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!bmp) { DeleteDC(dc); return; }
    HGDIOBJ old = SelectObject(dc, bmp);
    paint_bits = (uint32_t*)bits;   // (the start-up screen blends its logo in: vk_startup_paint.h)
    paint(dc, w, h, scale);
    paint_bits = nullptr;
    GdiFlush();
    if (const char* dir = &out == &menu_bmp ? getenv("RX_VK_MENU_DUMP") : nullptr) {   // (test runs: each drawing of the menu, as menu_<n>.bmp)
        static int n = 0;
        char path[1024];
        snprintf(path, sizeof path, "%s/menu_%03d.bmp", dir, n++);
        if (FILE* f = fopen(path, "wb")) {   // (the 14-byte file header and the 40-byte info header, little-endian: 32-bit, top-down)
            uint8_t hd[54] = {'B', 'M'};
            auto put = [&](int at, uint32_t v, int bytes) { for (int i = 0; i < bytes; i++) hd[at + i] = (uint8_t)(v >> (8 * i)); };
            put(2, 54 + w * h * 4, 4), put(10, 54, 4), put(14, 40, 4), put(18, w, 4), put(22, (uint32_t)-(int32_t)h, 4), put(26, 1, 2),
                put(28, 32, 2);
            fwrite(hd, 1, sizeof hd, f), fwrite(bits, 4, (size_t)w * h, f);
            fclose(f);
        }
    }
    const bool rgba = dev.swap_format == VK_FORMAT_R8G8B8A8_UNORM || dev.swap_format == VK_FORMAT_R8G8B8A8_SRGB;
    std::lock_guard lk(bmp_mu);
    out.w = w, out.h = h;
    out.px.assign((const uint8_t*)bits, (const uint8_t*)bits + (size_t)w * h * 4);
    if (out.keyed) {
        out.runs.clear();
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w;) {
                const uint8_t* p = out.px.data() + ((size_t)y * w + x) * 4;
                if (!(p[0] | p[1] | p[2])) { x++; continue; }
                uint32_t e = x + 1;
                for (; e < w; e++) {
                    const uint8_t* q = out.px.data() + ((size_t)y * w + e) * 4;
                    if (!(q[0] | q[1] | q[2])) break;
                }
                VkBufferImageCopy c{};
                c.bufferOffset = ((uint64_t)y * w + x) * 4;
                c.bufferRowLength = w;
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                c.imageOffset = {(int32_t)x, (int32_t)y, 0};
                c.imageExtent = {e - x, 1, 1};
                out.runs.push_back(c);
                x = e;
            }
        }
    }
    for (size_t i = 0; i < out.px.size(); i += 4) {
        if (rgba) std::swap(out.px[i], out.px[i + 2]);
        out.px[i + 3] = 255;
    }
    out.version++;
    SelectObject(dc, old);
    DeleteObject(bmp);
    DeleteDC(dc);
}
HFONT font(int px, int weight) {
    return CreateFontA(-px, 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                       DEFAULT_PITCH, "Segoe UI");
}
int help_px = 0;   // (the help text's height: measure_help)
constexpr int kHintGap = 4, kHintIndent = 10;   // (the value's line: below the row's help, in from its bar)
/// The help text word-wrapped to width, a TextOutA a line (the POSIX GDI stand-in has no DrawTextA), drawn at (x, y)
/// when draw: its height in pixels either way
int wrap_help(HDC dc, const char* text, int width, int x, int y, bool draw) {
    SIZE sz{};
    GetTextExtentPoint32A(dc, "Ag", 2, &sz);
    const int line_h = (int)sz.cy;
    int lines = 0;
    for (const char* p = text; *p;) {
        int n = 0;   // (the most whole words that fit; one word at least, however long)
        for (int end = 0;; end++) {
            while (p[end] && p[end] != ' ') end++;
            GetTextExtentPoint32A(dc, p, end, &sz);
            if (n && sz.cx > width) break;
            n = end;
            if (!p[end]) break;
        }
        if (draw) TextOutA(dc, x, y + lines * line_h, p, n);
        lines++;
        p += n;
        while (*p == ' ') p++;
    }
    return lines * line_h;
}
void fill(HDC dc, int x, int y, int w, int h, COLORREF c) {
    RECT r{x, y, x + w, y + h};
    HBRUSH b = CreateSolidBrush(c);
    FillRect(dc, &r, b);
    DeleteObject(b);
}
void paint_menu(HDC dc, uint32_t w, uint32_t h, int s) {
    fill(dc, 0, 0, (int)w, (int)h, RGB(16, 20, 18));
    fill(dc, 0, 0, (int)w, 2 * s, RGB(96, 176, 112));
    SetBkMode(dc, TRANSPARENT);
    HFONT title = font(20 * s, FW_SEMIBOLD), body = font(16 * s, FW_NORMAL);
    HGDIOBJ old = SelectObject(dc, title);
    SetTextColor(dc, RGB(236, 240, 236));
    const char* heading = page == 5 ? "Restart needed" : page == 4 ? "Inventory" : page == 3 ? "Teleport"
                        : page == 2 ? "Cheats" : "Hookshot";
    TextOutA(dc, 16 * s, 12 * s, heading, (int)strlen(heading));
    {   // (which build, beside the title, so a report can say it without opening the log; the release alone when the whole
        // line would run into the title)
        SIZE tsz{};
        GetTextExtentPoint32A(dc, heading, (int)strlen(heading), &tsz);
        HFONT small = font(12 * s, FW_NORMAL);
        SelectObject(dc, small);
        SetTextColor(dc, RGB(140, 150, 144));
        std::string ver = HOOKSHOT_VERSION;
        const int room = (int)w - 32 * s - tsz.cx - 12 * s;
        SIZE sz{};
        GetTextExtentPoint32A(dc, ver.c_str(), (int)ver.size(), &sz);
        if (sz.cx > room && ver.find(" (") != std::string::npos) {
            ver.resize(ver.find(" ("));
            GetTextExtentPoint32A(dc, ver.c_str(), (int)ver.size(), &sz);
        }
        if (sz.cx <= room) TextOutA(dc, (int)w - 16 * s - sz.cx, 18 * s, ver.c_str(), (int)ver.size());
        SelectObject(dc, title);
        DeleteObject(small);
    }
    SelectObject(dc, body);
    if (page == 5) {   // (hookshot) the restart question, as the menu closes with a setting that takes a restart
        SetTextColor(dc, RGB(236, 240, 236));
        const std::string ask = "Restart to apply " + restart_list(" and ") + "?";
        const int asked = wrap_help(dc, ask.c_str(), (int)w - 32 * s, 16 * s, 46 * s, true);   // (one line or two)
        HFONT small = font(14 * s, FW_NORMAL);
        SelectObject(dc, small);
        SetTextColor(dc, RGB(176, 186, 180));
        const char* lost = "Progress since your last save is lost.";
        TextOutA(dc, 16 * s, std::min(46 * s + asked + 4 * s, 94 * s), lost, (int)strlen(lost));
        SelectObject(dc, body);
        const char* const labels[2] = {"Later", "Restart now"};
        for (int i = 0; i < 2; i++) {
            const int x = menu_hit::ask_button_x(i, s), y = menu_hit::kAskButtonY * s;
            const int bw = menu_hit::kAskButtonW * s, bh = menu_hit::kAskButtonH * s;
            const bool sel = i == ask_sel;
            if (sel) startup_gdi::round_rect(dc, x - s, y - s, bw + 2 * s, bh + 2 * s, 7 * s, RGB(104, 186, 124));   // (a green edge)
            startup_gdi::round_rect(dc, x, y, bw, bh, 6 * s, sel ? RGB(44, 72, 54) : RGB(30, 40, 34));
            SIZE sz{};
            GetTextExtentPoint32A(dc, labels[i], (int)strlen(labels[i]), &sz);
            SetTextColor(dc, sel ? RGB(255, 255, 255) : RGB(196, 204, 198));
            TextOutA(dc, x + (bw - (int)sz.cx) / 2, y + (bh - (int)sz.cy) / 2, labels[i], (int)strlen(labels[i]));
        }
        SelectObject(dc, small);
        SetTextColor(dc, RGB(140, 150, 144));
        const char* hint = "Left/Right choose   Enter confirms   Esc: Later";
        TextOutA(dc, 16 * s, 168 * s, hint, (int)strlen(hint));
        SelectObject(dc, old);
        DeleteObject(small);
        DeleteObject(title);
        DeleteObject(body);
        return;
    }
    if (page == 1) {   // (hookshot) the Controls page
        TextOutA(dc, 16 * s + 110 * s, 12 * s + 4 * s, "Controls", 8);
        const int rh = menu_hit::kCtlRowH * s, top = menu_hit::kCtlTop * s, xpad = menu_hit::kCtlPad * s,
                  xkey = menu_hit::kCtlKey * s;
        SetTextColor(dc, RGB(140, 150, 144));
        TextOutA(dc, 16 * s, 44 * s, "Button", 6);
        TextOutA(dc, xpad, 44 * s, "Gamepad", 7);
        TextOutA(dc, xkey, 44 * s, "Keyboard", 8);
        for (int i = 0; i <= kControls; i++) {
            const int y = top + i * rh;
            const bool sel = i == csel;
            if (sel) startup_gdi::round_rect(dc, 8 * s, y, (int)w - 16 * s, rh, 6 * s, RGB(44, 72, 54));
            SetTextColor(dc, sel ? RGB(255, 255, 255) : RGB(196, 204, 198));
            if (i == kControls) {
                const char* t = "Reset to defaults";
                TextOutA(dc, 16 * s, y + 3 * s, t, (int)strlen(t));
                continue;
            }
            const Control& c = controls[i];
            TextOutA(dc, 16 * s, y + 3 * s, c.name, (int)strlen(c.name));
            for (int col = 0; col < 2; col++) {
                std::string v = col == 0 ? (keys_only(i) ? "-" : pad_name(host->binding(HOS_BIND_PAD, c.button, 0)))
                                         : keys_text(c.button);
                if (sel && col == ccol && listening) v = col == 0 ? "Press a button..." : "Press a key...";
                if (sel && col == ccol) {
                    SIZE sz{};
                    GetTextExtentPoint32A(dc, v.c_str(), (int)v.size(), &sz);
                    fill(dc, (col ? xkey : xpad) - menu_hit::kCtlCellPad * s, y + 2 * s, (int)sz.cx + 12 * s, rh - 4 * s,
                         listening ? RGB(150, 120, 40) : RGB(70, 120, 84));
                }
                TextOutA(dc, col ? xkey : xpad, y + 3 * s, v.c_str(), (int)v.size());
            }
        }
        SetTextColor(dc, RGB(140, 150, 144));
        const char* hint = listening ? "Press the new one   Esc cancels"
                                     : "Enter binds   Delete clears   Esc goes back";
        const char* hint2 = "Gamepad: D-pad moves, bottom button binds, right one goes back";
        TextOutA(dc, 16 * s, top + (kControls + 1) * rh + 6 * s, hint, (int)strlen(hint));
        TextOutA(dc, 16 * s, top + (kControls + 1) * rh + 26 * s, hint2, (int)strlen(hint2));
        SelectObject(dc, old);
        DeleteObject(title);
        DeleteObject(body);
        return;
    }
    if (page == 3 || page == 4) {   // the teleport list and the inventory: a heading line per section, then its rows
        const bool inv = page == 4;
        const int n = inv ? kItems : kTeleports, sections = inv ? kItemSections : kTeleportSections;
        const TeleportSection* sec = inv ? item_sections.s : teleport_sections;
        const int choice = inv ? item_choice : teleport_choice, scroll = inv ? item_scroll : teleport_scroll;
        const int rh = menu_hit::kListRowH * s, top = menu_hit::kListTop * s;
        for (int line = scroll; line < std::min(n + sections, scroll + kTeleportLines); line++) {
            const int y = top + (line - scroll) * rh;
            list_value_l[line - scroll] = (int)w;
            int section = 0;
            while (section + 1 < sections && line >= sec[section + 1].first + section + 1) section++;
            if (line == sec[section].first + section) {
                SetTextColor(dc, RGB(96, 176, 112));
                TextOutA(dc, 16 * s, y + 4 * s, sec[section].name, (int)strlen(sec[section].name));
                continue;
            }
            const int i = line - section - 1;
            if (i == choice) startup_gdi::round_rect(dc, 8 * s, y, (int)w - 16 * s, rh, 6 * s, RGB(44, 72, 54));
            SetTextColor(dc, i == choice ? RGB(255, 255, 255) : RGB(196, 204, 198));
            const char* label = inv ? item_names[i] : teleport_names[i];
            TextOutA(dc, 32 * s, y + 3 * s, label, (int)strlen(label));
            char value[24] = "";
            const char* note = nullptr;
            if (inv) {
                const int have = item_counts[i].load(std::memory_order_relaxed);
                if (have < 0) note = "?";
                else if (item_kinds[i]) snprintf(value, sizeof value, "%d", have), note = value;
                else note = have > 0 ? "Have" : "-";
            } else note = teleport_notes[i];
            if (note) {
                SIZE size{};
                GetTextExtentPoint32A(dc, note, (int)strlen(note), &size);
                if (!inv || strcmp(note, "Have")) SetTextColor(dc, RGB(140, 150, 144));
                list_value_l[line - scroll] = (int)w - menu_hit::kListValueRight * s - (int)size.cx;
                TextOutA(dc, list_value_l[line - scroll], y + 3 * s, note, (int)strlen(note));
            }
        }
        SetTextColor(dc, RGB(140, 150, 144));
        char count[32];
        snprintf(count, sizeof count, "%d of %d", choice + 1, n);
        SIZE size{};
        GetTextExtentPoint32A(dc, count, (int)strlen(count), &size);
        TextOutA(dc, (int)w - 16 * s - size.cx, 22 * s, count, (int)strlen(count));
        const int bottom = top + kTeleportLines * rh + 8 * s;
        const char* hint = inv ? "Up/Down choose   Enter unlocks or relocks   Left/Right amount"
                               : "Up/Down choose   Left/Right next section   Enter travels   Esc back";
        TextOutA(dc, 16 * s, bottom, hint, (int)strlen(hint));
        const char* hint2 = inv ? "Q/E or bumpers: next section   PgUp/PgDn/Home/End jump   Esc back"
                                : "PgUp/PgDn/Home/End jump. Link arrives at the map's own entrance.";
        TextOutA(dc, 16 * s, bottom + 20 * s, hint2, (int)strlen(hint2));
        SelectObject(dc, old);
        DeleteObject(title);
        DeleteObject(body);
        return;
    }
    if (page == 2) {
        const char* labels[] = {"Heart protection", "No-clip flight", "Mark position",
                                "Return to mark", "Set time", "Inventory item", "Change quantity", "Free camera",
                                "Teleport", "Switch age", "Inventory"};
        char clock_text[12], quantity[12];
        snprintf(clock_text, sizeof clock_text, "%02d:00", clock_hour);
        snprintf(quantity, sizeof quantity, "%+d", pouch_delta);
        const char* values[] = {tx_onoff(protect_hearts.load() == 1), tx_onoff(player_flight.load() >= 0),
                                "", "", clock_text, pouch_items[pouch_item], quantity, tx_onoff(free_camera.load()),
                                teleport_names[teleport_choice], "", ""};
        const char* help[] = {"Protects Link's hearts. Hits can still knock him back.",
            "Enable double-tap Jump for takeoff and landing. Hold Jump to climb or L to descend.",
            "Remember Link's position for this loaded scene.",
            "Teleport Link to the marked position in the current scene.",
            "Choose an hour with Left or Right. Enter sets the world clock.",
            "Choose a consumable with Left or Right.",
            "Choose how many to add or remove. Enter applies the change.",
            "Move and look with your usual controls. Page Up / Page Down or R / L changes height.",
            "Enter opens the map list (Left or Right also changes the map here). Link travels through the game's own map change, in his current age.",
            "Swaps Link between child and adult and reloads this map in the other age (the Market when this map has no match).",
            "Enter opens the inventory editor: unlock or relock equipment, items, upgrades and quest items, and change amounts."};
        for (int i = 0; i < kCheatRows; i++) {
            const int y = (menu_hit::kCheatTop + i * menu_hit::kCheatRowH) * s;
            if (i == cheat_selected) startup_gdi::round_rect(dc, 8 * s, y, (int)w - 16 * s, menu_hit::kCheatRowH * s, 6 * s, RGB(44, 72, 54));
            SetTextColor(dc, i == cheat_selected ? RGB(255, 255, 255) : RGB(196, 204, 198));
            TextOutA(dc, 16 * s, y + 4 * s, labels[i], (int)strlen(labels[i]));
            const char* value = values[i];
            SIZE size{};
            GetTextExtentPoint32A(dc, value, (int)strlen(value), &size);
            cheat_value_l[i] = *value ? (int)w - menu_hit::kListValueRight * s - (int)size.cx : (int)w;
            TextOutA(dc, (int)w - menu_hit::kListValueRight * s - size.cx, y + 4 * s, value, (int)strlen(value));
        }
        SetTextColor(dc, RGB(176, 186, 180));
        wrap_help(dc, help[cheat_selected], (int)w - 32 * s, 16 * s, (54 + kCheatRows * 28) * s, true);
        const char* hint = "Enter applies   Esc goes back";
        TextOutA(dc, 16 * s, (105 + kCheatRows * 28) * s, hint, (int)strlen(hint));
        SelectObject(dc, old);
        DeleteObject(title);
        DeleteObject(body);
        return;
    }
    // the tab strip: Graphics, Gameplay, Controls (Q and E, the bumpers or Tab change it: menu_tick)
    const int tab_y = menu_hit::kTabY * s;
    const int tab_h = menu_hit::kTabH * s;
    fill(dc, 0, tab_y, (int)w, tab_h, RGB(22, 28, 25));
    HFONT small_font = font(12 * s, FW_SEMIBOLD);
    SelectObject(dc, small_font);
    SetTextColor(dc, hover_tab == 3 ? RGB(210, 225, 215) : RGB(110, 135, 120));   // (lighter under the cursor: a click goes there)
    TextOutA(dc, 16 * s, tab_y + 8 * s, "[Q / LB]", 8);
    const char* rb_hint = "[E / RB]";
    SIZE rbsz{};
    GetTextExtentPoint32A(dc, rb_hint, (int)strlen(rb_hint), &rbsz);
    SetTextColor(dc, hover_tab == 4 ? RGB(210, 225, 215) : RGB(110, 135, 120));
    TextOutA(dc, (int)w - 16 * s - rbsz.cx, tab_y + 8 * s, rb_hint, (int)strlen(rb_hint));
    HFONT tab_font = font(13 * s, FW_BOLD);
    SelectObject(dc, tab_font);
    const char* const tab_titles[TAB_COUNT] = {"GRAPHICS", "GAMEPLAY", "CONTROLS"};
    const int center_start = menu_hit::kTabSide * s;
    const int tab_slot_w = menu_hit::tab_slot_w(s, (int)w);
    for (int t = 0; t < TAB_COUNT; t++) {
        const bool active = t == (cur_tab == TAB_MOTION ? TAB_CONTROLS : cur_tab);   // (the Motion page: under Controls)
        int slot_x = center_start + t * tab_slot_w;
        SIZE tsz{};
        GetTextExtentPoint32A(dc, tab_titles[t], (int)strlen(tab_titles[t]), &tsz);
        int text_x = slot_x + (tab_slot_w - tsz.cx) / 2;
        int text_y = tab_y + 8 * s;
        if (active) {   // (a pill behind it and a golden underline)
            fill(dc, text_x - 10 * s, tab_y + 3 * s, tsz.cx + 20 * s, tab_h - 6 * s, RGB(38, 64, 48));
            fill(dc, text_x - 10 * s, tab_y + tab_h - 2 * s, tsz.cx + 20 * s, 2 * s, RGB(235, 205, 95));
            SetTextColor(dc, RGB(255, 255, 255));
        } else if (t == hover_tab) {   // (under the cursor: a dimmer pill)
            fill(dc, text_x - 10 * s, tab_y + 3 * s, tsz.cx + 20 * s, tab_h - 6 * s, RGB(30, 46, 37));
            SetTextColor(dc, RGB(215, 225, 218));
        } else {
            SetTextColor(dc, RGB(135, 150, 142));
        }
        TextOutA(dc, text_x, text_y, tab_titles[t], (int)strlen(tab_titles[t]));
    }
    SelectObject(dc, body);
    DeleteObject(tab_font);
    DeleteObject(small_font);
    fill(dc, 0, tab_y + tab_h, (int)w, std::max(1, s), RGB(52, 68, 58));

    const int rh = menu_hit::kRowH * s, top = menu_hit::rows_top(s);
    const int tab_count = tab_row_count(cur_tab);
    marquee_max = 0;   // (set again below while the selected row's value is too long)
    for (int i = 0; i < tab_count; i++) {
        const Row& r = rows[tab_row_to_global(cur_tab, i)];
        const int y = top + i * rh;
        const bool sel = i == tab_selected[cur_tab];
        if (sel) {
            startup_gdi::round_rect(dc, 8 * s, y, (int)w - 16 * s, rh, 6 * s, RGB(44, 72, 54));
            startup_gdi::round_fill(dc, 10 * s, y + 6 * s, 3 * s, rh - 12 * s, RGB(104, 186, 124));   // (a green marker at its left)
        }
        SetTextColor(dc, sel ? RGB(255, 255, 255) : RGB(196, 204, 198));
        TextOutA(dc, 18 * s, y + 4 * s, r.name, (int)strlen(r.name));
        // (the value right-aligned, kept clear of the row's name: see marquee_pos)
        SIZE nsz{};
        GetTextExtentPoint32A(dc, r.name, (int)strlen(r.name), &nsz);
        const int room = (int)w - 18 * s - (18 * s + (int)nsz.cx + 12 * s);
        const char* action = pending == &r ? "Confirm" : ">";   // (">": a page)
        std::string label = r.value ? r.text(*r.value) : action;
        if (r.value && r.restart && *r.value != restart_state.start[restart_slot_of(r)]) label += " (after restart)";
        const bool arrows = r.value && sel && !row_fixed(r.value);
        auto width = [&](const std::string& t) {
            SIZE z{};
            GetTextExtentPoint32A(dc, t.c_str(), (int)t.size(), &z);
            return (int)z.cx;
        };
        const bool output_row = r.value == &opts.output_res;
        const bool left_arrow = arrows && (!output_row || step_output_choice(*r.value, -1) != *r.value);
        const bool right_arrow = arrows && (!output_row || step_output_choice(*r.value, 1) != *r.value);
        auto framed = [&](const std::string& t) {
            return arrows ? (left_arrow ? "<  " : "   ") + t + (right_arrow ? "  >" : "   ") : t;
        };
        std::string v = framed(label);
        if (width(v) > room && arrows) {   // (slides: the longest piece from marquee_pos that fits)
            size_t start = label.size();   // (the last start that still shows the name's end)
            while (start > 0 && width(framed(label.substr(start - 1))) <= room) start--;
            marquee_max = (int)start;
            const size_t from = (size_t)std::min(marquee_pos, marquee_max);
            size_t n = label.size() - from;
            while (n > 1 && width(framed(label.substr(from, n))) > room) n--;
            v = framed(label.substr(from, n));
        } else if (width(v) > room) {   // (not selected: cut with "...")
            std::string t = label;
            while (t.size() > 1 && width(v) > room) {
                t.pop_back();
                while (!t.empty() && t.back() == ' ') t.pop_back();
                v = t + "...";
            }
        } else if (arrows) {
            marquee_max = 0;
        }
        TextOutA(dc, (int)w - 18 * s - width(v), y + 4 * s, v.c_str(), (int)v.size());
        // (the mouse's: where the value is, its arrows counted whether they show or not, so a click's half does not move
        // as the row is selected)
        row_value_l[i] = (int)w - 18 * s - std::min(r.value ? width("<  " + label + "  >") : width(v), room);
        // (a slider's track between the name and the value, with a knob at the value: click or drag it)
        const int tl = menu_hit::kSliderL * s, tr = menu_hit::kSliderR * s;
        row_slider[i] = is_slider(r) && 18 * s + (int)nsz.cx + 8 * s <= tl && row_value_l[i] >= tr + 8 * s;
        if (row_slider[i]) {
            const int ty = y + rh / 2 - s, knob = tl + (tr - tl) * menu_hit::slider_permille(*r.value, r.lo, r.hi) / 1000;
            fill(dc, tl, ty, tr - tl, 2 * s, RGB(60, 80, 68));
            fill(dc, tl, ty, knob - tl, 2 * s, RGB(96, 210, 120));
            fill(dc, knob - 3 * s, y + rh / 2 - 6 * s, 6 * s, 12 * s, sel ? RGB(255, 255, 255) : RGB(176, 186, 180));
        }
    }
    const int help_top = top + max_tab_rows() * rh + 8 * s;   // (the same height whichever tab is shown)
    fill(dc, 16 * s, help_top - 4 * s, (int)w - 32 * s, std::max(1, s), RGB(52, 64, 56));
    HFONT help = font(14 * s, FW_NORMAL);
    SelectObject(dc, help);
    SetTextColor(dc, RGB(176, 186, 180));
    {   // (what the row is, then on a line of its own, set off by a bar, what its value does: measure_help keeps the room)
        const Row& r = rows[tab_row_to_global(cur_tab, tab_selected[cur_tab])];
        const int about = wrap_help(dc, r.help, (int)w - 32 * s, 16 * s, help_top + 4 * s, true);
        if (const char* hint = r.value && r.hint ? r.hint(*r.value, -1) : nullptr) {
            const int y = help_top + 4 * s + about + kHintGap * s;
            SetTextColor(dc, RGB(150, 166, 156));
            const int tall = wrap_help(dc, hint, (int)w - (32 + kHintIndent) * s, (16 + kHintIndent) * s, y, true);
            fill(dc, 16 * s, y + 2 * s, 2 * s, tall - 4 * s, RGB(96, 176, 112));
        }
    }
    SelectObject(dc, body);
    SetTextColor(dc, RGB(140, 150, 144));
    const char* hint = cur_tab == TAB_MOTION ? "Arrows change   Esc goes back   Insert, F1 or Home closes"
                                             : "Arrows change   Q/E or LB/RB: tabs   Insert, F1 or Home closes";   // (fits 448 px at 16 px)
    TextOutA(dc, 16 * s, help_top + 4 * s + help_px + 10 * s, hint, (int)strlen(hint));
    if (cur_tab == TAB_GRAPHICS && restart_pending()) {   // (the rows that take a restart, changed: measure_pending's room)
        SelectObject(dc, help);
        SetTextColor(dc, RGB(235, 205, 95));
        const std::string line = restart_line(dc, (int)w - 32 * s);
        TextOutA(dc, 16 * s, help_top + 4 * s + help_px + 34 * s, line.c_str(), (int)line.size());
    }
    SelectObject(dc, old);
    DeleteObject(title);
    DeleteObject(body);
    DeleteObject(help);
}
char fps_text[48] = "";
char fps_sub[32] = "";
/// The usual overlay: a bold green number with a dark edge, nothing behind it. Hard-edged letters: see-through pixels are copied or not.
void outlined(HDC dc, int x, int y, const char* text, COLORREF colour, int edge) {
    SetTextColor(dc, RGB(6, 8, 6));
    const int n = (int)strlen(text);
    for (int dy = -edge; dy <= edge; dy++)
        for (int dx = -edge; dx <= edge; dx++)
            if (dx || dy) TextOutA(dc, x + dx, y + dy, text, n);
    SetTextColor(dc, colour);
    TextOutA(dc, x, y, text, n);
}
HFONT hard_font(int px, int weight) {
    return CreateFontA(-px, 0, 0, 0, weight, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, NONANTIALIASED_QUALITY,
                       DEFAULT_PITCH, "Segoe UI");
}
void paint_fps(HDC dc, uint32_t w, uint32_t h, int s) {
    fill(dc, 0, 0, (int)w, (int)h, RGB(0, 0, 0));
    SetBkMode(dc, TRANSPARENT);
    SetTextAlign(dc, TA_RIGHT | TA_TOP);
    const COLORREF green = RGB(72, 255, 110);
    const int right = (int)w - 2 * s;
    HFONT label = hard_font(13 * s, FW_BOLD), big = hard_font(28 * s, FW_BOLD);
    HGDIOBJ old = SelectObject(dc, label);
    outlined(dc, right, 14 * s, "FPS", green, s);
    SelectObject(dc, big);
    outlined(dc, right - 30 * s, 0, fps_text, green, s);
    if (fps_sub[0]) {
        SelectObject(dc, label);
        outlined(dc, right, 32 * s, fps_sub, RGB(226, 232, 226), s);
    }
    SelectObject(dc, old);
    DeleteObject(label);
    DeleteObject(big);
}
/// The tallest help text at scale s, a row's help and its value's line, in pixels: the menu keeps that height whichever
/// row is selected and whatever its value
int measure_help(int s) {
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT f = font(14 * s, FW_NORMAL);
    HGDIOBJ old = SelectObject(dc, f);
    int most = 0;
    for (const Row& r : rows) {   // (the row's help, and the tallest line any of its values can have)
        int hint = 0;
        for (int v = r.lo; r.value && r.hint && v <= r.hi; v++)
            for (int k = 0; const char* t = r.hint(v, k); k++)
                hint = std::max(hint, wrap_help(dc, t, (kMenuWidth - 32 - kHintIndent) * s, 0, 0, false));
        most = std::max(most, wrap_help(dc, r.help, (kMenuWidth - 32) * s, 0, 0, false) + (hint ? kHintGap * s + hint : 0));
    }
    SelectObject(dc, old);
    DeleteObject(f);
    DeleteDC(dc);
    return most;
}
/// The Graphics tab's "Restart needed" line, one line high (the room measure_pending keeps)
int pending_px = 0;
int measure_pending(int s) {
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT f = font(14 * s, FW_NORMAL);
    HGDIOBJ old = SelectObject(dc, f);
    SIZE sz{};
    GetTextExtentPoint32A(dc, "Ag", 2, &sz);
    SelectObject(dc, old);
    DeleteObject(f);
    DeleteDC(dc);
    return (int)sz.cy;
}
void redraw_menu(int s) {
    if (page == 5) {
        gdi_draw(menu_bmp, kMenuWidth * s, (unsigned)menu_hit::kAskHeight * s, paint_menu, s);
        return;
    }
    if (page == 3 || page == 4) {
        gdi_draw(menu_bmp, kMenuWidth * s, (100u + kTeleportLines * (unsigned)menu_hit::kListRowH) * s, paint_menu, s);
        return;
    }
    if (page == 2) {
        gdi_draw(menu_bmp, kMenuWidth * s, (136u + kCheatRows * (unsigned)menu_hit::kCheatRowH) * s, paint_menu, s);
        return;
    }
    if (page == 1) {
        gdi_draw(menu_bmp, 600u * s, (uint32_t)((64 + (kControls + 1) * 24 + 52) * s), paint_menu, s);
        return;
    }
    static int measured_at = 0;
    if (measured_at != s) help_px = measure_help(s), pending_px = measure_pending(s), measured_at = s;
    // (title, the tab strip, the most rows a tab has, the rule and the help text, the hint line, the restart line)
    gdi_draw(menu_bmp, (uint32_t)(kMenuWidth * s),
             (uint32_t)((44 + 32 + 8 + max_tab_rows() * 28 + 8 + 10 + 30 + 6) * s + help_px + pending_px), paint_menu, s);
}

std::atomic<int64_t> game_frames{0}, shown_frames{0};
int cur_scale = 1;

struct Upload {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    uint64_t size = 0, version = 0;
};
Upload ups[3][3];   // (the menu's, the counter's, the start-up screen's; a frame in flight's each)

void copy_over(VkCommandBuffer cmd, VkImage dst, Bitmap& b, Upload& u, int x, int y, uint32_t w, uint32_t h) {
    std::lock_guard lk(bmp_mu);
    if (!b.w || (uint32_t)x + b.w > w || (uint32_t)y + b.h > h) return;
    const uint64_t size = (uint64_t)b.w * b.h * 4;
    if (u.size < size) {
        if (!create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, u.buf, u.alloc, &u.map)) return;
        u.size = size, u.version = 0;
    }
    if (u.version != b.version) std::memcpy(u.map, b.px.data(), size), u.version = b.version;
    if (b.keyed) {
        static thread_local std::vector<VkBufferImageCopy> at;
        at = b.runs;
        for (auto& r : at) r.imageOffset.x += x, r.imageOffset.y += y;
        if (!at.empty()) vkCmdCopyBufferToImage(cmd, u.buf, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)at.size(), at.data());
        return;
    }
    VkBufferImageCopy c{};
    c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.imageOffset = {x, y, 0};
    c.imageExtent = {b.w, b.h, 1};
    vkCmdCopyBufferToImage(cmd, u.buf, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
}

// ---- (hookshot) the start-up screen (vk_startup_paint.h): centred on black ----
StartupView startup;
Bitmap startup_bmp;
std::atomic<bool> startup_on{false};
void paint_startup(HDC dc, uint32_t w, uint32_t h, int pct) { paint_startup_view(dc, w, h, pct, startup, paint_bits); }

} // namespace

void startup_screen(const StartupView* view) {
    if (!view) {
        startup_on.store(false, std::memory_order_release);
        return;
    }
    static int drawn_pct = 0;
    const int pct = startup_scale(dev.swap_extent.width, dev.swap_extent.height);
    StartupView v = *view;
    if (v.version.empty()) v.version = version_text(HOOKSHOT_VERSION);   // ("v2.0.5"; "v2.0.5 dev" between releases)
    // (the bar to whole pixels: a redraw only when it moves)
    const double span = (kStartupW - 120) * pct / 100.0;
    if (v.fraction >= 0) v.fraction = std::floor(std::min(v.fraction, 1.0) * span) / span;
    if (pct != drawn_pct || !(v == startup)) {
        startup = v;
        drawn_pct = pct;
        gdi_draw(startup_bmp, (uint32_t)(kStartupW * pct / 100), (uint32_t)(kStartupH * pct / 100), paint_startup, pct);
    }
    startup_on.store(true, std::memory_order_release);
}

void menu_init() {
    static bool done;
    if (done) return;
    done = true;
    for (auto& count : item_counts) count.store(-1, std::memory_order_relaxed);
    bool has_options = false;
    const char* cheats = getenv("HOS_OOT_CHEATS");
    cheats_enabled = cheats && !strcmp(cheats, "1");
    if (cheats_enabled) {
        const char* hearts = getenv("HOS_OOT_PROTECT_HEARTS");
        protect_hearts.store(hearts && !strcmp(hearts, "1"));
    }
    if (FILE* f = fopen(ini_path().c_str(), "r")) {
        has_options = true;
        char line[80];
        bool has_resolution = false;
        bool v2 = false;
        int old_dynres = -1;   // (the row before Resolution: 0, the full scale, is its 100%; 1 is Auto)
        while (fgets(line, sizeof line, f)) {
            char* eq = strchr(line, '=');
            if (!eq) continue;
            *eq = 0;
            for (Row& r : rows) {
                if (!r.value || strcmp(line, ini_key(r))) continue;
                const int v = atoi(eq + 1);
                *r.value = r.value == &opts.output_res ? output_choice(v) : std::min(std::max(v, r.lo), r.hi);
            }
            has_resolution |= !strcmp(line, "resolution");
            v2 |= !strcmp(line, "options_version");
            if (!strcmp(line, "dynamic_resolution")) old_dynres = atoi(eq + 1);
            if (!strcmp(line, "microphone_device")) {   // (hookshot: the microphone, by its name)
                mic_want = mic_key(eq + 1);
                while (!mic_want.empty() && (mic_want.back() == '\n' || mic_want.back() == '\r')) mic_want.pop_back();
            }
        }
        fclose(f);
        // (the saved microphone by its name, not its place; the host told the row's choice. Off or the default: no list, no SDL
        // audio at the start)
        if (!mic_want.empty()) refresh_mics();
        else {
            if (opts.mic >= 2) opts.mic = 1;
            mic_tell_host();
        }
        if (!has_resolution && old_dynres >= 0) opts.resolution = old_dynres ? 0 : 1;
        opts.rumble = rumble_step(opts.rumble);   // (a hand-edited 60: 50, on the arrows' steps)
        // (Game speed is Real time by default now: Per frame played at half speed at 30 fps, the "slow motion" players
        // reported. An options file from before kept the old default whether chosen or not: Real time for it too)
        if (!v2) opts.realtime = 1;
    }
    // An opt-in profile for 4 GB cards: keep existing options authoritative, but make a first run usable on
    // low-VRAM hardware without requiring the menu to render first.
    if (!has_options && getenv("RX_VK_LOW_SPEC") && atoi(getenv("RX_VK_LOW_SPEC")) != 0) {
        apply_preset(1);
        opts.framegen = 0;
        opts.upscale = 0;
        opts.sharpness = 0;
    }
    if (opts.preset > 0) sync_preset_budgets(opts.preset);
    if (opts.framegen == 1) opts.framegen = 2;
    if (const char* e = getenv("RX_VK_FRAMEGEN_MULT")) opts.framegen = std::min(std::max(atoi(e), 2), 8);
    if (const char* e = getenv("RX_VK_PACE")) opts.pace = atoi(e) != 0;
    if (const char* e = getenv("RX_VK_UPSCALE"))
        for (Row& r : rows)
            if (r.value == &opts.upscale) *r.value = std::min(std::max(atoi(e), r.lo), r.hi);
    if (const char* e = getenv("HOS_REALTIME_STEP")) opts.realtime = atoi(e) != 0;
    if (const char* e = getenv("HOS_DYNRES")) opts.resolution = atoi(e) != 0 ? 0 : 1;
    if (const char* e = getenv("HOS_OUTPUT_RES")) opts.output_res = output_choice(atoi(e));
    restart_init();   // (what this run started with: the rows that take a restart differ from it once changed)
}

bool resolution_auto() { return opts.resolution == 0; }
int resolution_scale() {
    const float s = upscaler_rows::percent(opts.resolution) / 100.0f;
    return s > 0 ? (int)(twelfths(s) * 10000.0f / 12.0f + 0.5f) : 0;   // (the twelfth game_size labels)
}

void menu_set_gpu_count(int count) {
    for (Row& r : rows) {
        if (r.value == &opts.gpu_device) {
            r.hi = count;
            if (opts.gpu_device > count) opts.gpu_device = 0;
        }
    }
    restart_init();
    for (int i = 0; i < kRows; i++)   // (the row's value the game started with, the cards counted)
        if (rows[i].value == &opts.gpu_device) restart_state.restart_value(restart_slot[i], opts.gpu_device);
}

bool menu_is_open() { return open_now.load(std::memory_order_relaxed); }
int menu_heart_protection() { return protect_hearts.load(std::memory_order_relaxed); }
int menu_player_flight() { return cheats_enabled ? player_flight.load(std::memory_order_relaxed) : -1; }
int menu_free_camera() { return free_camera.load(std::memory_order_relaxed); }
int menu_take_cheat_action() { return cheat_action.exchange(0, std::memory_order_relaxed); }

void menu_note_present_sizes(uint32_t picture_w, uint32_t picture_h, uint32_t window_w, uint32_t window_h) {
    const auto c = [](uint32_t v) { return (uint64_t)std::min(v, 0xffffu); };
    present_sizes.store(c(picture_w) << 48 | c(picture_h) << 32 | c(window_w) << 16 | c(window_h), std::memory_order_relaxed);
}
void menu_note_frame(bool game_frame) {
    shown_frames.fetch_add(1, std::memory_order_relaxed);
    if (game_frame) game_frames.fetch_add(1, std::memory_order_relaxed);
}

// The Fullscreen row and F11 are one setting: the row asks the window's thread (host/display.c), F11 shows in the row
void fullscreen_tick() {
    static int applied = 0, settle = 0;
    const HWND main_window = app_window();
    if (!main_window) return;
    static const bool env = getenv("HOS_FULLSCREEN") != nullptr;   // (a test's own choice: the row is left alone)
    if (env) return;
    if (opts.fullscreen != applied) {
        applied = opts.fullscreen;
        settle = 60;
        PostMessageA(main_window, WM_APP + 7, (WPARAM)applied, 0);
        return;
    }
    if (settle > 0) { settle--; return; }
    const int now = !(GetWindowLongA(main_window, GWL_STYLE) & WS_CAPTION);
    if (now != applied) {
        opts.fullscreen = applied = now;
        menu_refresh = true;
        save();
    }
}

/// A rate over the half-second that just ended. The first call only starts that interval: the frames since launch,
/// divided by one second, were shown as the rate the moment the counter was turned on.
bool fps_take(int64_t now, int64_t shown, int64_t game, int64_t* next, int64_t* last_shown, int64_t* last_game,
              double* shown_rate, double* game_rate) {
    if (now < *next) return false;
    const bool have = *next != 0;
    if (have) {
        const double secs = (now - *next + 500000000) / 1e9;
        *shown_rate = (shown - *last_shown) / secs;
        *game_rate = (game - *last_game) / secs;
    }
    *last_shown = shown;
    *last_game = game;
    *next = now + 500000000;
    return have;
}

void menu_option_set(int id, int value) {
    if (id >= OOT_CHEAT_ITEM_BASE && id < OOT_CHEAT_ITEM_BASE + kItems) {   // (the game side's inventory counts)
        if (item_counts[id - OOT_CHEAT_ITEM_BASE].exchange(value, std::memory_order_relaxed) != value && page == 4)
            menu_refresh = true;
        return;
    }
    if (id == HOS_OPT_CHEAT_FLIGHT) {
        if (cheats_enabled) {
            int expected = value ? 0 : 1;
            player_flight.compare_exchange_strong(expected, value != 0, std::memory_order_relaxed);
        }
        return;
    }
    if (id != 7) return;
    opts.mouse_sens = std::min(std::max(value, 10), 500);   // (the keys work while the menu is closed: it shows it when opened)
    save();
}

/// A list page's line (from the top of the list, headings included) to its item: -1 a section's heading
int list_item(int line, const TeleportSection* sec, int sections) {
    int section = 0;
    while (section + 1 < sections && line >= sec[section + 1].first + section + 1) section++;
    return line == sec[section].first + section ? -1 : line - section - 1;
}
/// (hookshot) the mouse on a list page (the teleport list, the inventory): the item under the cursor chosen, a click on it
/// as Enter (on an amount's left or right half, with amounts, as Left or Right), the wheel moves the choice, a right click
/// goes back
void list_mouse(const MkwVkPointer& ptr, int mx, int my, int mw, int mh, int scale, bool moved, bool amounts, int n,
                int sections, const TeleportSection* sec, int scroll, int& choice, bool* press, bool& changed) {
    const int shown = std::min(n + sections - scroll, kTeleportLines);
    const menu_hit::Hit hit = menu_hit::list_lines(mx, my, scale, mw, mh, shown, list_value_l);
    const int item = hit.part == menu_hit::kListLine ? list_item(scroll + hit.index, sec, sections) : -1;
    if (item >= 0 && (moved || ptr.pressed) && item != choice) choice = item, changed = true;
    if ((ptr.pressed & HOS_MOUSE_LEFT) && item >= 0) {
        if (amounts && item_kinds[item] && hit.side != 0) press[hit.side < 0 ? 3 : 4] = true;
        else press[7] = true;
    }
    if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
    if (ptr.wheel) choice = std::clamp(choice - ptr.wheel, 0, n - 1), changed = true;
}
/// Scrolls the teleport list so the chosen map (and its section heading, at the top) is in view
void teleport_reveal() {
    int section = 0;
    while (section + 1 < kTeleportSections && teleport_choice >= teleport_sections[section + 1].first) section++;
    const int line = teleport_choice + section + 1;
    if (teleport_choice == teleport_sections[section].first) teleport_scroll = std::min(teleport_scroll, line - 1);
    if (line < teleport_scroll) teleport_scroll = line;
    if (line >= teleport_scroll + kTeleportLines) teleport_scroll = line - kTeleportLines + 1;
    teleport_scroll = std::max(0, std::min(teleport_scroll, kTeleports + kTeleportSections - kTeleportLines));
}
/// Scrolls the inventory so the chosen item (and its section heading, at the top) is in view
void item_reveal() {
    int section = 0;
    while (section + 1 < kItemSections && item_choice >= item_sections.s[section + 1].first) section++;
    const int line = item_choice + section + 1;
    if (item_choice == item_sections.s[section].first) item_scroll = std::min(item_scroll, line - 1);
    if (line < item_scroll) item_scroll = line;
    if (line >= item_scroll + kTeleportLines) item_scroll = line - kTeleportLines + 1;
    item_scroll = std::max(0, std::min(item_scroll, kItems + kItemSections - kTeleportLines));
}
void menu_tick() {
    fullscreen_tick();
    enum { K_N = 17 };
    static bool held[K_N];
    static const int keys[K_N] = {VK_INSERT, VK_UP, VK_DOWN, VK_LEFT, VK_RIGHT, VK_ESCAPE, VK_F1,   // (F1: for keyboards without Insert)
                                  VK_RETURN, VK_DELETE, VK_BACK,
                                  'Q', 'E', VK_TAB,   // (the tabs: Q and E, Tab and Shift+Tab)
                                  VK_PRIOR, VK_NEXT, VK_HOME, VK_END};   // (the long lists: a page, the ends)
    HWND fg = GetForegroundWindow();
    DWORD pid = 0;
    if (fg) GetWindowThreadProcessId(fg, &pid);
    const bool mine = pid == GetCurrentProcessId();
    bool press[K_N] = {};
    for (int i = 0; i < K_N; i++) {
        const bool down = mine && (GetAsyncKeyState(keys[i]) & 0x8000) != 0;
        const bool queued = host && host->take_key_press && host->take_key_press(keys[i]);
        press[i] = host && host->take_key_press ? mine && queued : down && !held[i];
        held[i] = down;
    }
    // (hookshot) the gamepads, by position: Home, or Select and Start together, as Insert; the D-pad as the arrows,
    // the bottom face button as Enter, the right one as Esc (not while a binding waits for its button)
    static uint32_t pad_held = 0;
    const uint32_t raw = mine && host && host->pad_raw ? host->pad_raw() : 0, pad_press = raw & ~pad_held;
    const uint32_t chord = 1u << HOS_PAD_BACK | 1u << HOS_PAD_START;
    const bool chord_now = (raw & chord) == chord && (pad_held & chord) != chord;
    pad_held = raw;
    auto pad = [&](int p) { return listening != 1 && (pad_press >> p & 1) != 0; };
    if (pad(HOS_PAD_GUIDE) || chord_now) press[0] = true;
    {   // holding an arrow or the D-pad repeats it: after 0.35 s, 12 a second, then 30 a second after 1.5 s held
        static int64_t held_since[5], next_repeat[5];
        const int pads[5] = {0, HOS_PAD_UP, HOS_PAD_DOWN, HOS_PAD_LEFT, HOS_PAD_RIGHT};
        const int64_t t = prof_now();
        for (int i = 1; i <= 4; i++) {
            const bool down = held[i] || (open_now.load() && listening != 1 && (raw >> pads[i] & 1));
            if (!down) { held_since[i] = 0; continue; }
            if (!held_since[i]) { held_since[i] = t, next_repeat[i] = t + 350000000LL; continue; }
            if (t >= next_repeat[i] && open_now.load() && listening == 0) {
                press[i] = true;
                next_repeat[i] = t + (t - held_since[i] > 1500000000LL ? 33000000LL : 83000000LL);
            }
        }
    }
    if (open_now.load() && listening != 1) {
        press[1] |= pad(HOS_PAD_UP), press[2] |= pad(HOS_PAD_DOWN), press[3] |= pad(HOS_PAD_LEFT);
        press[4] |= pad(HOS_PAD_RIGHT), press[5] |= pad(HOS_PAD_EAST), press[7] |= pad(HOS_PAD_SOUTH);
    }
    {   // RX_VK_MENU_SCRIPT (test runs, which never have the focus): timed menu keys, "<seconds>:<INS|UP|DOWN|LEFT|RIGHT|ESC|ENTER|TABL|TABR>,..."
        static const char* next = getenv("RX_VK_MENU_SCRIPT");
        static const int64_t start = prof_now();
        double at;
        char key[8];
        int n = 0;
        if (next && sscanf(next, "%lf:%7[A-Z]%n", &at, key, &n) == 2 && (prof_now() - start) / 1e9 >= at) {
            static const char* names[] = {"INS", "UP", "DOWN", "LEFT", "RIGHT", "ESC", "", "ENTER", "", "", "TABL", "TABR"};
            for (int i = 0; i < 12; i++) press[i] |= *names[i] && !strcmp(key, names[i]);
            next = next[n] == ',' ? next + n + 1 : nullptr;
        }
    }
    bool tab_left = press[10] || pad(HOS_PAD_LSHOULDER) || pad(HOS_PAD_LTRIGGER);
    bool tab_right = press[11] || press[12] || pad(HOS_PAD_RSHOULDER) || pad(HOS_PAD_RTRIGGER);
    if (press[12] && (GetAsyncKeyState(VK_SHIFT) & 0x8000)) tab_left = true, tab_right = false;
    const int scale = scale_of(dev.swap_extent.height);
    bool changed = scale != cur_scale || menu_refresh;
    menu_refresh = false;
    cur_scale = scale;
    // (hookshot) the mouse, while the menu is open: the cursor in the menu's pixels (mx, my), whether it moved since the
    // last frame, and its clicks and wheel since then (vk_menu_hit.h)
    MkwVkPointer ptr{};
    int mx = -1, my = -1, mw = 0, mh = 0;
    bool mouse = false, moved = false;
    if (open_now.load() && host && host->pointer) {
        mouse = host->pointer(&ptr) != 0;
        {
            std::lock_guard lk(bmp_mu);
            mw = (int)menu_bmp.w, mh = (int)menu_bmp.h;
        }
        int ix = 0, iy = 0, x0 = 0, y0 = 0;
        menu_hit::to_image(ptr.x, ptr.y, ptr.width, ptr.height, (int)dev.swap_extent.width, (int)dev.swap_extent.height,
                           &ix, &iy);
        menu_hit::origin((int)dev.swap_extent.width, (int)dev.swap_extent.height, mw, mh, &x0, &y0);
        if (mouse) mx = ix - x0, my = iy - y0;
        static int last_x = -1, last_y = -1;
        moved = mouse && mouse_read && (mx != last_x || my != last_y);   // (not the first read: a cursor resting over the
        last_x = mx, last_y = my, mouse_read = true;                     // menu as it opens leaves the selection alone)
        if (!mouse) ptr.pressed = 0, ptr.wheel = 0;
    }
    // (closing with a setting that takes a restart changed: the restart question first, Later already chosen on it)
    auto later = [&] {
        int now[mr::kMax];
        restart_values(now);
        restart_state.choose_later(now);
        open_now = false, page = 0, changed = true;
    };
    if ((press[0] || press[6]) && open_now.load() && page == 5) {
        later();
    } else if ((press[0] || press[6]) && open_now.load() && restart_ask()) {
        page = 5, ask_sel = 0, listening = 0, changed = true;
    } else if (press[0] || press[6]) {
        open_now = !open_now.load(), changed = true;
        page = 0, listening = 0;
        // (the layout row shows the controls as they are: host/nn_hid.c keeps them, the ini's pad_ lines, and a saved
        // layout is not applied over them at start)
        if (open_now.load()) opts.pad_layout = detect_pad_layout();
        if (open_now.load()) dlss_probe();   // (whether the Upscaler row can offer DLSS here: NGX asked once)
        if (open_now.load()) refresh_mics();   // (the microphones as they are now)
    }
    if (open_now.load() && page == 5) {   // (hookshot) the restart question: Later (the default) or Restart now
        if (mouse) {   // (a button under the cursor chosen, a click on it pressed; the wheel changes it, a right click is Later)
            const menu_hit::Hit hit = menu_hit::ask(mx, my, scale, mw, mh);
            if (hit.part == menu_hit::kButton && (moved || ptr.pressed) && hit.index != ask_sel) ask_sel = hit.index, changed = true;
            if ((ptr.pressed & HOS_MOUSE_LEFT) && hit.part == menu_hit::kButton) press[7] = true;
            if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
            if (ptr.wheel) ask_sel ^= 1, changed = true;
        }
        if (press[1] || press[2] || press[3] || press[4] || tab_left || tab_right) ask_sel ^= 1, changed = true;
        if (press[5] || (press[7] && ask_sel == 0)) {
            later();
        } else if (press[7]) {   // (Restart now: everything saved, then the window closes as the player would close it)
            logf("menu: restart now, to apply %s", restart_list(", ").c_str());
            save();
            pipeline_cache_save(true);
            open_now = false, page = 0, changed = true;
            PostMessageA(app_window(), WM_APP + 9, 0, 0);
        }
        if (changed && open_now.load()) redraw_menu(scale);
    } else if (open_now.load() && page == 1) {   // (hookshot) the Controls page
        if (!can_bind()) page = 0;
        else if (listening) {
            if (press[5] || prof_now() > listen_end) listening = 0, changed = true;   // (Esc, or nothing for a while)
            else if (listening == 1) {
                if (!listen_armed) listen_armed = (raw & ~(1u << HOS_PAD_GUIDE)) == 0;
                else if (const uint32_t p = pad_press & ~(1u << HOS_PAD_GUIDE)) {
                    int bit = 0;
                    while (!(p >> bit & 1)) bit++;
                    host->bind(HOS_BIND_PAD, controls[csel].button, 0, bit);
                    listening = 0, changed = true;
                    opts.pad_layout = detect_pad_layout();
                    save();
                }
            } else {
                int down = 0;
                for (int k : bindable_keys())
                    if (mine && (GetAsyncKeyState(k) & 0x8000)) {
                        down = k;
                        break;
                    }
                if (!listen_armed) listen_armed = !down;
                else if (down) {
                    host->bind(HOS_BIND_KEY, controls[csel].button, 0, down);
                    host->bind(HOS_BIND_KEY, controls[csel].button, 1, 0);
                    listening = 0, changed = true;
                    save();
                }
            }
        } else {
            if (mouse) {   // (the mouse: the row under it selected, a click on a binding listens for its new input, on
                           // Reset resets; the wheel moves the selection, a right click goes back)
                const menu_hit::Hit hit = menu_hit::controls(mx, my, scale, mw, mh, kControls);
                const bool on = hit.part == menu_hit::kCtlRow || hit.part == menu_hit::kCtlCell;
                if (on && (moved || ptr.pressed)) {
                    if (hit.index != csel) csel = hit.index, changed = true;
                    if (hit.part == menu_hit::kCtlCell && !keys_only(csel) && ccol != hit.side) ccol = hit.side, changed = true;
                }
                if (ptr.pressed & HOS_MOUSE_LEFT) {
                    if (hit.part == menu_hit::kCtlCell || (hit.part == menu_hit::kCtlRow && hit.index == kControls))
                        press[7] = true;   // (as Enter)
                }
                if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
                if (ptr.wheel) csel = ((csel - ptr.wheel) % (kControls + 1) + kControls + 1) % (kControls + 1), changed = true;
            }
            if (press[5]) page = 0, changed = true;
            if (press[1]) csel = (csel + kControls) % (kControls + 1), changed = true;
            if (press[2]) csel = (csel + 1) % (kControls + 1), changed = true;
            if ((press[3] || press[4]) && csel < kControls && !keys_only(csel)) ccol ^= 1, changed = true;
            if (keys_only(csel)) ccol = 1;   // (the stick's and the camera's directions: keys only)
            if (press[7]) {
                if (csel == kControls) {
                    host->bind(HOS_BIND_PAD, -1, 0, 0);
                    opts.pad_layout = 0;
                    save();
                }
                else listening = ccol ? 2 : 1, listen_armed = false, listen_end = prof_now() + 6000000000LL;
                changed = true;
            }
            if ((press[8] || press[9]) && csel < kControls) {
                if (ccol == 0) {
                    host->bind(HOS_BIND_PAD, controls[csel].button, 0, -1);
                    opts.pad_layout = detect_pad_layout();
                }
                else for (int k = 0; k < 2; k++) host->bind(HOS_BIND_KEY, controls[csel].button, k, 0);
                changed = true;
                save();
            }
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 2) {
        // (the Cheats rows that choose with Left and Right: Set time, Inventory item, Change quantity, Teleport)
        auto chooses = [](int row) { return row == 4 || row == 5 || row == 6 || row == 8; };
        if (mouse) {   // (the mouse: the row under it selected, a click as Enter, or on a choice's left or right half as
                       // Left or Right; the wheel moves the selection, or over a choice changes it; a right click goes back)
            const menu_hit::Hit hit = menu_hit::cheats(mx, my, scale, mw, mh, kCheatRows, cheat_value_l);
            const bool on = hit.part == menu_hit::kCheatRow;
            if (on && (moved || ptr.pressed) && hit.index != cheat_selected) cheat_selected = hit.index, changed = true;
            const bool on_choice = on && hit.index == cheat_selected && chooses(hit.index) && hit.side != 0;
            if ((ptr.pressed & HOS_MOUSE_LEFT) && on) {
                if (on_choice) press[hit.side < 0 ? 3 : 4] = true;
                else press[7] = true;
            }
            if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;
            if (ptr.wheel && on_choice) press[ptr.wheel > 0 ? 4 : 3] = true;   // (up for the wheel away, as the options')
            else if (ptr.wheel) cheat_selected = ((cheat_selected - ptr.wheel) % kCheatRows + kCheatRows) % kCheatRows, changed = true;
        }
        if (press[5]) page = 0, changed = true;
        else {
            if (press[1] || press[2]) {
                cheat_selected = (cheat_selected + kCheatRows + (press[2] ? 1 : -1)) % kCheatRows;
                changed = true;
            }
            if (cheat_selected < 2 && (press[3] || press[4] || press[7])) {
                if (cheat_selected == 0) protect_hearts.store(!protect_hearts.load());
                else player_flight.store(player_flight.load() < 0 ? 0 : -1);
                changed = true;
            }
            if (cheat_selected == 4 && (press[3] || press[4])) {
                clock_hour = (clock_hour + (press[4] ? 1 : 23)) % 24;
                changed = true;
            }
            if (cheat_selected == 5 && (press[3] || press[4])) {
                pouch_item = (pouch_item + kPouchItems + (press[4] ? 1 : -1)) % kPouchItems;
                changed = true;
            }
            if (cheat_selected == 6 && (press[3] || press[4])) {
                const int step = press[4] ? 1 : -1;
                pouch_delta = std::clamp(pouch_delta + step, -10, 10);
                if (!pouch_delta) pouch_delta = step;
                changed = true;
            }
            if (press[7] && cheat_selected >= 2 && cheat_selected <= 4) {
                cheat_action.store(cheat_selected == 4 ? 100 + clock_hour : cheat_selected);
            }
            if (press[7] && cheat_selected == 6) {
                cheat_action.store(200 + pouch_item * 32 + pouch_delta + 10);
            }
            if (press[7] && cheat_selected == 7) {
                free_camera.store(!free_camera.load());
                changed = true;
            }
            if (cheat_selected == 8 && (press[3] || press[4])) {
                teleport_choice = (teleport_choice + kTeleports + (press[4] ? 1 : -1)) % kTeleports;
                changed = true;
            }
            if (press[7] && cheat_selected == 8) page = 3, teleport_reveal(), changed = true;
            if (press[7] && cheat_selected == 10) page = 4, item_reveal(), changed = true;
            if (press[7] && cheat_selected == 9) {
                cheat_action.store(999);   // (host/hos_main.c OOT_AGE_ACTION)
                open_now = false;
                changed = true;
            }
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 4) {   // the inventory
        if (mouse)
            list_mouse(ptr, mx, my, mw, mh, scale, moved, true, kItems, kItemSections, item_sections.s, item_scroll,
                       item_choice, press, changed);
        if (press[5]) page = 2, changed = true;
        else {
            if (press[1] || press[2]) {
                item_choice = (item_choice + kItems + (press[2] ? 1 : -1)) % kItems;
                changed = true;
            }
            if (press[13] || press[14] || press[15] || press[16]) {   // (Page Up / Page Down, Home / End)
                item_choice = press[15] ? 0 : press[16] ? kItems - 1
                            : std::clamp(item_choice + (press[14] ? 1 : -1) * (kTeleportLines - 2), 0, kItems - 1);
                changed = true;
            }
            if (tab_left || tab_right) {
                int section = 0;
                while (section + 1 < kItemSections && item_choice >= item_sections.s[section + 1].first) section++;
                if (tab_right || item_choice == item_sections.s[section].first)
                    section = (section + kItemSections + (tab_left ? -1 : 1)) % kItemSections;
                item_choice = item_sections.s[section].first;
                changed = true;
            }
            if (press[7]) cheat_action.store(OOT_CHEAT_ITEM_ACTION + item_choice * 256 + 200);
            else if (press[3] || press[4]) cheat_action.store(OOT_CHEAT_ITEM_ACTION + item_choice * 256 + 64 + (press[4] ? 1 : -1));
            if (changed) item_reveal();
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load() && page == 3) {   // the teleport list
        if (mouse)
            list_mouse(ptr, mx, my, mw, mh, scale, moved, false, kTeleports, kTeleportSections, teleport_sections,
                       teleport_scroll, teleport_choice, press, changed);
        if (press[5]) page = 2, changed = true;
        else if (press[7]) {
            cheat_action.store(1000 + teleport_choice);
            open_now = false, page = 2;   // (back to the game: the map change starts with an event)
            changed = true;
        } else {
            if (press[1] || press[2]) {
                teleport_choice = (teleport_choice + kTeleports + (press[2] ? 1 : -1)) % kTeleports;
                changed = true;
            }
            if (press[13] || press[14] || press[15] || press[16]) {   // (Page Up / Page Down, Home / End)
                teleport_choice = press[15] ? 0 : press[16] ? kTeleports - 1
                                : std::clamp(teleport_choice + (press[14] ? 1 : -1) * (kTeleportLines - 2), 0, kTeleports - 1);
                changed = true;
            }
            if (press[3] || press[4] || tab_left || tab_right) {   // (to the first map of the previous or next section)
                int section = 0;
                while (section + 1 < kTeleportSections && teleport_choice >= teleport_sections[section + 1].first) section++;
                const bool back = press[3] || tab_left;
                if (!back || teleport_choice == teleport_sections[section].first)
                    section = (section + kTeleportSections + (back ? -1 : 1)) % kTeleportSections;
                teleport_choice = teleport_sections[section].first;
                changed = true;
            }
            if (changed) teleport_reveal();
        }
        if (changed) redraw_menu(scale);
    } else if (open_now.load()) {
        if (mouse) {   // (the mouse: see the top of this file)
            const int count = tab_row_count(cur_tab);
            const menu_hit::Hit hit = menu_hit::options(mx, my, scale, mw, mh, TAB_COUNT, count, row_value_l, row_slider);
            const int ht = hit.part == menu_hit::kTab ? hit.index : hit.part == menu_hit::kTabPrev ? 3
                         : hit.part == menu_hit::kTabNext ? 4 : -1;
            if (ht != hover_tab) hover_tab = ht, changed = true;
            int& sel = tab_selected[cur_tab];
            const bool on_row = hit.part == menu_hit::kRow || hit.part == menu_hit::kSlider;
            if (drag_row >= 0) {   // (a slider held: it follows the cursor's x until let go)
                Row& d = rows[tab_row_to_global(cur_tab, drag_row)];
                if (!(ptr.held & HOS_MOUSE_LEFT) || !d.value) {
                    drag_row = -1;
                    save();
                } else {
                    const int v = menu_hit::slider_value(menu_hit::slider_drag(mx, scale), d.lo, d.hi, d.step);
                    if (v != *d.value) set_row(d, v, v > *d.value ? 1 : -1, false), changed = true;
                }
            } else if (on_row && (moved || ptr.pressed) && hit.index != sel) {
                sel = hit.index, changed = true;
            }
            if ((ptr.pressed & HOS_MOUSE_LEFT) && drag_row < 0) {
                if (hit.part == menu_hit::kTabPrev) tab_left = true;
                else if (hit.part == menu_hit::kTabNext) tab_right = true;
                else if (hit.part == menu_hit::kTab && hit.index != cur_tab) cur_tab = hit.index, changed = true;
                else if (hit.part == menu_hit::kSlider) {
                    Row& d = rows[tab_row_to_global(cur_tab, hit.index)];
                    drag_row = hit.index;
                    const int v = menu_hit::slider_value(hit.permille, d.lo, d.hi, d.step);
                    if (v != *d.value) set_row(d, v, v > *d.value ? 1 : -1, false);
                    changed = true;
                } else if (hit.part == menu_hit::kRow) {
                    Row& d = rows[tab_row_to_global(cur_tab, hit.index)];
                    if (!d.value) press[7] = true;   // (a page: as Enter)
                    else if (d.hi - d.lo == d.step || hit.side >= 0) press[4] = true;   // (a toggle flips; the right half, or the name, as Right)
                    else press[3] = true;                                               // (the left half as Left)
                }
            }
            if (ptr.pressed & HOS_MOUSE_RIGHT) press[5] = true;   // (as Esc)
            if (ptr.wheel && drag_row < 0) {
                Row& d = rows[tab_row_to_global(cur_tab, sel)];
                const bool on_value = on_row && hit.index == sel && d.value &&
                                      (hit.part == menu_hit::kSlider || hit.side != 0);
                if (hit.part == menu_hit::kTab || hit.part == menu_hit::kTabPrev || hit.part == menu_hit::kTabNext) {
                    if (ptr.wheel > 0) tab_left = true;
                    else tab_right = true;
                } else if (on_value) {   // (over a value: it changes, up for the wheel away; a slider or a scale stops at its ends)
                    const int dir = ptr.wheel > 0 ? 1 : -1;
                    for (int n = ptr.wheel > 0 ? ptr.wheel : -ptr.wheel; n > 0; n--) {
                        const int v = *d.value + dir * d.step;
                        if ((is_slider(d) || stops_at_ends(d)) && (v < d.lo || v > d.hi)) break;
                        set_row(d, v, dir);
                    }
                    changed = true;
                } else if (on_row || hit.part == menu_hit::kNone) {   // (elsewhere on the menu: the selection moves)
                    sel = ((sel - ptr.wheel) % count + count) % count, changed = true;
                }
            }
        }
        if (press[5] && cur_tab == TAB_MOTION) cur_tab = TAB_CONTROLS, press[5] = false, changed = true;   // (back)
        if (press[5] && pending) pending = nullptr, changed = true;   // (a Restart or Quit waiting for its second press: cancelled)
        else if (press[5] && restart_ask()) page = 5, ask_sel = 0, changed = true;   // (the restart question first)
        else if (press[5]) open_now = false;
        // (the tabs: LB and RB, Q and E, or Tab; from the Motion page, as from Controls)
        if ((tab_left || tab_right) && cur_tab == TAB_MOTION) cur_tab = TAB_CONTROLS;
        if (tab_left) cur_tab = (cur_tab + TAB_COUNT - 1) % TAB_COUNT, changed = true;
        else if (tab_right) cur_tab = (cur_tab + 1) % TAB_COUNT, changed = true;
        const int tab_count = tab_row_count(cur_tab);
        int& selected = tab_selected[cur_tab];
        if (press[1]) selected = (selected + tab_count - 1) % tab_count, changed = true;
        if (press[2]) selected = (selected + 1) % tab_count, changed = true;
        Row& r = rows[tab_row_to_global(cur_tab, selected)];
        const bool confirm_selected = &r == &rows[kQuitRow] || &r == &rows[kRestartRow];
        if (pending && (pending != &r || !mine)) pending = nullptr, changed = true;
        if (press[7] && confirm_selected) {
            if (mine && !press[5]) {
                if (pending != &r) {
                    pending = &r, changed = true;
                } else {   // (as Restart now: saved first, as the window's own close does not force the pipeline cache)
                    save();
                    pipeline_cache_save(true);
                    pending = nullptr;
                    if (&r == &rows[kRestartRow]) {   // (the restart question's Restart now: the window closes, then the game starts again)
                        logf("menu: restart");
                        open_now = false, changed = true;
                        if (const HWND window = app_window()) PostMessageA(window, WM_APP + 9, 0, 0);
                    } else if (const HWND window = app_window()) {
                        PostMessageA(window, WM_CLOSE, 0, 0);
                    }
                }
            }
        } else if (press[7] && &r == &rows[kCheatsRow]) page = 2, changed = true;
        else if (press[7] && is_motion_row(r)) cur_tab = TAB_MOTION, tab_selected[TAB_MOTION] = 0, press[7] = false, changed = true;
        else if (press[7] && !r.value && can_bind()) page = 1, csel = 0, ccol = 0, changed = true;
        if ((press[3] || press[4] || press[7]) && r.value) {   // (Enter, or the bottom face button, as Right)
            if (press[7]) press[4] = true;
            set_row(r, *r.value + (press[4] ? r.step : -r.step), press[4] ? 1 : -1);
            changed = true;
        }
        const int64_t now = prof_now();
        if (changed) {   // (a new row or value: its name from the start, after a rest)
            marquee_pos = 0, marquee_next = now + kMarqueeRest;
        } else if (marquee_max > 0 && now >= marquee_next) {
            if (marquee_pos >= marquee_max) marquee_pos = 0, marquee_next = now + kMarqueeRest;   // (back to the start)
            else if (++marquee_pos >= marquee_max) marquee_next = now + kMarqueeRest;            // (rest at the end)
            else marquee_next = now + kMarqueeStep;
            changed = true;
        }
        if (changed) redraw_menu(scale);
    }
    {   // (hookshot) the window lets the mouse go while the menu is open, and takes it back after (host/display.c)
        static bool told = false;
        const bool now_open = open_now.load();
        if (!now_open) {
            if (drag_row >= 0) save();   // (a slider let go off the window: kept now)
            hover_tab = -1, drag_row = -1, mouse_read = false;
            pending = nullptr;
        }
        if (now_open != told && app_window()) {
            told = now_open;
            PostMessageA(app_window(), WM_APP + 8, (WPARAM)now_open, 0);
        }
    }
    if (opts.show_fps) {
        static int64_t next = 0, last_game = 0, last_shown = 0;
        const int64_t now = prof_now();
        double shown = 0, game = 0;
        if (fps_take(now, shown_frames.load(), game_frames.load(), &next, &last_shown, &last_game, &shown, &game)) {
            snprintf(fps_text, sizeof fps_text, "%.0f", shown);
            if (opts.framegen >= 2) snprintf(fps_sub, sizeof fps_sub, "game %.0f", game);
            else fps_sub[0] = 0;
            fps_bmp.keyed = true;
            gdi_draw(fps_bmp, 120u * scale, 54u * scale, paint_fps, scale);
        }
    }
}

void menu_record(VkCommandBuffer cmd, VkImage dst, int slot, uint32_t w, uint32_t h) {
    if (startup_on.load(std::memory_order_acquire)) {   // (hookshot) under the counter and the menu
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        std::unique_lock lk(bmp_mu);
        const uint32_t sw = startup_bmp.w, sh = startup_bmp.h;
        lk.unlock();
        if (sw) copy_over(cmd, dst, startup_bmp, ups[2][slot], (int)(w - sw) / 2, (int)(h - sh) / 2, w, h);
    }
    if (opts.show_fps && !startup_on.load(std::memory_order_relaxed)) {   // (no game frames yet: not "0 FPS" over the start-up screen)
        // The game image was just blitted (or cleared) into dst. Its write must
        // finish before the FPS pixels overwrite it; command order alone does
        // not order these transfer writes. The menu below has its own barrier.
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        std::unique_lock lk(bmp_mu);
        const uint32_t fw = fps_bmp.w;
        lk.unlock();
        if (fw) copy_over(cmd, dst, fps_bmp, ups[1][slot], (int)w - (int)fw - (int)(12 * cur_scale), (int)(10 * cur_scale), w, h);
    }
    if (open_now.load(std::memory_order_relaxed)) {
        std::unique_lock lk(bmp_mu);
        const uint32_t mw = menu_bmp.w, mh = menu_bmp.h;
        lk.unlock();
        // (one copy finished before the next: on a Radeon the menu came out with its top torn away beside the FPS counter)
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
        if (mw) copy_over(cmd, dst, menu_bmp, ups[0][slot], (int)(w - mw) / 2, (int)(h - mh) / 2, w, h);
    }
}

} // namespace mkwvk
