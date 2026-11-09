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
#include "vk_dlss.h"
#include "oot_cheat_items.h"
#include "vk_startup_paint.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)
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
    TAB_COUNT = 3
};

struct Row {
    MenuTab tab;
    const char* name;
    int* value;
    int lo, hi, step;
    const char* (*text)(int v);
    const char* help;   // (shown under the list while the row is selected)
    bool preset_exempt = false;   // (a change here leaves the Quality preset alone: the controls, the window, the counter)
};
const char* tx_framegen(int v) {
    static const char* t[] = {"Off", "", "2x", "3x", "4x", "5x", "6x", "7x", "8x"};
    return t[v];
}
const char* tx_onoff(int v) { return v ? "On" : "Off"; }
const char* tx_mousecam(int v) { return v ? "Native" : "As the stick"; }
const char* tx_sens(int v) {
    static char b[12];
    snprintf(b, sizeof b, "%.2f", v / 100.0);
    return b;
}
const char* tx_upscale(int v) {
    static const char* t[] = {"Off", "Catmull-Rom", "Lanczos"};
    if ((unsigned)v >= sizeof t / sizeof t[0]) return t[0];   // (a setting past the row: the label, not the next bytes)
    return t[v];
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
// (hookshot) The Resolution row: 0 Auto, then 1-9 for 12/12 down to 4/12 of the output, every twelfth the game draws exactly (below)
constexpr int kResSteps = 9;
int res_twelfths(int v) { return v >= 1 && v <= kResSteps ? 13 - v : 0; }
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
    if (v <= 0 || v > kResSteps) return "Auto (dynamic)";
    uint32_t w, h;
    game_size(res_twelfths(v) / 12.0f, w, h);
    snprintf(b, sizeof b, "%d%% (%ux%u)", (res_twelfths(v) * 100 + 6) / 12, w, h);
    return b;
}
// (hookshot, vk_dlss.cpp) NVIDIA DLSS in place of the game's upscale networks
const char* tx_dlss(int v) {
    const int st = dlss_state();
    if (st == 0) return v ? "On (restart)" : "Off";
    if (st == 1) return v ? "On (not available here)" : "Off";
    return v ? "On" : "Off";
}
const char* tx_dlss_preset(int v) {
    static const char* t[] = {"NVIDIA's choice", "J", "K", "L", "M"};
    return (unsigned)v < 5 ? t[v] : t[0];
}
const char* tx_output_res(int v) {
    static char b[48];
    v = output_choice(v);
    const Out o = kOutputs[v];
    snprintf(b, sizeof b, "%ux%u (%srestart)", o.w, o.h, v ? "" : "game, ");
    return b;
}
const char* tx_speed(int v) { return v ? "Real time" : "Per frame"; }
const char* tx_text_speed(int v) {
    static const char* t[] = {"Normal", "2x", "4x", "Instant"};
    if ((unsigned)v >= sizeof t / sizeof t[0]) return t[0];
    return t[v];
}
const char* tx_handheld(int v) { return v ? "On (restart)" : "Off (restart)"; }
const char* tx_ultrawide(int v) { return v ? "Monitor's shape (restart)" : "16:9 (restart)"; }
int gpu_device_at_start = 0;   // (the row's value the game started with: menu_set_gpu_count)
const char* tx_gpu(int v) {
    static char buf[300];
    const int idx = v - 1;
    if (v <= 0) snprintf(buf, sizeof buf, "Auto");
    else if (idx < (int)gpu_device_names.size()) snprintf(buf, sizeof buf, "%d: %s", idx, gpu_device_names[idx].c_str());
    else snprintf(buf, sizeof buf, "%d: Device %d", idx, idx);
    // (a GPU other than the one in use: it takes a restart, said where the choice is, as Handheld and Ultrawide say it.
    // The whole name shows: the selected row slides through it, see marquee_pos)
    if (v != gpu_device_at_start) strncat(buf, " (restart)", sizeof buf - strlen(buf) - 1);
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
const char* tx_rumble(int v) { return v ? tx_percent(v) : "Off"; }
const char* tx_robust(int v) {
    static const char* t[] = {"Auto (restart)", "Off (restart)", "On (restart)"};
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
void apply_preset(int p) {
    switch (p) {
    case 1: // Minimum (4GB VRAM / iGPU)
        opts.resolution = 7; opts.upscale = 2; opts.sharpness = 50;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 1;
        break;
    case 2: // Low (6GB VRAM)
        opts.resolution = 4; opts.upscale = 2; opts.sharpness = 35;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 1;
        break;
    case 3: // Balanced (8GB VRAM)
        opts.resolution = 1; opts.upscale = 1; opts.sharpness = 25;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    case 4: // High (12GB VRAM)
        opts.resolution = 1; opts.upscale = 2; opts.sharpness = 30;
        opts.framegen = 2;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    case 5: // Ultra (16GB+ VRAM)
        opts.resolution = 1; opts.upscale = 2; opts.sharpness = 40;
        opts.framegen = 3;   opts.pace = 1;    opts.realtime = 1; opts.handheld = 0;
        break;
    default:
        break;
    }
    sync_preset_budgets(p);
}

Row rows[] = {
    // Graphics (11 rows)
    {TAB_GRAPHICS, "Quality preset", &opts.preset, 0, 5, 1, tx_preset,
     "Automatically configures resolution, upscaling, frame generation, and dynamic VRAM limits tailored to your GPU tier."},
    {TAB_GRAPHICS, "Fullscreen", &opts.fullscreen, 0, 1, 1, tx_onoff,
     "Fills the whole screen with the game, without the window's border. The same as F11.", true},
    {TAB_GRAPHICS, "GPU device", &opts.gpu_device, 0, 8, 1, tx_gpu,
     "Selects which graphics card Vulkan uses. Auto picks the best dedicated GPU. Applies after restart.", true},
    {TAB_GRAPHICS, "Frame generation", &opts.framegen, 0, 8, 1, tx_framegen,
     "Makes extra frames between the game's own (2x up to 8x), for smoother motion. Adds a little input delay, and fast motion or camera cuts "
     "can show artifacts."},
    {TAB_GRAPHICS, "Frame pacing", &opts.pace, 0, 1, 1, tx_onoff,
     "Shows each frame at an even interval instead of as soon as it is ready. Less stutter, at up to one frame of delay."},
    {TAB_GRAPHICS, "Resolution", &opts.resolution, 0, kResSteps, 1, tx_resolution,
     "How sharp the 3D is drawn. Auto lets the game lower it when the GPU is busy. A fixed percentage stays steady: lower is "
     "faster. With NVIDIA DLSS on, DLSS makes the rest up to the output: 100% is DLAA, 75% about its Quality mode."},
    {TAB_GRAPHICS, "NVIDIA DLSS", &opts.dlss, 0, 1, 1, tx_dlss,
     "NVIDIA DLSS in place of the game's own upscaler, on GeForce RTX cards. Needs NVIDIA Streamline's files in a streamline folder "
     "in the game folder, or Fritchii's. Switching it on the first time applies after a restart."},
    {TAB_GRAPHICS, "DLSS preset", &opts.dlss_preset, 0, 4, 1, tx_dlss_preset,
     "The DLSS model. M is the newest and sharpest, K a little softer and lighter on the GPU. NVIDIA's choice picks one per "
     "resolution."},
    {TAB_GRAPHICS, "Output resolution", &opts.output_res, 0, kLastOutput, 1, tx_output_res,
     "The size the game draws its whole frame at, HUD included, before it is scaled to your window. The 3D is drawn at the "
     "Resolution percentage of it. Docked only. Applies after a restart."},
    {TAB_GRAPHICS, "Upscaling", &opts.upscale, 0, 2, 1, tx_upscale,
     "The filter that scales the picture to your window. Off is the plain copy. Catmull-Rom is sharper, Lanczos sharper still."},
    {TAB_GRAPHICS, "Sharpness", &opts.sharpness, 0, 100, 10, tx_percent,
     "Sharpens the picture as it goes to the window, with or without an upscaling filter. Too much shows halos around edges."},
    {TAB_GRAPHICS, "Handheld profile", &opts.handheld, 0, 1, 1, tx_handheld,
     "Runs the game as if the console were undocked: lower resolutions, lighter on the GPU. Applies after a restart."},
    {TAB_GRAPHICS, "Ultrawide", &opts.ultrawide, 0, 1, 1, tx_ultrawide,
     "Monitor's shape widens the camera to fill a screen wider than 16:9. The HUD is still stretched. Applies after a restart."},
    {TAB_GRAPHICS, "Robust buffers", &opts.robust, 0, 2, 1, tx_robust,
     "Bounds-checked buffer reads: a read past a buffer's end is dropped instead of stopping the GPU. Auto is on, on every card; "
     "Off can be faster on some NVIDIA cards. Applies after restart."},

    // Gameplay (3 rows)
    {TAB_GAMEPLAY, "FPS counter", &opts.show_fps, 0, 1, 1, tx_onoff,
     "Shows frames per second in the top right corner. With frame generation on, it also shows the game's own rate.", true},
    {TAB_GAMEPLAY, "Game speed", &opts.realtime, 0, 1, 1, tx_speed,
     "Per frame: the game slows down whenever it runs below 60 fps. Real time: it keeps normal speed down to 15 fps by taking "
     "longer steps."},
    {TAB_GAMEPLAY, "Text speed", &opts.text_speed, 0, 3, 1, tx_text_speed,
     "How fast the letters of dialogue and other messages appear. Normal is the game's own pace, with its pauses. Instant shows "
     "each page at once, as pressing A does."},
    {TAB_GAMEPLAY, "Microphone", &opts.mic, 0, 2 + kMaxMics, 1, tx_mic,
     "What the ocarina hears when the game's microphone option is on (playing an instrument, or humming). Default follows "
     "Windows' default recording device. Windows must allow desktop apps to use the microphone.", true},

    // Controls (9 rows)
    {TAB_CONTROLS, "Controller in background", &opts.pad_background, 0, 1, 1, tx_onoff,
     "On: the controller keeps playing the game while another window has the focus, for streaming or playing beside other "
     "apps. The keyboard and mouse still need the game's window."},
    {TAB_CONTROLS, "Mouse camera", &opts.mouse_native, 0, 1, 1, tx_mousecam,
     "Native: the captured mouse turns the camera itself, one to one, as in a PC game. As the stick: the mouse stands in for "
     "the right stick, with its curve and top speed.", true},
    {TAB_CONTROLS, "Mouse sensitivity", &opts.mouse_sens, 10, 500, 10, tx_sens,
     "How far the camera turns for a given mouse movement, for either Mouse camera. At 1.00 native look turns a tenth of a "
     "degree a count. The - and = keys change it in the game too.", true},
    {TAB_CONTROLS, "Invert mouse vertical", &opts.mouse_invert, 0, 1, 1, tx_onoff,
     "Swaps up and down for the mouse: moving it forward tilts the camera down.", true},
    {TAB_CONTROLS, "Invert camera X", &opts.invert_cam_x, 0, 1, 1, tx_onoff,
     "Inverts horizontal look (left/right) on the right analog stick.", true},
    {TAB_CONTROLS, "Invert camera Y", &opts.invert_cam_y, 0, 1, 1, tx_onoff,
     "Inverts vertical look (up/down) on the right analog stick.", true},
    {TAB_CONTROLS, "Rumble strength", &opts.rumble, 0, 150, 25, tx_rumble,
     "How strongly the controller rumbles, in percent of the game's rumble. Off turns it off.", true},
    {TAB_CONTROLS, "Layout preset", &opts.pad_layout, 0, 3, 1, tx_pad_layout,
     "Nintendo (default Switch order), Xbox / PC (A for action/talk), or Swap Face. Individual bindings switch this to Custom."},
    {TAB_CONTROLS, "Controls", nullptr, 0, 0, 0, nullptr,   // (hookshot: opens its page)
     "Which gamepad button and which keys press each of the Switch's buttons."},
    {TAB_GAMEPLAY, "Cheats", nullptr, 0, 0, 0, nullptr,
     "Open the private gameplay tools."},
};
constexpr int kCheatsRow = sizeof rows / sizeof rows[0] - 1;
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
int tab_selected[TAB_COUNT] = {0, 0, 0};

int tab_row_count(int t) {
    int c = 0;
    for (int i = 0; i < kRows; i++) if (rows[i].tab == (MenuTab)t && (i != kCheatsRow || cheats_enabled)) c++;
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
        if (rows[i].tab == (MenuTab)t && (i != kCheatsRow || cheats_enabled)) {
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
bool is_slider(const Row& r) { return r.value == &opts.sharpness || r.value == &opts.mouse_sens; }
/// (a scale: the arrows, a click and the wheel stop at its ends; Rumble strength would go from Off to its strongest)
bool stops_at_ends(const Row& r) { return r.value == &opts.mouse_sens || r.value == &opts.rumble; }
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
    if (r.value == &opts.output_res) {
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
        if (!r.preset_exempt) {   // (the controls, the window and the counter are not the preset's)
            opts.preset = 0; // Custom
            sync_preset_budgets(0);
        }
    }
    if (r.value == &opts.mic && can_mic()) mic_chosen();   // (hookshot)
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
int page = 0;          // 0 the options, 1 the controls, 2 the cheats, 3 the teleport list
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
    const char* heading = page == 4 ? "Inventory" : page == 3 ? "Teleport" : page == 2 ? "Cheats" : "Hookshot";
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
        const bool active = (t == cur_tab);
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
        const std::string label = r.value ? r.text(*r.value) : ">";   // (">": a page)
        const bool arrows = r.value && sel;
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
    wrap_help(dc, rows[tab_row_to_global(cur_tab, tab_selected[cur_tab])].help, (int)w - 32 * s, 16 * s, help_top + 4 * s, true);
    SelectObject(dc, body);
    SetTextColor(dc, RGB(140, 150, 144));
    const char* hint = "Arrows change   Q/E or LB/RB: tabs   Insert, F1 or Home closes";   // (fits 448 px at 16 px)
    TextOutA(dc, 16 * s, help_top + 4 * s + help_px + 10 * s, hint, (int)strlen(hint));
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
/// The tallest help text at scale s, in pixels: the menu keeps that height whichever row is selected
int measure_help(int s) {
    HDC dc = CreateCompatibleDC(nullptr);
    HFONT f = font(14 * s, FW_NORMAL);
    HGDIOBJ old = SelectObject(dc, f);
    int most = 0;
    for (const Row& r : rows) most = std::max(most, wrap_help(dc, r.help, (kMenuWidth - 32) * s, 0, 0, false));
    SelectObject(dc, old);
    DeleteObject(f);
    DeleteDC(dc);
    return most;
}
void redraw_menu(int s) {
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
    if (measured_at != s) help_px = measure_help(s), measured_at = s;
    // (title, the tab strip, the most rows a tab has, the rule and the help text, the hint line)
    gdi_draw(menu_bmp, (uint32_t)(kMenuWidth * s), (uint32_t)((44 + 32 + 8 + max_tab_rows() * 28 + 8 + 10 + 30) * s + help_px),
             paint_menu, s);
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
}

bool resolution_auto() { return opts.resolution == 0; }
int resolution_scale() {
    const int k = res_twelfths(opts.resolution);
    return k > 0 ? (int)(k * 10000.0f / 12.0f + 0.5f) : 0;   // (the twelfth game_size labels)
}

void menu_set_gpu_count(int count) {
    for (Row& r : rows) {
        if (r.value == &opts.gpu_device) {
            r.hi = count;
            if (opts.gpu_device > count) opts.gpu_device = 0;
        }
    }
    gpu_device_at_start = opts.gpu_device;
}

bool menu_is_open() { return open_now.load(std::memory_order_relaxed); }
int menu_heart_protection() { return protect_hearts.load(std::memory_order_relaxed); }
int menu_player_flight() { return cheats_enabled ? player_flight.load(std::memory_order_relaxed) : -1; }
int menu_free_camera() { return free_camera.load(std::memory_order_relaxed); }
int menu_take_cheat_action() { return cheat_action.exchange(0, std::memory_order_relaxed); }

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
    if (press[0] || press[6]) {
        open_now = !open_now.load(), changed = true;
        page = 0, listening = 0;
        // (the layout row shows the controls as they are: host/nn_hid.c keeps them, the ini's pad_ lines, and a saved
        // layout is not applied over them at start)
        if (open_now.load()) opts.pad_layout = detect_pad_layout();
        if (open_now.load()) refresh_mics();   // (the microphones as they are now)
    }
    if (open_now.load() && page == 1) {   // (hookshot) the Controls page
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
        if (press[5]) open_now = false;
        // (the tabs: LB and RB, Q and E, or Tab)
        if (tab_left) cur_tab = (cur_tab + TAB_COUNT - 1) % TAB_COUNT, changed = true;
        else if (tab_right) cur_tab = (cur_tab + 1) % TAB_COUNT, changed = true;
        const int tab_count = tab_row_count(cur_tab);
        int& selected = tab_selected[cur_tab];
        if (press[1]) selected = (selected + tab_count - 1) % tab_count, changed = true;
        if (press[2]) selected = (selected + 1) % tab_count, changed = true;
        Row& r = rows[tab_row_to_global(cur_tab, selected)];
        if (press[7] && &r == &rows[kCheatsRow]) page = 2, changed = true;
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
