// (hookshot) NVIDIA DLSS (through NVIDIA Streamline) in place of OoT's upscale networks: fd029a317a9a4b94, the scene at 3/4 to its
// output, and 9dc98871eb7fc101, at its full size. Ported from Eden's/Lichii's OoT DLSS as Fritchii has it (vulkan_common/streamline.cpp,
// renderer_vulkan/vk_streamline_dlss.cpp, vk_ocarina_camera_motion.cpp). Only Streamline's MIT-licensed headers are built in
// (third_party/streamline): NVIDIA's DLLs are loaded at run time from the player's PC, none ship with the game.
//
// Where the DLLs are looked for: RX_VK_STREAMLINE_DIR, a "streamline" folder in the game folder, then Fritchii's
// (%APPDATA%\eden\streamline): sl.interposer.dll, sl.common.dll, sl.dlss.dll, and nvngx_dlss.dll there or the NVIDIA app's DLSS.
//
// The game gives its networks no motion vectors: each pixel's comes from the scene depth reprojected through this and the
// last frame's camera, which the scene's draws keep in their uniform buffer c3 (dlss_note_draw). Anything unexpected leaves the
// game's own network to run (logged with its reason).
#pragma once

#include <cstdint>
#include <vector>

#include "vk_loader.h"

namespace mkwvk {

struct Texture;

/// Before the instance: loads Streamline when the NVIDIA DLSS row is on (opts.dlss at the start)
bool dlss_start();
/// What Streamline asks of the instance and the device (none when it is not running)
void dlss_instance_extensions(std::vector<const char*>& out);
void dlss_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& available);
/// After the device: hands it to Streamline and routes the swapchain calls through it (manual hooking). False: no DLSS
bool dlss_attach();
/// Streamline runs and DLSS is supported on this GPU
bool dlss_available();
/// DLSS stands in for the networks now: available, and the NVIDIA DLSS row on
bool dlss_active();
/// 0 Streamline not started, 1 started but no DLSS on this GPU, 2 DLSS available (the menu's NVIDIA DLSS row)
int dlss_state();
/// The networks DLSS stands in for (when active)
bool dlss_key(uint64_t program_key);
/// A scene draw: its vertex stage's c3 (0x230 bytes when bound) and its targets (vk_draw.cpp draw)
void dlss_note_draw(const void* c3, uint64_t c3_size, uint32_t color_w, uint32_t color_h, VkFormat color_format,
                    const Texture* depth, uint32_t depth_level);
/// In place of the network's dispatch: DLSS from its input into its output. False: the network runs
bool dlss_dispatch(VkCommandBuffer cmd, uint64_t key, VkImageView input_view, const Texture* input, VkImageView output_view,
                   const Texture* output, uint32_t output_level);

} // namespace mkwvk
