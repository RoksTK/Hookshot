// (hookshot) NVIDIA DLSS in place of OoT's upscale networks: see vk_dlss.h.
#ifdef _WIN32
// (NVIDIA's sl_security.h checks Streamline's own signatures with Windows 8's WINTRUST_SIGNATURE_SETTINGS: this file alone is
// built for Windows 10, before anything includes <windows.h>; the game needs Windows 10 anyway)
#undef _WIN32_WINNT
#undef NTDDI_VERSION
#define _WIN32_WINNT 0x0A00
#define NTDDI_VERSION 0x0A000000
#endif
#include "vk_dlss.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "vk_internal.h"
#include "vk_draw.h"
#include "vk_menu.h"
#include "vk_texture.h"

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#include <softpub.h>
#include <wintrust.h>
#include <sl.h>
#include <sl_dlss.h>
#include <sl_helpers_vk.h>
#include <sl_security.h>

#include "vk_paths.h"
#include "shaders/dlss_color_comp_spv.h"
#include "shaders/dlss_motion_comp_spv.h"
#include "shaders/dlss_output_comp_spv.h"
#endif

namespace mkwvk {

constexpr uint64_t kNetworkScaled = 0xfd029a317a9a4b94ull;   // (the scene at 3/4 of the output into it)
constexpr uint64_t kNetworkFull = 0x9dc98871eb7fc101ull;     // (the scene at the output's size)

#ifndef _WIN32
bool dlss_start() { return false; }
void dlss_instance_extensions(std::vector<const char*>&) {}
void dlss_device_extensions(std::vector<const char*>&, const std::vector<VkExtensionProperties>&) {}
bool dlss_attach() { return false; }
bool dlss_available() { return false; }
int dlss_state() { return 0; }
bool dlss_active() { return false; }
bool dlss_key(uint64_t) { return false; }
void dlss_note_draw(const void*, uint64_t, uint32_t, uint32_t, VkFormat, const Texture*, uint32_t) {}
bool dlss_dispatch(VkCommandBuffer, uint64_t, VkImageView, const Texture*, VkImageView, const Texture*, uint32_t) { return false; }
#else

namespace {

namespace fs = std::filesystem;

// ---- Streamline: found, checked, loaded (as Fritchii's vulkan_common/streamline.cpp) ----

// (a custom engine names itself with its own project id: Fritchii's, the same game's DLSS integration)
constexpr const char* kProjectId = "8d3e2f9a-6b41-4c7e-9a52-1f0c3b7d5e64";
constexpr const wchar_t* kInterposer = L"sl.interposer.dll";
constexpr const wchar_t* kModel = L"nvngx_dlss.dll";
constexpr uint32_t kMinDlssMajor = 310;   // (the first DLSS with presets K and M)

using Version = std::array<uint32_t, 4>;

struct Api {
    PFun_slInit* init{};
    PFun_slShutdown* shutdown{};
    PFun_slIsFeatureSupported* is_feature_supported{};
    PFun_slGetFeatureRequirements* get_feature_requirements{};
    PFun_slGetFeatureVersion* get_feature_version{};
    PFun_slGetFeatureFunction* get_feature_function{};
    PFun_slGetNewFrameToken* get_new_frame_token{};
    PFun_slSetConstants* set_constants{};
    PFun_slEvaluateFeature* evaluate_feature{};
    PFun_slFreeResources* free_resources{};
    PFun_slSetVulkanInfo* set_vulkan_info{};
};

struct SlState {
    HMODULE module{};
    Api api{};
    std::wstring plugin_dir, model_dir;
    std::vector<std::string> instance_exts, device_exts;
    bool started = false, attached = false, failed = false;
    PFun_slDLSSSetOptions* set_options{};
};
SlState sl_state;

std::string narrow(const fs::path& p) {
    const std::wstring w = p.wstring();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t)std::max(n, 0), '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

void sl_log(sl::LogType type, const char* message) {
    std::string text = message ? message : "";
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    static const bool verbose = getenv("RX_VK_DLSS_LOG") != nullptr;
    if (type == sl::LogType::eError) logf("dlss: Streamline error: %s", text.c_str());
    else if (type == sl::LogType::eWarn) logf("dlss: Streamline warning: %s", text.c_str());
    else if (verbose) logf("dlss: Streamline: %s", text.c_str());
}

fs::path game_folder() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    return fs::path(std::string(exe, n)).parent_path();
}

std::vector<fs::path> search_dirs() {
    std::vector<fs::path> dirs;
    if (const char* d = getenv("RX_VK_STREAMLINE_DIR"); d && *d) dirs.emplace_back(d);
    dirs.push_back(game_folder() / "streamline");
    if (const wchar_t* appdata = _wgetenv(L"APPDATA")) dirs.push_back(fs::path(appdata) / L"eden" / L"streamline");   // (Fritchii's)
    return dirs;
}

std::optional<Version> read_version(const fs::path& path) {
    DWORD handle = 0;
    const DWORD size = GetFileVersionInfoSizeW(path.c_str(), &handle);
    if (!size) return std::nullopt;
    std::vector<uint8_t> data(size);
    VS_FIXEDFILEINFO* info = nullptr;
    UINT length = 0;
    if (!GetFileVersionInfoW(path.c_str(), 0, size, data.data()) || !VerQueryValueW(data.data(), L"\\", (void**)&info, &length) || !info)
        return std::nullopt;
    return Version{HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS), HIWORD(info->dwFileVersionLS),
                   LOWORD(info->dwFileVersionLS)};
}

std::string version_text(const fs::path& path) {
    const auto v = path.empty() ? std::nullopt : read_version(path);
    if (!v) return "?";
    char b[64];
    snprintf(b, sizeof b, "%u.%u.%u.%u", (*v)[0], (*v)[1], (*v)[2], (*v)[3]);
    return b;
}

bool signed_by_nvidia(const fs::path& path) {
    std::error_code ec;
    const std::wstring file_path = fs::absolute(path, ec).wstring();
    WINTRUST_FILE_INFO file{};
    file.cbStruct = sizeof file;
    file.pcwszFilePath = file_path.c_str();
    WINTRUST_DATA data{};
    data.cbStruct = sizeof data;
    data.dwUIChoice = WTD_UI_NONE;
    data.fdwRevocationChecks = WTD_REVOKE_NONE;
    data.dwUnionChoice = WTD_CHOICE_FILE;
    data.pFile = &file;
    data.dwStateAction = WTD_STATEACTION_VERIFY;
    data.dwProvFlags = WTD_CACHE_ONLY_URL_RETRIEVAL;
    GUID action = WINTRUST_ACTION_GENERIC_VERIFY_V2;
    bool valid = WinVerifyTrust(nullptr, &action, &data) == ERROR_SUCCESS;
    if (valid) {
        CRYPT_PROVIDER_DATA* provider = WTHelperProvDataFromStateData(data.hWVTStateData);
        CRYPT_PROVIDER_SGNR* signer = provider ? WTHelperGetProvSignerFromChain(provider, 0, FALSE, 0) : nullptr;
        CRYPT_PROVIDER_CERT* cert = signer ? WTHelperGetProvCertFromChain(signer, 0) : nullptr;
        wchar_t name[128]{};
        valid = cert && CertGetNameStringW(cert->pCert, CERT_NAME_SIMPLE_DISPLAY_TYPE, 0, nullptr, name, 128) > 1 &&
                std::wstring(name) == L"NVIDIA Corporation";
    }
    data.dwStateAction = WTD_STATEACTION_CLOSE;
    WinVerifyTrust(nullptr, &action, &data);
    return valid;
}

/// The NVIDIA app keeps every DLSS it downloads under ProgramData: the newest
fs::path app_model() {
    const wchar_t* pd = _wgetenv(L"ProgramData");
    if (!pd) return {};
    const fs::path root = fs::path(pd) / L"NVIDIA" / L"NGX" / L"models" / L"dlss" / L"versions";
    fs::path best_path;
    Version best{};
    std::error_code ec;
    for (const auto& vd : fs::directory_iterator(root, ec)) {
        std::error_code fe;
        for (const auto& f : fs::directory_iterator(vd.path() / L"files", fe)) {
            if (f.path().extension() != L".bin") continue;
            const auto v = read_version(f.path());
            if (v && (*v)[0] >= kMinDlssMajor && *v > best) best = *v, best_path = f.path();
        }
    }
    return best_path;
}

/// The NVIDIA app's DLSS copied beside the loader as nvngx_dlss.dll (the name Streamline loads), when it is newer than the copy
fs::path use_app_model(const fs::path& folder, const fs::path& model) {
    const fs::path dir = folder / L"nvidia-app";
    const fs::path target = dir / kModel;
    std::error_code ec;
    const bool current = fs::is_regular_file(target, ec) && fs::file_size(target, ec) == fs::file_size(model, ec) &&
                         read_version(target) == read_version(model);
    if (!current) {
        if (!signed_by_nvidia(model)) {
            logf("dlss: the NVIDIA app's DLSS %s is not signed by NVIDIA", narrow(model).c_str());
            return {};
        }
        fs::create_directories(dir, ec);
        fs::copy_file(model, target, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            logf("dlss: could not copy the NVIDIA app's DLSS to %s: %s", narrow(target).c_str(), ec.message().c_str());
            return {};
        }
    }
    return signed_by_nvidia(target) ? dir : fs::path{};
}

template <typename T>
bool resolve(HMODULE m, const char* name, T*& f) {
    f = reinterpret_cast<T*>((void*)GetProcAddress(m, name));
    if (!f) logf("dlss: sl.interposer.dll has no %s", name);
    return f != nullptr;
}

std::vector<std::string> names(uint32_t n, const char** list) {
    std::vector<std::string> out;
    for (uint32_t i = 0; i < n; i++) out.emplace_back(list[i]);
    return out;
}

// ---- the passes around DLSS (push descriptors, as vk_draw.cpp's typed_upscale) ----

struct Pass {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

Pass make_pass(const uint32_t* spv, size_t bytes, uint32_t push_bytes) {
    Pass p;
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                         {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    li.bindingCount = 2, li.pBindings = b;
    VkDescriptorSetLayout set{};
    if (vkCreateDescriptorSetLayout(dev.device, &li, nullptr, &set) != VK_SUCCESS) return p;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &set;
    pi.pushConstantRangeCount = push_bytes ? 1 : 0, pi.pPushConstantRanges = push_bytes ? &pr : nullptr;
    if (vkCreatePipelineLayout(dev.device, &pi, nullptr, &p.layout) != VK_SUCCESS) return p;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = bytes, mi.pCode = spv;
    VkShaderModule m{};
    if (vkCreateShaderModule(dev.device, &mi, nullptr, &m) != VK_SUCCESS) return p;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = p.layout;
    if (vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipe) != VK_SUCCESS) p.pipe = VK_NULL_HANDLE;
    vkDestroyShaderModule(dev.device, m, nullptr);
    return p;
}

void run_pass(VkCommandBuffer cmd, const Pass& p, VkSampler sampler, VkImageView src, VkImageView dst, uint32_t w, uint32_t h,
              const void* push, uint32_t push_bytes) {
    const VkDescriptorImageInfo in{sampler, src, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorImageInfo out{VK_NULL_HANDLE, dst, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    wr[0].dstBinding = 0, wr[0].descriptorCount = 1, wr[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, wr[0].pImageInfo = &in;
    wr[1].dstBinding = 1, wr[1].descriptorCount = 1, wr[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, wr[1].pImageInfo = &out;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 2, wr);
    if (push_bytes) vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes, push);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}

/// Everything before it written and visible to everything after (the passes are few and small)
void barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1, di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &di);
}

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    uint32_t width = 0, height = 0;
};
struct Retired {
    VkImage image;
    VmaAllocation memory;
    VkImageView view;
    uint64_t frame;
};
std::vector<Retired> retired;   // (given back 8 frames later, when no work in flight can use them)

void retire(Image& im) {
    if (im.image || im.view) retired.push_back({im.image, im.memory, im.view, texture_frame_no()});
    im = Image{};
}
void retire_view(VkImageView v) {
    if (v) retired.push_back({VK_NULL_HANDLE, nullptr, v, texture_frame_no()});
}
void free_retired() {
    const uint64_t now = texture_frame_no();
    for (size_t i = 0; i < retired.size();) {
        if (now - retired[i].frame < 8) { i++; continue; }
        if (retired[i].view) vkDestroyImageView(dev.device, retired[i].view, nullptr);
        if (retired[i].image) destroy_image(retired[i].image, retired[i].memory);
        retired.erase(retired.begin() + (ptrdiff_t)i);
    }
}

/// An image of this size and format, made again (the old one retired) when either changes
bool ensure(VkCommandBuffer cmd, Image& im, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage) {
    if (im.image && im.width == w && im.height == h && im.format == f) return true;
    retire(im);
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D, ic.format = f, ic.extent = {w, h, 1}, ic.mipLevels = 1, ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT, ic.tiling = VK_IMAGE_TILING_OPTIMAL, ic.usage = usage;
    if (!create_image(ic, im.image, im.memory)) {
        im = Image{};
        return false;
    }
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = im.image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = f;
    vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(dev.device, &vc, nullptr, &im.view) != VK_SUCCESS) {
        destroy_image(im.image, im.memory);
        im = Image{};
        return false;
    }
    VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};   // (UNDEFINED -> GENERAL, once)
    ib.srcStageMask = VK_PIPELINE_STAGE_2_NONE, ib.srcAccessMask = VK_ACCESS_2_NONE;
    ib.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    ib.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    ib.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, ib.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ib.image = im.image, ib.subresourceRange = vc.subresourceRange;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.imageMemoryBarrierCount = 1, di.pImageMemoryBarriers = &ib;
    vkCmdPipelineBarrier2(cmd, &di);
    im.width = w, im.height = h, im.format = f, im.usage = usage;
    return true;
}

// ---- the scene's camera and depth, from its draws (dlss_note_draw) ----

using Matrix = std::array<float, 16>;
struct Camera {
    uint64_t frame = ~0ull;
    uint32_t width = 0, height = 0;   // (the scene's render size, which the camera's own extent says)
    Matrix current{}, previous{};
    VkImage depth = VK_NULL_HANDLE;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    uint32_t depth_w = 0, depth_h = 0, depth_level = 0;
};
std::mutex camera_mu;
Camera camera;

/// From the current clip space straight into the last frame's (Fritchii's ClipToPreviousClip): previous * inverse(current)
std::optional<Matrix> clip_to_previous(const Matrix& current, const Matrix& previous) {
    double rows[4][8]{};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) rows[i][j] = current[i * 4 + j];
        rows[i][i + 4] = 1.0;
    }
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        for (int r = col + 1; r < 4; r++)
            if (std::abs(rows[r][col]) > std::abs(rows[pivot][col])) pivot = r;
        if (pivot != col)
            for (int k = 0; k < 8; k++) std::swap(rows[col][k], rows[pivot][k]);
        const double d = rows[col][col];
        if (!std::isfinite(d) || std::abs(d) < 1e-10) return std::nullopt;
        for (int k = 0; k < 8; k++) rows[col][k] /= d;
        for (int r = 0; r < 4; r++) {
            if (r == col) continue;
            const double s = rows[r][col];
            for (int k = 0; k < 8; k++) rows[r][k] -= s * rows[col][k];
        }
    }
    Matrix out{};
    for (int r = 0; r < 4; r++)
        for (int c = 0; c < 4; c++) {
            double v = 0;
            for (int i = 0; i < 4; i++) v += previous[r * 4 + i] * rows[i][c + 4];
            out[r * 4 + c] = (float)v;
            if (!std::isfinite(out[r * 4 + c])) return std::nullopt;
        }
    return out;
}

// ---- DLSS's own state ----

struct Dlss {
    bool tried = false, ready = false;
    Pass color, motion, output;
    VkSampler point = VK_NULL_HANDLE, linear = VK_NULL_HANDLE;
    Image color_img, motion_img, out_img;
    uint32_t render_w = 0, render_h = 0, out_w = 0, out_h = 0;
    int preset = -1;
    uint32_t frame_index = 0;
    uint64_t last_frame = ~0ull;
    Matrix last_current{};
    const char* last_reason = nullptr;
    int reason_repeats = 0;
};
Dlss dlss;

bool dlss_setup() {
    if (dlss.tried) return dlss.ready;
    dlss.tried = true;
    if (!dev.push_descriptor || !vkCmdPushDescriptorSetKHR) {
        logf("dlss: no push descriptors: the game's networks run");
        return false;
    }
    void* f = nullptr;
    if (sl_state.api.get_feature_function(sl::kFeatureDLSS, "slDLSSSetOptions", f) != sl::Result::eOk || !f) {
        logf("dlss: Streamline's DLSS has no slDLSSSetOptions: the game's networks run");
        return false;
    }
    sl_state.set_options = reinterpret_cast<PFun_slDLSSSetOptions*>(f);
    dlss.color = make_pass(DLSS_COLOR_COMP_SPV, sizeof DLSS_COLOR_COMP_SPV, 0);
    dlss.motion = make_pass(DLSS_MOTION_COMP_SPV, sizeof DLSS_MOTION_COMP_SPV, 64);
    dlss.output = make_pass(DLSS_OUTPUT_COMP_SPV, sizeof DLSS_OUTPUT_COMP_SPV, 8);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.25f;
    vkCreateSampler(dev.device, &si, nullptr, &dlss.point);
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    vkCreateSampler(dev.device, &si, nullptr, &dlss.linear);
    dlss.ready = dlss.color.pipe && dlss.motion.pipe && dlss.output.pipe && dlss.point && dlss.linear;
    logf("dlss: passes %s", dlss.ready ? "made" : "NOT made: the game's networks run");
    return dlss.ready;
}

/// Why the game's network drew this frame: logged when the reason changes and every 300th repeat
bool decline(const char* reason) {
    if (reason != dlss.last_reason || ++dlss.reason_repeats % 300 == 0)
        logf("dlss: frame %llu: the game's network draws (%s)", (unsigned long long)texture_frame_no(), reason);
    if (reason != dlss.last_reason) dlss.last_reason = reason, dlss.reason_repeats = 0;
    return false;
}

sl::DLSSMode mode_for(uint32_t render_w, uint32_t out_w) {
    const float r = (float)render_w / (float)out_w;
    if (r >= 0.99f) return sl::DLSSMode::eDLAA;
    if (r >= 0.65f) return sl::DLSSMode::eMaxQuality;
    if (r >= 0.57f) return sl::DLSSMode::eBalanced;
    if (r >= 0.49f) return sl::DLSSMode::eMaxPerformance;
    return sl::DLSSMode::eUltraPerformance;
}

/// The DLSS preset row: 0 NVIDIA's choice, 1-4 J, K, L, M (RX_VK_DLSS_PRESET=J|K|L|M|default over it)
sl::DLSSPreset preset_of(int v) {
    if (const char* e = getenv("RX_VK_DLSS_PRESET")) {
        if (e[0] >= 'J' && e[0] <= 'M') return (sl::DLSSPreset)((uint32_t)sl::DLSSPreset::ePresetJ + (uint32_t)(e[0] - 'J'));
        return sl::DLSSPreset::eDefault;
    }
    if (v >= 1 && v <= 4) return (sl::DLSSPreset)((uint32_t)sl::DLSSPreset::ePresetJ + (uint32_t)(v - 1));
    return sl::DLSSPreset::eDefault;
}

sl::float4x4 identity() {
    sl::float4x4 m;
    m.setRow(0, {1, 0, 0, 0});
    m.setRow(1, {0, 1, 0, 0});
    m.setRow(2, {0, 0, 1, 0});
    m.setRow(3, {0, 0, 0, 1});
    return m;
}

sl::Resource resource(VkImage image, VkImageView view, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    sl::Resource r{sl::ResourceType::eTex2d, image, nullptr, view, (uint32_t)VK_IMAGE_LAYOUT_GENERAL};
    r.width = w, r.height = h, r.nativeFormat = (uint32_t)format, r.mipLevels = 1, r.arrayLayers = 1, r.flags = 0, r.usage = usage;
    return r;
}

} // namespace

bool dlss_start() {
    menu_init();   // (the options, before the instance: the NVIDIA DLSS row)
    if (!opts.dlss && !getenv("RX_VK_DLSS")) return false;
    if (getenv("RX_VK_DLSS") && !atoi(getenv("RX_VK_DLSS"))) return false;
    SlState& st = sl_state;
    if (st.started) return true;
    fs::path folder, interposer;
    std::error_code ec;
    for (const fs::path& d : search_dirs())
        if (fs::is_regular_file(d / kInterposer, ec)) {
            folder = d, interposer = d / kInterposer;
            break;
        }
    if (interposer.empty()) {
        logf("dlss: NVIDIA DLSS is on, but sl.interposer.dll (NVIDIA Streamline) is in none of: RX_VK_STREAMLINE_DIR, %s, "
             "%%APPDATA%%\\eden\\streamline. The game's networks run", narrow(game_folder() / "streamline").c_str());
        return false;
    }
    if (!sl::security::verifyEmbeddedSignature(interposer.c_str())) {
        logf("dlss: %s is not signed by NVIDIA: not loaded", narrow(interposer).c_str());
        return false;
    }
    st.module = LoadLibraryW(interposer.c_str());
    if (!st.module) {
        logf("dlss: could not load %s (error %lu)", narrow(interposer).c_str(), GetLastError());
        return false;
    }
    Api& a = st.api;
    const HMODULE m = st.module;
    if (!(resolve(m, "slInit", a.init) && resolve(m, "slShutdown", a.shutdown) &&
          resolve(m, "slIsFeatureSupported", a.is_feature_supported) &&
          resolve(m, "slGetFeatureRequirements", a.get_feature_requirements) &&
          resolve(m, "slGetFeatureVersion", a.get_feature_version) && resolve(m, "slGetFeatureFunction", a.get_feature_function) &&
          resolve(m, "slGetNewFrameToken", a.get_new_frame_token) && resolve(m, "slSetConstants", a.set_constants) &&
          resolve(m, "slEvaluateFeature", a.evaluate_feature) && resolve(m, "slFreeResources", a.free_resources) &&
          resolve(m, "slSetVulkanInfo", a.set_vulkan_info))) {
        FreeLibrary(st.module);
        st.module = nullptr;
        return false;
    }
    st.plugin_dir = folder.wstring();
    const fs::path model = folder / kModel;
    if (fs::is_regular_file(model, ec)) {
        if (!signed_by_nvidia(model)) {
            logf("dlss: %s is not signed by NVIDIA: not used", narrow(model).c_str());
            return false;
        }
        logf("dlss: DLSS %s from %s", version_text(model).c_str(), narrow(folder).c_str());
    } else if (const fs::path am = app_model(); !am.empty()) {
        if (const fs::path dir = use_app_model(folder, am); !dir.empty()) {
            st.model_dir = dir.wstring();
            logf("dlss: the NVIDIA app's DLSS %s", version_text(am).c_str());
        }
    } else {
        logf("dlss: no nvngx_dlss.dll in %s and no DLSS from the NVIDIA app", narrow(folder).c_str());
    }
    const wchar_t* paths[] = {st.plugin_dir.c_str(), st.model_dir.c_str()};
    const sl::Feature features[] = {sl::kFeatureDLSS};
    sl::Preferences pref{};
    pref.logLevel = getenv("RX_VK_DLSS_LOG") ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
    pref.logMessageCallback = sl_log;
    pref.pathsToPlugins = paths;
    pref.numPathsToPlugins = st.model_dir.empty() ? 1 : 2;
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseManualHooking |
                 sl::PreferenceFlags::eAllowOTA | sl::PreferenceFlags::eLoadDownloadedPlugins;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = 1;
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "1.0";
    pref.projectId = kProjectId;
    pref.renderAPI = sl::RenderAPI::eVulkan;
    if (const sl::Result r = a.init(pref, sl::kSDKVersion); r != sl::Result::eOk) {
        logf("dlss: NVIDIA Streamline did not start (result %d): the game's networks run", (int)r);
        return false;
    }
    sl::FeatureRequirements req{};
    if (const sl::Result r = a.get_feature_requirements(sl::kFeatureDLSS, req); r != sl::Result::eOk) {
        logf("dlss: Streamline found no DLSS plugin (result %d): sl.dlss.dll beside sl.interposer.dll? The game's networks run", (int)r);
        a.shutdown();
        return false;
    }
    if (req.vkNumComputeQueuesRequired || req.vkNumGraphicsQueuesRequired) {
        logf("dlss: this DLSS wants queues of its own: the game's networks run");
        a.shutdown();
        return false;
    }
    st.instance_exts = names(req.vkNumInstanceExtensions, req.vkInstanceExtensions);
    st.device_exts = names(req.vkNumDeviceExtensions, req.vkDeviceExtensions);
    if (std::find(st.device_exts.begin(), st.device_exts.end(), VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) != st.device_exts.end())
        st.device_exts.erase(std::remove(st.device_exts.begin(), st.device_exts.end(), std::string(VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)),
                             st.device_exts.end());
    sl::FeatureVersion v{};
    a.get_feature_version(sl::kFeatureDLSS, v);
    logf("dlss: NVIDIA Streamline %s started from %s (DLSS %s)", v.versionSL.toStr().c_str(), narrow(folder).c_str(),
         v.versionNGX.toStr().c_str());
    st.started = true;
    return true;
}

void dlss_instance_extensions(std::vector<const char*>& out) {
    if (!sl_state.started) return;
    for (const std::string& e : sl_state.instance_exts)
        if (std::none_of(out.begin(), out.end(), [&](const char* o) { return e == o; })) out.push_back(e.c_str());
}

void dlss_device_extensions(std::vector<const char*>& want, const std::vector<VkExtensionProperties>& available) {
    if (!sl_state.started) return;
    for (const std::string& e : sl_state.device_exts) {
        const bool has = std::any_of(available.begin(), available.end(), [&](const VkExtensionProperties& p) { return e == p.extensionName; });
        if (!has) {
            logf("dlss: the GPU lacks %s, which DLSS needs: the game's networks run", e.c_str());
            sl_state.failed = true;
            return;
        }
    }
    for (const std::string& e : sl_state.device_exts)
        if (std::none_of(want.begin(), want.end(), [&](const char* o) { return e == o; })) want.push_back(e.c_str());
}

bool dlss_attach() {
    SlState& st = sl_state;
    if (!st.started || st.attached || st.failed) return false;
    sl::VulkanInfo info{};
    info.device = dev.device;
    info.instance = dev.instance;
    info.physicalDevice = dev.physical;
    info.computeQueueFamily = info.graphicsQueueFamily = dev.queue_family;
    if (const sl::Result r = st.api.set_vulkan_info(info); r != sl::Result::eOk) {
        logf("dlss: Streamline did not take the device (result %d): the game's networks run", (int)r);
        st.api.shutdown(), st.started = false;
        return false;
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = dev.physical;
    if (const sl::Result r = st.api.is_feature_supported(sl::kFeatureDLSS, adapter); r != sl::Result::eOk) {
        logf("dlss: NVIDIA DLSS is not supported on this GPU or driver (result %d): the game's networks run", (int)r);
        st.api.shutdown(), st.started = false;
        return false;
    }
    // (manual hooking: Streamline does its per-frame work in these)
    const auto hook = [&](auto& f, const char* name) {
        if (FARPROC proxy = GetProcAddress(st.module, name)) f = reinterpret_cast<std::remove_reference_t<decltype(f)>>((void*)proxy);
        else logf("dlss: sl.interposer.dll has no %s", name);
    };
    hook(vkx::vkCreateSwapchainKHR, "vkCreateSwapchainKHR");
    hook(vkx::vkDestroySwapchainKHR, "vkDestroySwapchainKHR");
    hook(vkx::vkGetSwapchainImagesKHR, "vkGetSwapchainImagesKHR");
    hook(vkx::vkAcquireNextImageKHR, "vkAcquireNextImageKHR");
    hook(vkx::vkQueuePresentKHR, "vkQueuePresentKHR");
    hook(vkx::vkDeviceWaitIdle, "vkDeviceWaitIdle");
    st.attached = true;
    logf("dlss: NVIDIA DLSS is available");
    return true;
}

bool dlss_available() { return sl_state.started && sl_state.attached && !sl_state.failed; }

int dlss_state() {
    if (dlss_available()) return 2;
    return sl_state.started || sl_state.module ? 1 : 0;
}

bool dlss_active() { return dlss_available() && opts.dlss; }

bool dlss_key(uint64_t key) {
    if (!dlss_active()) return false;
    return key == kNetworkScaled || key == kNetworkFull;
}

void dlss_note_draw(const void* c3, uint64_t c3_size, uint32_t color_w, uint32_t color_h, VkFormat color_format,
                    const Texture* depth, uint32_t depth_level) {
    if (!dlss_active() || !c3 || c3_size < 0x230 || !depth) return;
    if (color_format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || color_w < 640 || color_h < 360) return;
    if (depth->format != VK_FORMAT_D32_SFLOAT && depth->format != VK_FORMAT_D32_SFLOAT_S8_UINT) return;
    const float* f = static_cast<const float*>(c3);
    // (the scene camera's block: its render extent at 0x120, a perspective projection's -1 at 0xa8 and 0 at 0xac)
    if (f[0x120 / 4] != (float)color_w || f[0x120 / 4 + 1] != (float)color_h || f[0xa8 / 4] != -1.0f || f[0xac / 4] != 0.0f) return;
    const uint64_t frame = texture_frame_no();
    std::lock_guard lk(camera_mu);
    // (reflections and half-size passes share the layout: the largest of the frame)
    if (camera.frame == frame && (uint64_t)color_w * color_h <= (uint64_t)camera.width * camera.height) return;
    camera.frame = frame;
    camera.width = color_w, camera.height = color_h;
    std::copy_n(f + 0x30 / 4, 16, camera.current.begin());
    std::copy_n(f + 0x180 / 4, 16, camera.previous.begin());
    camera.depth = depth->image;
    camera.depth_format = depth->format;
    camera.depth_level = depth_level;
    camera.depth_w = std::max(1u, depth->width >> depth_level), camera.depth_h = std::max(1u, depth->height >> depth_level);
}

bool dlss_dispatch(VkCommandBuffer cmd, uint64_t key, VkImageView input_view, const Texture* input, VkImageView output_view,
                   const Texture* output, uint32_t output_level) {
    if (!dlss_key(key) || sl_state.failed) return false;
    if (!dlss_setup()) return false;
    free_retired();
    if (!input || !output || !input_view || !output_view) return decline("no input or output image");
    if (output->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return decline("unexpected output format");
    Camera cam;
    {
        std::lock_guard lk(camera_mu);
        cam = camera;
    }
    const uint64_t frame = texture_frame_no();
    if (cam.frame != frame) return decline(cam.frame == ~0ull ? "no scene camera yet" : "no scene camera this frame");
    const uint32_t rw = cam.width, rh = cam.height;
    const uint32_t ow = std::max(1u, output->width >> output_level), oh = std::max(1u, output->height >> output_level);
    if (rw > input->width || rh > input->height) return decline("the camera's extent is past the input image");
    if (cam.depth_w < rw || cam.depth_h < rh) return decline("the scene depth is smaller than the camera's extent");
    if (ow < rw || oh < rh) return decline("the output is smaller than the input");
    const auto transform = clip_to_previous(cam.current, cam.previous);
    if (!transform) return decline("the camera's matrix has no inverse");
    if (dlss.last_frame == frame) return decline("a second network this frame");

    // (the depth's own view, depth aspect: made each frame, given back 8 frames later)
    VkImageView depth_view = VK_NULL_HANDLE;
    {
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = cam.depth, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = cam.depth_format;
        vc.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, cam.depth_level, 1, 0, 1};
        if (vkCreateImageView(dev.device, &vc, nullptr, &depth_view) != VK_SUCCESS) return decline("no view of the scene depth");
        retire_view(depth_view);
    }
    constexpr VkImageUsageFlags kWork = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    constexpr VkImageUsageFlags kOut = kWork | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!ensure(cmd, dlss.color_img, rw, rh, VK_FORMAT_R16G16B16A16_SFLOAT, kWork) ||
        !ensure(cmd, dlss.motion_img, rw, rh, VK_FORMAT_R16G16_SFLOAT, kWork) ||
        !ensure(cmd, dlss.out_img, ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, kOut))
        return decline("DLSS's images were not made");

    // (the game's colour and depth as the draws before left them)
    barrier(cmd);
    run_pass(cmd, dlss.color, dlss.point, input_view, dlss.color_img.view, rw, rh, nullptr, 0);
    run_pass(cmd, dlss.motion, dlss.point, depth_view, dlss.motion_img.view, rw, rh, transform->data(), 64);
    barrier(cmd);

    const bool sizes_changed = rw != dlss.render_w || rh != dlss.render_h || ow != dlss.out_w || oh != dlss.out_h;
    const int preset_row = opts.dlss_preset;
    const bool options_changed = sizes_changed || preset_row != dlss.preset;
    bool same_camera = true;
    for (int i = 0; i < 16 && same_camera; i++) same_camera = std::abs(cam.previous[i] - dlss.last_current[i]) < 0.001f;
    const bool reset = sizes_changed || dlss.last_frame == ~0ull || frame != dlss.last_frame + 1 || !same_camera;
    const sl::ViewportHandle viewport{0u};
    if (options_changed) {
        sl::DLSSOptions o{};
        o.mode = mode_for(rw, ow);
        o.outputWidth = ow, o.outputHeight = oh;
        o.colorBuffersHDR = sl::eFalse;   // (the network's input is tonemapped, perceptually encoded)
        o.useAutoExposure = sl::eFalse;
        const sl::DLSSPreset p = preset_of(preset_row);
        o.dlaaPreset = o.qualityPreset = o.balancedPreset = o.performancePreset = o.ultraPerformancePreset = o.ultraQualityPreset = p;
        if (const sl::Result r = sl_state.set_options(viewport, o); r != sl::Result::eOk) {
            sl_state.failed = true;
            logf("dlss: slDLSSSetOptions failed (result %d): the game's networks run from now on", (int)r);
            return false;
        }
        logf("dlss: %ux%u -> %ux%u (%.1f%%), mode %d, preset %d (%s)", rw, rh, ow, oh, 100.0 * rw / ow, (int)o.mode, (int)p,
             key == kNetworkFull ? "full-size network" : "3/4 network");
        dlss.render_w = rw, dlss.render_h = rh, dlss.out_w = ow, dlss.out_h = oh, dlss.preset = preset_row;
    }
    sl::FrameToken* token = nullptr;
    const uint32_t index = dlss.frame_index++;
    if (sl_state.api.get_new_frame_token(token, &index) != sl::Result::eOk || !token) return decline("no Streamline frame token");
    sl::Constants c{};
    c.cameraViewToClip = c.clipToCameraView = c.clipToLensClip = c.clipToPrevClip = c.prevClipToClip = identity();
    c.jitterOffset = {0.0f, 0.0f};
    c.mvecScale = {1.0f / (float)rw, 1.0f / (float)rh};   // (render pixels into Streamline's [-1, 1] screen units)
    c.cameraPinholeOffset = {0.0f, 0.0f};
    c.cameraPos = {0.0f, 0.0f, 0.0f};
    c.cameraUp = {0.0f, 1.0f, 0.0f};
    c.cameraRight = {1.0f, 0.0f, 0.0f};
    c.cameraFwd = {0.0f, 0.0f, 1.0f};
    c.cameraNear = 0.1f, c.cameraFar = 1000.0f, c.cameraFOV = 1.0f, c.cameraAspectRatio = (float)rw / (float)rh;
    c.depthInverted = sl::eFalse;
    c.cameraMotionIncluded = sl::eTrue;
    c.motionVectors3D = sl::eFalse;
    c.reset = reset ? sl::eTrue : sl::eFalse;
    c.motionVectorsJittered = sl::eFalse;
    if (sl_state.api.set_constants(c, *token, viewport) != sl::Result::eOk) return decline("slSetConstants failed");

    sl::Resource color = resource(dlss.color_img.image, dlss.color_img.view, dlss.color_img.format, rw, rh, kWork);
    sl::Resource depth = resource(cam.depth, depth_view, cam.depth_format, cam.depth_w, cam.depth_h, VK_IMAGE_USAGE_SAMPLED_BIT);
    sl::Resource motion = resource(dlss.motion_img.image, dlss.motion_img.view, dlss.motion_img.format, rw, rh, kWork);
    sl::Resource out = resource(dlss.out_img.image, dlss.out_img.view, dlss.out_img.format, ow, oh, kOut);
    const sl::Extent render_area{0, 0, rw, rh}, output_area{0, 0, ow, oh};
    sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
    sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
    sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render_area};
    sl::ResourceTag out_tag{&out, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &output_area};
    const sl::BaseStructure* inputs[] = {&viewport, &color_tag, &depth_tag, &motion_tag, &out_tag};
    if (const sl::Result r = sl_state.api.evaluate_feature(sl::kFeatureDLSS, *token, inputs, 5, reinterpret_cast<sl::CommandBuffer*>(cmd));
        r != sl::Result::eOk) {
        sl_state.failed = true;
        logf("dlss: slEvaluateFeature failed (result %d): the game's networks run from now on", (int)r);
        // (nothing written to the network's output yet: the network runs this frame too)
        return false;
    }
    barrier(cmd);
    const int32_t size[2] = {(int32_t)ow, (int32_t)oh};
    run_pass(cmd, dlss.output, dlss.out_img.width == ow && dlss.out_img.height == oh ? dlss.point : dlss.linear, dlss.out_img.view,
             output_view, ow, oh, size, 8);
    barrier(cmd);
    if (dlss.last_reason) logf("dlss: frame %llu: DLSS draws again", (unsigned long long)frame), dlss.last_reason = nullptr;
    dlss.last_frame = frame;
    dlss.last_current = cam.current;
    return true;
}

#endif

} // namespace mkwvk
