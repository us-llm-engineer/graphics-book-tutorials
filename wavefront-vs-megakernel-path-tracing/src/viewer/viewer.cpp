// wfviewer: interactive progressive viewer. The GPU arms render into float accumulators; each frame the
// running mean is tone mapped on the host, a text overlay is drawn into the pixels, and the image is
// copied into the Vulkan swapchain (host-staged buffer copy, no shaders).
//   wfviewer [--scene NAME] [--obj FILE] [--width N] [--height N] [--mode megakernel|wavefront]
//            [--shots DIR] [--selftest N OUT.png] [--vkprobe]
// Keys: drag = orbit, wheel = zoom, W A D X = pan, P = pause, R = reset camera, L = reload scene,
//       M = switch arm, 1/2/3 = cornell/divergence/trianglehv, [ ] = exposure, S = screenshot, Esc = quit.
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "wf/film.hpp"
#include "wf/image_io.hpp"
#include "wf/renderer.hpp"
#include "wf/scene.hpp"
#include "wf/visual.hpp"

namespace {
using namespace wf;

std::string fmt(const char* f, double a = 0, double b = 0, double c = 0) {
    char buf[160]; std::snprintf(buf, sizeof buf, f, a, b, c); return buf;
}

// ---- viewer state ----
struct Options {
    std::string scene = "divergence", obj, shots = ".", selftest_out;
    int width = 960, height = 540, selftest_passes = 0;
    GpuMode mode = GpuMode::Megakernel;
    bool vkprobe = false;
};

struct App {
    Options opt;
    bool height_tint = true;
    Scene scene;
    Camera home{};
    std::unique_ptr<GpuRenderer> renderer;
    int spp_done = 0, shot_count = 0;
    bool paused = false;
    float exposure = 1.0f;
    double frame_ms = 16.0, pass_ms = 0.0;
    std::vector<float> mean;
    std::vector<uint8_t> shot_rgb8;   // last tone-mapped frame without overlay

    // orbit camera around look_at
    float yaw = 0, pitch = 0, dist = 1;
    bool dragging = false; double last_x = 0, last_y = 0;

    static constexpr int kMaxSpp = 4096;

    void load_scene() {
        scene = opt.obj.empty() ? make_scene(opt.scene, opt.width, opt.height) : load_obj(opt.obj, opt.width, opt.height);
        tint_by_height();
        home = scene.camera;
        reset_camera();
    }
    // trianglehv only: recolour the hills by height and brighten the sky (the benchmark scene itself is unchanged).
    void tint_by_height() { if (height_tint && opt.obj.empty()) vis::tint_by_height(scene); }
    void toggle_height() {
        Camera keep = scene.camera; float y = yaw, pt = pitch, d = dist;
        height_tint = !height_tint;
        scene = make_scene(opt.scene, opt.width, opt.height);
        tint_by_height();
        scene.camera = keep; yaw = y; pitch = pt; dist = d;
        rebuild();
    }
    void reset_camera() {
        scene.camera = home;
        float dx = home.position.x - home.look_at.x, dy = home.position.y - home.look_at.y, dz = home.position.z - home.look_at.z;
        dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        yaw = std::atan2(dx, dz);
        pitch = std::asin(std::max(-1.0f, std::min(1.0f, dy / std::max(dist, 1e-6f))));
        rebuild();
    }
    void apply_orbit() {
        float cp = std::cos(pitch);
        scene.camera.position = {scene.camera.look_at.x + dist * cp * std::sin(yaw), scene.camera.look_at.y + dist * std::sin(pitch),
                                 scene.camera.look_at.z + dist * cp * std::cos(yaw)};
        rebuild();
    }
    void pan(float right, float forward) {   // in units of the current distance
        float fx = -std::sin(yaw), fz = -std::cos(yaw), rx = std::cos(yaw), rz = -std::sin(yaw);
        float mx = (right * rx + forward * fx) * dist * 0.05f, mz = (right * rz + forward * fz) * dist * 0.05f;
        scene.camera.look_at.x += mx; scene.camera.look_at.z += mz;
        apply_orbit();
    }
    // The camera rig is uploaded with the scene, so a camera change re-creates the renderer
    // (the scene and its BVH are already built) and restarts accumulation.
    void rebuild() {
        renderer.reset();
        renderer = std::make_unique<GpuRenderer>(scene, opt.mode);
        spp_done = 0;
    }
    void toggle_mode() {
        opt.mode = opt.mode == GpuMode::Megakernel ? GpuMode::Wavefront : GpuMode::Megakernel;
        rebuild();
    }
    void switch_scene(const std::string& name) { opt.scene = name; opt.obj.clear(); load_scene(); }

    void step() {
        if (paused || spp_done >= kMaxSpp) return;
        RenderParams p; RenderStats st;
        renderer->run_passes(spp_done, 1, p, &st);
        ++spp_done;
        pass_ms = st.render_ms;
    }
    // Tone maps the current mean into rgb8 (3 bytes per pixel) and draws the overlay.
    void compose(std::vector<uint8_t>& rgb8) {
        if (spp_done > 0) renderer->download_mean(spp_done, &mean);
        else mean.assign((size_t)scene.width * scene.height * 3, 0.0f);
        rgb8 = tonemap_to_srgb8(mean, exposure);
        shot_rgb8 = rgb8;
        const int w = scene.width, h = scene.height, s = std::max(2, w / 480);
        const char* arm = opt.mode == GpuMode::Megakernel ? "MEGAKERNEL" : "WAVEFRONT";
        std::string scene_name = opt.obj.empty() ? opt.scene : "OBJ";
        vis::draw_text(rgb8, w, h, 8, 8, "SCENE " + scene_name + "   ARM " + arm, s, 255, 255, 255);
        vis::draw_text(rgb8, w, h, 8, 8 + 8 * s, fmt("SAMPLES %.0f", spp_done) + (spp_done >= kMaxSpp ? " (MAX)" : "") +
                  fmt("   %.1f FPS   %.1f MS/PASS", 1000.0 / std::max(frame_ms, 0.01), pass_ms), s, 255, 235, 120);
        vis::draw_text(rgb8, w, h, 8, 8 + 16 * s, std::string(paused ? "PAUSED" : "RUNNING") + fmt("   EXPOSURE %.2f", exposure), s,
                  paused ? 255 : 150, paused ? 150 : 255, 150);
        if (height_tint && opt.obj.empty() && scene.name == "trianglehv") {   // colour bar: high at the top
            const int bx = w - 16 * s, by = 8 + 8 * s, bh = h / 3, bw = 5 * s;
            for (int j = 0; j < bh; ++j) {
                float c[3]; vis::height_color(1.0f - (float)j / (bh - 1), c);
                for (int i = 0; i < bw; ++i) {
                    uint8_t* p = &rgb8[3 * ((size_t)(by + j) * w + bx + i)];
                    for (int k = 0; k < 3; ++k) p[k] = (uint8_t)(255.0f * std::pow(c[k], 1.0f / 2.2f));
                }
            }
            vis::draw_text(rgb8, w, h, bx - 6 * s, by - 8 * s, "HIGH", s, 255, 255, 255);
            vis::draw_text(rgb8, w, h, bx - 5 * s, by + bh + 3 * s, "LOW", s, 255, 255, 255);
        }
        vis::draw_text(rgb8, w, h, 8, h - 8 - 14 * s, "DRAG ORBIT   WHEEL ZOOM   W A D X PAN   P PAUSE   R RESET   L RELOAD", s, 200, 200, 200);
        vis::draw_text(rgb8, w, h, 8, h - 8 - 6 * s, "M ARM   H HEIGHT COLOR   1 2 3 SCENE   [ ] EXPOSURE   S SCREENSHOT   ESC QUIT", s, 200, 200, 200);
    }
    std::string screenshot() {
        std::string path = opt.shots + "/screenshot-" + opt.scene + "-" + (opt.mode == GpuMode::Megakernel ? "megakernel" : "wavefront") +
                           "-" + std::to_string(spp_done) + "spp-" + std::to_string(++shot_count) + ".png";
        write_png(path, scene.width, scene.height, shot_rgb8);
        return path;
    }
};

// ---- Vulkan presentation (copies a host buffer into the swapchain image) ----
#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) throw std::runtime_error(std::string(#x) + " failed: " + std::to_string((int)r_)); } while (0)

struct Present {
    VkInstance inst = VK_NULL_HANDLE; VkSurfaceKHR surf = VK_NULL_HANDLE; VkPhysicalDevice phys = VK_NULL_HANDLE; VkDevice dev = VK_NULL_HANDLE;
    uint32_t family = 0; VkQueue queue = VK_NULL_HANDLE; VkSwapchainKHR sc = VK_NULL_HANDLE; VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{}; std::vector<VkImage> images; std::vector<VkSemaphore> render_done;
    VkSemaphore image_avail = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE; VkCommandPool pool = VK_NULL_HANDLE; VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkBuffer staging = VK_NULL_HANDLE; VkDeviceMemory staging_mem = VK_NULL_HANDLE; void* mapped = nullptr; VkDeviceSize staging_bytes = 0;
    int w = 0, h = 0; bool bgra = true;

    void create_swapchain(GLFWwindow* win) {
        VkSurfaceCapabilitiesKHR caps; VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surf, &caps));
        if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) throw std::runtime_error("surface does not support transfer destination");
        uint32_t n = 0; vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surf, &n, nullptr);
        std::vector<VkSurfaceFormatKHR> fm(n); vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surf, &n, fm.data());
        VkSurfaceFormatKHR chosen{}; bool found = false;
        for (auto f : fm) if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM ||
                              f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) { chosen = f; found = true; break; }
        if (!found) throw std::runtime_error("no 8-bit RGBA/BGRA surface format");
        format = chosen.format; bgra = (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB);
        if (caps.currentExtent.width != 0xFFFFFFFFu) extent = caps.currentExtent;
        else { int fw, fh; glfwGetFramebufferSize(win, &fw, &fh); extent = {(uint32_t)fw, (uint32_t)fh}; }
        uint32_t count = std::max(caps.minImageCount, 2u); if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
        VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        ci.surface = surf; ci.minImageCount = count; ci.imageFormat = format; ci.imageColorSpace = chosen.colorSpace; ci.imageExtent = extent;
        ci.imageArrayLayers = 1; ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT; ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.preTransform = caps.currentTransform; ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        ci.clipped = VK_TRUE; ci.oldSwapchain = sc;
        VkSwapchainKHR fresh; VK_CHECK(vkCreateSwapchainKHR(dev, &ci, nullptr, &fresh));
        if (sc) vkDestroySwapchainKHR(dev, sc, nullptr);
        sc = fresh;
        uint32_t m = 0; vkGetSwapchainImagesKHR(dev, sc, &m, nullptr); images.resize(m); vkGetSwapchainImagesKHR(dev, sc, &m, images.data());
        for (VkSemaphore s : render_done) vkDestroySemaphore(dev, s, nullptr);
        render_done.assign(m, VK_NULL_HANDLE);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (auto& s : render_done) VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &s));
    }

    void init(GLFWwindow* win, int width, int height) {
        w = width; h = height;
        uint32_t ne = 0; const char** ext = glfwGetRequiredInstanceExtensions(&ne);
        if (!ext) throw std::runtime_error("GLFW reports no Vulkan support");
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.pApplicationName = "wfviewer"; ai.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai; ici.enabledExtensionCount = ne; ici.ppEnabledExtensionNames = ext;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
        VK_CHECK(glfwCreateWindowSurface(inst, win, nullptr, &surf));
        uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
        if (!nd) throw std::runtime_error("no Vulkan device");
        std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst, &nd, devs.data());
        bool ok = false;
        for (auto d : devs) {
            uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
            std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, q.data());
            for (uint32_t i = 0; i < nq && !ok; ++i) {
                VkBool32 present = VK_FALSE; vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surf, &present);
                if (present && (q[i].queueFlags & VK_QUEUE_TRANSFER_BIT || q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { phys = d; family = i; ok = true; }
            }
            if (ok) break;
        }
        if (!ok) throw std::runtime_error("no Vulkan queue family can present to the window");
        float prio = 1.0f; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex = family; qi.queueCount = 1; qi.pQueuePriorities = &prio;
        const char* dext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qi; dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = &dext;
        VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
        vkGetDeviceQueue(dev, family, 0, &queue);
        create_swapchain(win);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VK_CHECK(vkCreateSemaphore(dev, &si, nullptr, &image_avail));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; VK_CHECK(vkCreateFence(dev, &fi, nullptr, &fence));
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi.queueFamilyIndex = family;
        VK_CHECK(vkCreateCommandPool(dev, &pi, nullptr, &pool));
        VkCommandBufferAllocateInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cb.commandPool = pool; cb.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cb.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(dev, &cb, &cmd));
        staging_bytes = (VkDeviceSize)w * h * 4;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = staging_bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &staging));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, staging, &mr);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
        if (type == UINT32_MAX) throw std::runtime_error("no host-visible coherent memory");
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size; mai.memoryTypeIndex = type;
        VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &staging_mem)); VK_CHECK(vkBindBufferMemory(dev, staging, staging_mem, 0));
        VK_CHECK(vkMapMemory(dev, staging_mem, 0, VK_WHOLE_SIZE, 0, &mapped));
    }

    // Returns false when the swapchain was re-created (skip this frame).
    bool present(GLFWwindow* win, const std::vector<uint8_t>& rgb8) {
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
        uint32_t idx = 0; VkResult r = vkAcquireNextImageKHR(dev, sc, UINT64_MAX, image_avail, VK_NULL_HANDLE, &idx);
        if (r == VK_ERROR_OUT_OF_DATE_KHR) { vkDeviceWaitIdle(dev); create_swapchain(win); return false; }
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) throw std::runtime_error("vkAcquireNextImageKHR failed");
        vkResetFences(dev, 1, &fence);
        uint8_t* dst = static_cast<uint8_t*>(mapped);
        for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
            dst[4 * i + 0] = bgra ? rgb8[3 * i + 2] : rgb8[3 * i + 0]; dst[4 * i + 1] = rgb8[3 * i + 1];
            dst[4 * i + 2] = bgra ? rgb8[3 * i + 0] : rgb8[3 * i + 2]; dst[4 * i + 3] = 255;
        }
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; vkBeginCommandBuffer(cmd, &bi);
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}; b.image = images[idx]; b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy c{}; c.bufferRowLength = (uint32_t)w; c.bufferImageHeight = (uint32_t)h; c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageExtent = {std::min(extent.width, (uint32_t)w), std::min(extent.height, (uint32_t)h), 1};
        vkCmdCopyBufferToImage(cmd, staging, images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        vkEndCommandBuffer(cmd);
        VkPipelineStageFlags wait = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO}; sub.waitSemaphoreCount = 1; sub.pWaitSemaphores = &image_avail; sub.pWaitDstStageMask = &wait;
        sub.commandBufferCount = 1; sub.pCommandBuffers = &cmd; sub.signalSemaphoreCount = 1; sub.pSignalSemaphores = &render_done[idx];
        VK_CHECK(vkQueueSubmit(queue, 1, &sub, fence));
        VkPresentInfoKHR pr{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR}; pr.waitSemaphoreCount = 1; pr.pWaitSemaphores = &render_done[idx]; pr.swapchainCount = 1; pr.pSwapchains = &sc; pr.pImageIndices = &idx;
        r = vkQueuePresentKHR(queue, &pr);
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) { vkDeviceWaitIdle(dev); create_swapchain(win); }
        return true;
    }

    void destroy() {
        if (!dev) return;
        vkDeviceWaitIdle(dev);
        vkUnmapMemory(dev, staging_mem); vkDestroyBuffer(dev, staging, nullptr); vkFreeMemory(dev, staging_mem, nullptr);
        vkDestroyFence(dev, fence, nullptr); vkDestroySemaphore(dev, image_avail, nullptr);
        for (VkSemaphore s : render_done) vkDestroySemaphore(dev, s, nullptr);
        vkDestroyCommandPool(dev, pool, nullptr); vkDestroySwapchainKHR(dev, sc, nullptr); vkDestroyDevice(dev, nullptr);
        vkDestroySurfaceKHR(inst, surf, nullptr); vkDestroyInstance(inst, nullptr);
    }
};

App* g_app = nullptr;

void on_key(GLFWwindow* win, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    App& a = *g_app;
    try {
        switch (key) {
            case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(win, GLFW_TRUE); break;
            case GLFW_KEY_P: if (action == GLFW_PRESS) a.paused = !a.paused; break;
            case GLFW_KEY_R: if (action == GLFW_PRESS) a.reset_camera(); break;
            case GLFW_KEY_L: if (action == GLFW_PRESS) a.load_scene(); break;
            case GLFW_KEY_H: if (action == GLFW_PRESS && a.opt.obj.empty()) a.toggle_height(); break;
            case GLFW_KEY_M: if (action == GLFW_PRESS) a.toggle_mode(); break;
            case GLFW_KEY_1: a.switch_scene("cornell"); break;
            case GLFW_KEY_2: a.switch_scene("divergence"); break;
            case GLFW_KEY_3: a.switch_scene("trianglehv"); break;
            case GLFW_KEY_LEFT_BRACKET: a.exposure = std::max(0.05f, a.exposure / 1.25f); break;
            case GLFW_KEY_RIGHT_BRACKET: a.exposure = std::min(32.0f, a.exposure * 1.25f); break;
            case GLFW_KEY_S: if (action == GLFW_PRESS) std::cout << "screenshot: " << a.screenshot() << std::endl; break;
            case GLFW_KEY_W: a.pan(0, 1); break;
            default: break;
        }
        if (key == GLFW_KEY_A) a.pan(-1, 0);
        if (key == GLFW_KEY_D) a.pan(1, 0);
        if (key == GLFW_KEY_X) a.pan(0, -1);
    } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << std::endl; }
}
void on_mouse(GLFWwindow* win, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    g_app->dragging = action == GLFW_PRESS;
    glfwGetCursorPos(win, &g_app->last_x, &g_app->last_y);
}
void on_cursor(GLFWwindow*, double x, double y) {
    App& a = *g_app;
    if (!a.dragging) return;
    a.yaw -= (float)(x - a.last_x) * 0.006f;
    a.pitch = std::max(-1.5f, std::min(1.5f, a.pitch + (float)(y - a.last_y) * 0.006f));
    a.last_x = x; a.last_y = y;
    a.apply_orbit();
}
void on_scroll(GLFWwindow*, double, double dy) {
    g_app->dist = std::max(0.05f, g_app->dist * (dy > 0 ? 0.9f : 1.1f));
    g_app->apply_orbit();
}

int probe_vulkan() {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> d(n); vkEnumeratePhysicalDevices(inst, &n, d.data());
    for (auto p : d) { VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr); std::cout << "vulkan device: " << pr.deviceName << " (api " << VK_VERSION_MAJOR(pr.apiVersion) << "." << VK_VERSION_MINOR(pr.apiVersion) << ")\n"; }
    std::cout << "vulkan devices: " << n << "\n";
    vkDestroyInstance(inst, nullptr);
    return n ? 0 : 1;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) throw std::invalid_argument("missing value after " + a); return argv[++i]; };
        if (a == "--scene") o.scene = next();
        else if (a == "--obj") o.obj = next();
        else if (a == "--width") o.width = std::stoi(next());
        else if (a == "--height") o.height = std::stoi(next());
        else if (a == "--shots") o.shots = next();
        else if (a == "--mode") { std::string m = next(); if (m == "megakernel") o.mode = GpuMode::Megakernel; else if (m == "wavefront") o.mode = GpuMode::Wavefront; else throw std::invalid_argument("mode must be megakernel or wavefront"); }
        else if (a == "--selftest") { o.selftest_passes = std::stoi(next()); o.selftest_out = next(); }
        else if (a == "--vkprobe") o.vkprobe = true;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (o.width < 64 || o.height < 64) throw std::invalid_argument("width and height must be at least 64");
    return o;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        App app; app.opt = parse(argc, argv);
        if (app.opt.vkprobe) return probe_vulkan();
        g_app = &app;
        app.load_scene();
        if (app.opt.selftest_passes > 0) {   // no window: render, draw the overlay, write one PNG
            for (int i = 0; i < app.opt.selftest_passes; ++i) app.step();
            std::vector<uint8_t> rgb8; app.compose(rgb8);
            write_png(app.opt.selftest_out, app.scene.width, app.scene.height, rgb8);
            std::cout << "selftest: " << app.spp_done << " spp, wrote " << app.opt.selftest_out << "\n";
            return 0;
        }
        if (!glfwInit()) throw std::runtime_error("glfwInit failed (no display?)");
        if (!glfwVulkanSupported()) throw std::runtime_error("GLFW found no Vulkan loader");
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        GLFWwindow* win = glfwCreateWindow(app.scene.width, app.scene.height, "wfviewer", nullptr, nullptr);
        if (!win) throw std::runtime_error("cannot create window");
        glfwSetKeyCallback(win, on_key); glfwSetMouseButtonCallback(win, on_mouse); glfwSetCursorPosCallback(win, on_cursor); glfwSetScrollCallback(win, on_scroll);
        Present pres; pres.init(win, app.scene.width, app.scene.height);
        auto last = std::chrono::steady_clock::now();
        std::vector<uint8_t> rgb8;
        while (!glfwWindowShouldClose(win)) {
            glfwPollEvents();
            app.step();
            app.compose(rgb8);
            pres.present(win, rgb8);
            auto now = std::chrono::steady_clock::now();
            app.frame_ms = 0.9 * app.frame_ms + 0.1 * std::chrono::duration<double, std::milli>(now - last).count();
            last = now;
            std::string t = fmt("wfviewer  %.0f spp  ", app.spp_done) + fmt("%.1f fps", 1000.0 / std::max(app.frame_ms, 0.01));
            glfwSetWindowTitle(win, t.c_str());
        }
        pres.destroy(); glfwDestroyWindow(win); glfwTerminate();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
}
