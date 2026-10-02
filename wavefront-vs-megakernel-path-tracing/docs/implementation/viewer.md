# Vulkan viewer

Source: [`src/viewer/viewer.cpp`](../../src/viewer/viewer.cpp); shared helpers in [`include/wf/visual.hpp`](../../include/wf/visual.hpp).

## How a frame is produced

1. The selected GPU arm renders one more sample per pixel into its float accumulators (`run_passes`).
2. The running mean is copied to the host (`download_mean`) and tone mapped.
3. A text overlay (scene, arm, samples, frames per second, milliseconds per pass, controls) is drawn into the pixels with a built-in 3x5 bitmap font.
4. The pixels are written into a host-visible staging buffer (converted to the swapchain's BGRA or RGBA order) and copied into the swapchain image with `vkCmdCopyBufferToImage`.

No graphics pipeline or shader is used, so the viewer needs only the Vulkan loader, GLFW and a presentable device. The swapchain uses FIFO presentation and is recreated when it becomes out of date.

## Controls

| Input | Action |
|---|---|
| left-drag / wheel | orbit / zoom around the look-at point |
| W A D X | pan forward, left, right, back |
| P | pause or resume |
| R | reset the camera |
| L | reload the scene (re-reads an OBJ file) |
| M | switch between megakernel and wavefront |
| 1 / 2 / 3 | cornell / divergence / trianglehv |
| H | hills scene: toggle height colouring |
| [ and ] | exposure down / up |
| S | save `screenshot-<scene>-<arm>-<spp>spp-<n>.png` (`--shots DIR`) |
| Esc | quit |

## Camera and restarts

The camera is an orbit (yaw, pitch, distance) around the look-at point. The camera rig is uploaded to the device together with the scene, so any camera change re-creates the renderer (the scene and its BVH are already built) and restarts accumulation. Accumulation stops at 4096 samples.

## Height colouring

For the hills scene the viewer recolours the triangles by height band (96 bands from dark indigo in the valleys through green and yellow to orange), brightens the sky and makes the two mirror spheres matte so the colours read clearly. The benchmark scene is not changed.

## Headless checks

`wfviewer --vkprobe` lists Vulkan devices. `wfviewer --selftest N out.png [--scene S --mode M]` renders `N` samples without a window and writes one frame with the overlay.
