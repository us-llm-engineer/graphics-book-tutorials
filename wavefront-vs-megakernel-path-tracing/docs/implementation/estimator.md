# The shared estimator

All three renderers call the same inline functions in [`include/wf/path_core.hpp`](../../include/wf/path_core.hpp). The functions are compiled for host and device (`WF_HD`), so there is exactly one implementation of the shading logic.

## Per-path state

`PathState` holds the running `throughput` and radiance `L`, the bounce `depth`, a `specular` flag (the last scattering event was a delta), `prev_pdf` (solid-angle density of the last BSDF sample, 0 for delta events) and the RNG key. The camera ray is built by `camera_ray` from the pixel, the sample index and a jitter drawn from the RNG; `init_path` packages it.

## One bounce

1. **Intersect** (`intersect`): nearest hit along the ray, or a miss. A miss adds `throughput * env` and ends the path.
2. **Emission** (`classify`): if the hit material emits, add `throughput * Le * w`, where `w = 1` for the first vertex and after a specular event, otherwise the power-heuristic weight of `prev_pdf` against the density with which the light sampler would have produced this point (`light_pdf_at_hit`). A purely emissive material, or reaching the depth limit, ends the path.
3. **Next-event estimation** (`shade_lambert`): pick one light uniformly, sample a point uniformly by area (`sample_light`), build a shadow ray. The contribution is added only if the shadow ray is unoccluded; the weight is the power heuristic of the light density against the cosine density.
4. **BSDF sampling**: Lambertian samples the cosine-weighted hemisphere (throughput times albedo); metal mirrors; dielectric chooses reflection or refraction with the exact Fresnel probability (`fresnel_dielectric`), including total internal reflection.
5. **Russian roulette** (`russian_roulette`): from depth 3, survive with `q = clamp(max component of throughput, 0.05, 0.95)` and divide throughput by `q`.
6. **Next ray** (`set_next_ray`): the origin is offset by `kRayEps = 1e-4` along the facing normal and `tmin = 1e-4`.

`trace_path` is the megakernel's loop over these steps; the wavefront runs the same steps as separate kernels connected by queues.

## Numerical choices

- Sphere intersection uses the projected-distance discriminant: with `w = oc - d (oc.d / d.d)`, `disc = r^2 - |w|^2` and `t = -(oc.d)/(d.d) +- sqrt(disc/(d.d))`. This avoids the cancellation in `|oc|^2 - r^2` for small or distant spheres.
- `--fmad=false` (CUDA) and `-ffp-contract=off` (host) forbid fused multiply-add so that the CPU, the megakernel and the wavefront perform the same float operations in the same order per path. This is why the two GPU arms agree bit for bit.
- A pass renders exactly one sample per pixel; pixel `i` accumulates sample `s` only after pass `s-1` finished, so there is no atomic floating-point addition anywhere and results do not depend on thread scheduling.

## Film and tone mapping

Per pixel the film accumulates `sum(L)` and `sum(lum(L)^2)`; the image is `sum(L)/spp`, the variance estimate is the unbiased variance of per-sample luminance. For PNG output the mean is scaled by an exposure, passed through Reinhard `x/(1+x)` and the sRGB transfer function; `.pfm` stores the linear mean.

## What the tests check

A closed-furnace oracle (`L_D = Le * sum_{k<=D} rho^k`), agreement with an independent naive BSDF-only path tracer, MIS correctness for small bright lights, unchanged mean under Russian roulette between depths 64 and 200, and bitwise equality across thread counts (CPU) and across the two GPU arms.
