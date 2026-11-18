# .plans/video_compression

Local video architecture and experiment plans.

Start with the [outermost AGENTS.md](../../AGENTS.md); it contains the project-wide directory lookup table.

## System design

Plans distinguish continuous shared-latent V1, residual V2, and later quantization/rate-distortion work. Dataset loading, video metrics, 3D baseline, and Perceiver V1 are already implemented.

## File map

| File(s) | Responsibility |
| --- | --- |
| `video_compression_plan.md` | Optional architecture brainstorm, experiment phases, and current V1 assumptions. |
| `go-through-plans-video-compression-plan-gentle-naur.md` | Optional implementation plan for metrics, 3D AE, and Perceiver V1. |

## Working here

Portable remaining work: [ROADMAP.md](../../docs/ROADMAP.md). V1 has no entropy-coded bitrate; reconstruction quality and latent-size proxies are separate from real compression claims.

## Keep this guide useful

- This is broad system-design and navigation guidance. The actual implementation is the source of truth; configs and tests provide supporting evidence.
- Correct inaccurate guidance and add missing/new information to the relevant AGENTS.md files as you discover it; update parent/root lookup entries when scope changes.
- Keep this file concise (aim below 100 lines). Move detailed designs or growing inventories into a focused sibling document or child guide and link it here; avoid duplicating parent guidance.
