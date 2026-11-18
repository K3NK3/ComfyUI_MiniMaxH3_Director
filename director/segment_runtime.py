"""Per-segment helpers shared by the Director executor."""

from __future__ import annotations

import logging
import torch

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.segment_runtime")

from ..lib.image_prep import fit_canvas, fit_video_long_edge
from ..lib.video_io import load_timeline_segment
from .frame_align import pad_or_trim_frames
from .plan import DirectorPlan


def needs_source_video(task_key: str) -> bool:
    return task_key in {"i2v", "fl2v", "v2v", "rv2v"}


def is_gen_timeline_plan(plan: DirectorPlan) -> bool:
    mode = str((plan.raw or {}).get("timelineMode") or "").lower()
    return mode in ("gen_blank", "gen_image", "prompt_batch", "image_batch", "fl2v")


def resolve_segment_raw_clip(plan: DirectorPlan, seg) -> torch.Tensor:
    """Prefer in-memory gen canvas / segment clip; fall back to timeline video decode."""
    log.info("resolve_segment_raw_clip called for segment %d", seg.index + 1)
    if seg.source_clip is not None and seg.source_clip.shape[0] > 0:
        log.info("  -> using in-memory source_clip (%s frames)", seg.source_clip.shape[0])
        return seg.source_clip.clone()
    log.info("  -> no in-memory source_clip")

    # t2v/r2v (incl. external groups) have no source frames. Do not slice the
    # placeholder gen source_video (len=segment_count 16×16 gray frames).
    if getattr(seg, "task_key", "") in {"t2v", "r2v"}:
        log.info("  -> t2v/r2v task, returning empty tensor")
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    # fl2v end-only: plan leaves source_clip=None on purpose. Do not slice the
    # placeholder gen source_video (len=segment_count gray frames) — that was
    # incorrectly becoming first_frame in the single-node path.
    if getattr(seg, "task_key", "") == "fl2v" and is_gen_timeline_plan(plan):
        log.info("  -> fl2v task with gen timeline, returning empty tensor")
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    sv = plan.source_video
    if is_gen_timeline_plan(plan) and sv is not None and int(sv.shape[0]) > 0:
        start = max(0, int(seg.start_frame))
        end = min(int(seg.end_frame), int(sv.shape[0]))
        if end > start:
            log.info("  -> using plan.source_video slice [%s:%s]", start, end)
            return sv[start:end].clone()

    if (plan.raw or {}).get("externalGroups", {}).get("active"):
        log.info("  -> external groups active, returning empty tensor")
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    log.info("  -> load_timeline_segment...")
    result = load_timeline_segment(plan.raw, seg.start_frame, seg.end_frame)
    log.info("  -> load_timeline_segment returned: %s frames", result.shape[0])
    return result


def resolve_segment_raw_clip_with_lookahead(
    plan: DirectorPlan,
    seg,
    *,
    end_extra: int = 0,
) -> torch.Tensor:
    """Like ``resolve_segment_raw_clip``, but may pull frames past ``seg.end_frame``.

    Extra frames are conditioning-only (continuity gen length matching); they are
    not kept in the exported segment after trim.
    """
    extra = max(0, int(end_extra))
    if extra <= 0:
        return resolve_segment_raw_clip(plan, seg)

    if seg.source_clip is not None and seg.source_clip.shape[0] > 0:
        # Gen canvases have no timeline lookahead beyond the clip itself.
        return seg.source_clip.clone()

    if getattr(seg, "task_key", "") in {"t2v", "r2v"}:
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    if getattr(seg, "task_key", "") == "fl2v" and is_gen_timeline_plan(plan):
        return torch.zeros((0, 16, 16, 3), dtype=torch.float32)

    end = int(seg.end_frame) + extra
    sv = plan.source_video
    if is_gen_timeline_plan(plan) and sv is not None and int(sv.shape[0]) > 0:
        start = max(0, int(seg.start_frame))
        end = min(end, int(sv.shape[0]))
        if end > start:
            return sv[start:end].clone()

    from ..lib.video_io import logical_frame_count

    total = logical_frame_count(plan.raw)
    end = min(end, total)
    start = max(0, int(seg.start_frame))
    if end <= start:
        return resolve_segment_raw_clip(plan, seg)
    return load_timeline_segment(plan.raw, start, end)


def source_passthrough_chunk(plan: DirectorPlan, seg) -> torch.Tensor:
    """Scaled source frames for skipped v2v segments with no generation cache yet."""
    log.info("source_passthrough_chunk called for segment %d", seg.index + 1)
    log.info("  -> resolve_segment_raw_clip...")
    raw_clip = resolve_segment_raw_clip(plan, seg)
    log.info("  -> resolve_segment_raw_clip returned: %s frames", raw_clip.shape[0])
    target_len = raw_clip.shape[0]
    if plan.output_mode == "fixed":
        log.info("  -> fit_canvas...")
        clip = fit_canvas(raw_clip, plan.width, plan.height)
    else:
        log.info("  -> fit_video_long_edge...")
        clip = fit_video_long_edge(raw_clip, plan.ref_max_size)
    log.info("  -> pad_or_trim_frames...")
    result = pad_or_trim_frames(clip, target_len).cpu().float()
    log.info("  -> source_passthrough_chunk done, returning %s frames", result.shape[0])
    return result


def segment_passthrough_chunk(plan: DirectorPlan, seg) -> torch.Tensor | None:
    """Best-effort fill for skipped segments from **real source video** only.

    Gen timelines (t2v/r2v have no source; i2v/fl2v use held refs as
    ``source_clip``) — never splice those into「全部导出」or the merge goes gray.
    Unselected gen segments must come from disk cache instead.
    """
    log.info("segment_passthrough_chunk called for segment %d (task=%s)", seg.index + 1, seg.task_key)
    if is_gen_timeline_plan(plan):
        log.info("  -> is_gen_timeline_plan: True, returning None")
        return None
    log.info("  -> is_gen_timeline_plan: False")
    if (plan.raw or {}).get("externalGroups", {}).get("active"):
        # External i2v/r2v packs also lack real camera footage for fill.
        if not needs_source_video(seg.task_key) or seg.task_key in {"i2v", "fl2v"}:
            log.info("  -> external groups active, no source video needed, returning None")
            return None
    log.info("  -> needs_source_video: %s", needs_source_video(seg.task_key))
    if needs_source_video(seg.task_key):
        try:
            log.info("  -> calling source_passthrough_chunk...")
            result = source_passthrough_chunk(plan, seg)
            log.info("  -> source_passthrough_chunk returned: %s", "tensor" if result is not None else "None")
            return result
        except Exception as e:
            log.error("  -> source_passthrough_chunk raised exception: %s", e)
            return None
    log.info("  -> returning None (no source video needed)")
    return None


def frames_label(seg) -> str:
    return f"帧 {seg.start_frame}–{seg.end_frame} ({seg.frame_count}f)"
