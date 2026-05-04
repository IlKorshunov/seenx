"""
scheduler
DAG + VRAM cap + per-task FLOPS → parallel steps
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    name: str
    vram_gb: float = 0.0
    flops: float = 0.0
    duration_est_sec: float = 0.0
    depends_on: list[str] = field(default_factory=list)


@dataclass
class ScheduledTask:
    task: Task
    start_sec: float
    end_sec: float
    worker_id: int


@dataclass
class ExecutionPlan:
    steps: list[list[ScheduledTask]]
    total_time_sec: float
    peak_vram_gb: float
    total_flops: float

    def print_summary(self) -> str:
        lines = [
            f"Total estimated time: {self.total_time_sec:.1f}s ({self.total_time_sec / 60:.1f}min)",
            f"Peak VRAM usage: {self.peak_vram_gb:.2f} GB",
            f"Total FLOPS: {self.total_flops:.2e}",
            f"Execution steps: {len(self.steps)}",
        ]
        t = 0.0
        for i, step in enumerate(self.steps):
            vram_sum, step_dur, body = 0.0, 0.0, []
            for st in step:
                tk = st.task
                vram_sum, step_dur = vram_sum + tk.vram_gb, max(step_dur, tk.duration_est_sec)
                body.append(f" [GPU] {tk.name} ({tk.duration_est_sec:.0f}s, {tk.vram_gb:.1f})")
            lines.append(f"Step {i + 1} (t={t:.1f}s, parallel={len(step)}, vram={vram_sum:.1f}):")
            lines.extend(body)
            t += step_dur
        text = os.linesep.join(lines)
        print(text)
        return text


class FeaturePlanner:
    """VRAM-limited Kahn + greedy pack per step (all tasks GPU)."""

    def __init__(self, vram_budget_gb: float = 8.0, max_parallel: int = 4, flops_per_sec: float = 1e13):
        self.vram_budget_gb, self.max_parallel, self.flops_per_sec = vram_budget_gb, max_parallel, flops_per_sec

    def _pack(self, tasks: list[Task], ready_key: Callable[[Task], tuple[float, ...]]) -> ExecutionPlan:
        task_map = {t.name: t for t in tasks}
        children: dict[str, list[str]] = defaultdict(list)
        in_degree = {t.name: 0 for t in tasks}
        for t in tasks:
            for dep in t.depends_on:
                if dep not in task_map:
                    raise ValueError(f"Task '{t.name}' depends on unknown task '{dep}'")
                children[dep].append(t.name)
                in_degree[t.name] += 1
        completed: set[str] = set()
        steps: list[list[ScheduledTask]] = []
        current_time = peak_vram = 0.0
        total_flops = sum(t.flops for t in tasks)
        while len(completed) < len(tasks):
            ready = [n for n, d in in_degree.items() if d == 0 and n not in completed]
            if not ready:
                raise RuntimeError(f"Cycle or bad deps: {set(task_map) - completed}")
            ordered = sorted(ready, key=lambda n: ready_key(task_map[n]))
            step, vram_used, wid = [], 0.0, 0
            for name in ordered:
                t = task_map[name]
                if wid >= self.max_parallel:
                    break
                if vram_used + t.vram_gb > self.vram_budget_gb:
                    continue
                step.append(ScheduledTask(task=t, start_sec=current_time, end_sec=current_time + t.duration_est_sec, worker_id=wid))
                vram_used, wid = vram_used + t.vram_gb, wid + 1
            if not step:
                t = task_map[ordered[0]]
                step = [ScheduledTask(task=t, start_sec=current_time, end_sec=current_time + t.duration_est_sec, worker_id=0)]
                vram_used = t.vram_gb
            peak_vram = max(peak_vram, vram_used)
            step_duration = max(st.task.duration_est_sec for st in step)
            for st in step:
                completed.add(st.task.name)
                for child in children[st.task.name]:
                    in_degree[child] -= 1
            steps.append(step)
            current_time += step_duration
        return ExecutionPlan(steps=steps, total_time_sec=current_time, peak_vram_gb=peak_vram, total_flops=total_flops)

    def schedule(self, tasks: list[Task]) -> ExecutionPlan:
        return self._pack(tasks, lambda t: (-t.vram_gb, -t.duration_est_sec))

    def estimate_serial_time(self, tasks: list[Task]) -> float:
        return sum(t.duration_est_sec for t in tasks)

    def speedup(self, tasks: list[Task]) -> float:
        return self.estimate_serial_time(tasks) / max(self.schedule(tasks).total_time_sec, 1e-6)

    def schedule_longest_job_first(self, tasks: list[Task]) -> ExecutionPlan:
        return self._pack(tasks, lambda t: (-t.duration_est_sec, -t.vram_gb))


def build_default_feature_tasks(video_duration_sec: float = 300.0) -> list[Task]:
    vd = video_duration_sec
    return [
        Task("shot_segmentation", vram_gb=1.5, flops=3e11, duration_est_sec=max(15, vd * 0.1)),
        Task("optical_flow", vram_gb=2.0, flops=8e11, duration_est_sec=max(30, vd * 0.3)),
        Task("face_detection", vram_gb=1.5, flops=4e11, duration_est_sec=max(20, vd * 0.15)),
        Task("pose_estimation", vram_gb=1.0, flops=3e11, duration_est_sec=max(15, vd * 0.12), depends_on=["face_detection"]),
        Task("clip_embeddings", vram_gb=2.5, flops=6e11, duration_est_sec=max(25, vd * 0.2)),
        Task("emotion_recognition", vram_gb=0.8, flops=2e11, duration_est_sec=max(10, vd * 0.08), depends_on=["face_detection"]),
        Task("whisper_transcribe", vram_gb=4.0, flops=2e12, duration_est_sec=max(60, vd * 0.5)),
        Task("source_separation", vram_gb=2.0, flops=5e11, duration_est_sec=max(30, vd * 0.25)),
        Task("audio_features", vram_gb=0.0, flops=5e9, duration_est_sec=max(5, vd * 0.03), depends_on=["source_separation"]),
        Task("text_features", vram_gb=0.5, flops=1e10, duration_est_sec=max(10, vd * 0.05), depends_on=["whisper_transcribe"]),
        Task("llm_features", vram_gb=0.0, flops=1e10, duration_est_sec=max(30, vd * 0.2), depends_on=["whisper_transcribe"]),
        Task("visual_features", vram_gb=0.0, flops=2e9, duration_est_sec=max(5, vd * 0.02), depends_on=["shot_segmentation", "optical_flow"]),
        Task("scene_features", vram_gb=0.0, flops=1e9, duration_est_sec=max(3, vd * 0.01), depends_on=["clip_embeddings", "shot_segmentation"]),
        Task("embedding_alignment", vram_gb=0.0, flops=5e8, duration_est_sec=5, depends_on=["clip_embeddings", "whisper_transcribe", "source_separation"]),
        Task("aggregation", vram_gb=0.0, flops=1e8, duration_est_sec=3, depends_on=["visual_features", "audio_features", "text_features", "scene_features", "emotion_recognition", "pose_estimation"]),
    ]


def detect_vram_budget_gb(fallback: float = 8.0) -> float:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=5, text=True)
        mibs = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if mibs:
            return max(fallback * 0.25, mibs[0] / 1024.0)
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, OSError):
        pass
    return fallback


def detect_flops_per_sec_guess(fallback: float = 1e13) -> float:
    env = os.environ.get("SEENX_FLOPS_PER_SEC")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return fallback


def tasks_minus_completed(tasks: list[Task], completed: Iterable[str]) -> list[Task]:
    done = set(completed)
    kept = [t for t in tasks if t.name not in done]
    for t in kept:
        t.depends_on = [d for d in t.depends_on if d not in done]
    return kept


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Feature plan CLI")
    p.add_argument("--vram-gb", type=float, default=-1.0, help="VRAM GB, -1=nvidia-smi")
    p.add_argument("--max-parallel", type=int, default=4)
    p.add_argument("--video-duration", type=float, default=300.0)
    p.add_argument("--completed-file", default="", help="one task name per line")
    p.add_argument("--algo", choices=("default", "ljf"), default="default", help="sort: vram vs duration")
    p.add_argument("--auto-flops", action="store_true")
    args = p.parse_args()
    vram = detect_vram_budget_gb(8.0) if args.vram_gb < 0 else args.vram_gb
    if args.vram_gb < 0:
        print(f"[auto] vram_budget_gb={vram:.2f}")
    flops_sec = detect_flops_per_sec_guess(1e13) if args.auto_flops else 1e13
    if args.auto_flops:
        print(f"[auto] flops_per_sec={flops_sec:.2e}")
    tasks = build_default_feature_tasks(args.video_duration)
    if args.completed_file and (dp := Path(args.completed_file)).is_file():
        done = [ln.strip() for ln in dp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        before, tasks = len(tasks), tasks_minus_completed(tasks, done)
        print(f"[skip] completed={len(done)}; {before} -> {len(tasks)}")
    planner = FeaturePlanner(vram_budget_gb=vram, max_parallel=args.max_parallel, flops_per_sec=flops_sec)
    sched = planner.schedule_longest_job_first(tasks) if args.algo == "ljf" else planner.schedule(tasks)
    sched.print_summary()
    serial = planner.estimate_serial_time(tasks)
    print(f"\n  Serial: {serial:.1f}s ({serial / 60:.1f}min)\n  Parallel: {sched.total_time_sec:.1f}s\n  Speedup: {planner.speedup(tasks):.2f}x")
