# mt50_evo1_client.py — MetaWorld MT50 client for EvoDepth websocket server.
import argparse
import asyncio
import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import metaworld  # noqa: F401
import numpy as np
import websockets
import yaml


def _default_flat_config() -> dict:
    """Defaults if YAML is missing; keep in sync with metaworld_eval.yaml."""
    return {
        "server_url": "ws://127.0.0.1:9000",
        "log_dir": "./metaworld_logs",
        "order_json": "mt50_order.json",
        "tasks_jsonl": "tasks.jsonl",
        "mujoco_gl": "egl",
        "camera_name": "corner2",
        "img_width": 448,
        "img_height": 448,
        "seed": 10,
        "episodes_per_task": 10,
        "episode_horizon": 400,
        "target_level": "all",
        "state_take": 8,
        "action_horizon": 17,
        "fallback_use_first_n": 5,
        "fallback_idx_list": None,
        "save_video": True,
        "video_fps": 10,
        "video_dup_frames": 1,
        "show_window": False,
        "save_image": False,
        "inspect_sample_per_episode": True,
        "inspect_dir": "inspect_frames",
        "apply_rot_180": True,
        "apply_center_crop": True,
        "crop_keep_ratio": 2 / 3,
        "inspect_save_step_tag": True,
    }


def load_metaworld_eval_yaml(config_path: Optional[Path]) -> Tuple[dict, Path]:
    """
    Load flat config dict. Returns (config, base_dir) for resolving relative paths.
    base_dir is the YAML file's parent, or the script directory if no file.
    """
    here = Path(__file__).resolve().parent
    cfg = _default_flat_config()
    path = Path(config_path) if config_path else here / "metaworld_eval.yaml"
    if not path.is_absolute():
        path = (here / path).resolve()
    base_dir = here
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        if not isinstance(user, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        cfg.update(user)
        base_dir = path.parent
    return cfg, base_dir


def resolve_paths(cfg: dict, base_dir: Path) -> dict:
    out = dict(cfg)
    for key in ("order_json", "tasks_jsonl", "log_dir", "inspect_dir"):
        if out.get(key):
            p = Path(out[key])
            if not p.is_absolute():
                out[key] = str((base_dir / p).resolve())
    return out


@dataclass
class EvalConfig:
    server_url: str
    log_path: str
    video_save_dir: str
    order_json: str
    tasks_jsonl: str
    camera_name: str
    img_size: tuple
    seed: int
    episodes_per_task: int
    episode_horizon: int
    target_level: str
    state_take: int
    action_horizon: int
    fallback_use_first_n: int
    fallback_idx_list: Optional[List[int]]
    save_video: bool
    video_fps: float
    video_dup_frames: int
    show_window: bool
    save_image: bool
    inspect_sample_per_episode: bool
    inspect_dir: str
    apply_rot_180: bool
    apply_center_crop: bool
    crop_keep_ratio: float
    inspect_save_step_tag: bool
    mujoco_gl: str = "egl"


def build_eval_config(flat: dict, run_name: Optional[str]) -> EvalConfig:
    log_dir = Path(flat["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(run_dir / "eval.txt")
    video_save_dir = str(run_dir / "videos")
    w, h = int(flat["img_width"]), int(flat["img_height"])
    fallback_list = flat.get("fallback_idx_list")
    if fallback_list is not None:
        fallback_list = [int(x) for x in fallback_list]
    return EvalConfig(
        server_url=str(flat["server_url"]),
        log_path=log_path,
        video_save_dir=video_save_dir,
        order_json=str(flat["order_json"]),
        tasks_jsonl=str(flat["tasks_jsonl"]),
        camera_name=str(flat["camera_name"]),
        img_size=(w, h),
        seed=int(flat["seed"]),
        episodes_per_task=int(flat["episodes_per_task"]),
        episode_horizon=int(flat["episode_horizon"]),
        target_level=str(flat["target_level"]),
        state_take=int(flat["state_take"]),
        action_horizon=int(flat["action_horizon"]),
        fallback_use_first_n=int(flat["fallback_use_first_n"]),
        fallback_idx_list=fallback_list,
        save_video=bool(flat["save_video"]),
        video_fps=float(flat["video_fps"]),
        video_dup_frames=int(flat["video_dup_frames"]),
        show_window=bool(flat["show_window"]),
        save_image=bool(flat["save_image"]),
        inspect_sample_per_episode=bool(flat["inspect_sample_per_episode"]),
        inspect_dir=str(flat["inspect_dir"]),
        apply_rot_180=bool(flat["apply_rot_180"]),
        apply_center_crop=bool(flat["apply_center_crop"]),
        crop_keep_ratio=float(flat["crop_keep_ratio"]),
        inspect_save_step_tag=bool(flat["inspect_save_step_tag"]),
        mujoco_gl=str(flat.get("mujoco_gl", "egl")),
    )


def parse_args():
    p = argparse.ArgumentParser(description="MetaWorld MT50 EvoDepth evaluation client")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to metaworld_eval.yaml (default: ./metaworld_eval.yaml next to this script)",
    )
    p.add_argument("--server_url", type=str, default=None, help="Override websocket URL")
    p.add_argument("--log_dir", type=str, default=None, help="Base directory for logs and videos")
    p.add_argument("--run_name", type=str, default=None, help="Subfolder under log_dir (default: timestamp)")
    p.add_argument("--horizon", type=int, default=None, help="Action chunk horizon")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None, help="Episodes per task")
    p.add_argument("--episode_horizon", type=int, default=None)
    p.add_argument(
        "--target_level",
        type=str,
        default=None,
        help="all | easy | medium | hard | very_hard",
    )
    p.add_argument("--camera_name", type=str, default=None)
    p.add_argument("--order_json", type=str, default=None)
    p.add_argument("--tasks_jsonl", type=str, default=None)
    return p.parse_args()


# ---------------- Utils ----------------
def encode_image_uint8_list(img_bgr: np.ndarray):
    return img_bgr.astype(np.uint8).tolist()


def obs_to_state(obs, take: int) -> List[float]:
    if isinstance(obs, dict):
        if "observation" in obs:
            arr = np.asarray(obs["observation"], dtype=np.float32).ravel()
        else:
            parts = [np.asarray(v).ravel() for v in obs.values()]
            arr = np.concatenate(parts).astype(np.float32)
    else:
        arr = np.asarray(obs, dtype=np.float32).ravel()
    return arr[: min(take, arr.shape[0])].tolist()


def center_crop_keep_ratio(rgb: np.ndarray, keep_ratio: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    keep_ratio = float(keep_ratio)
    keep_ratio = max(1e-6, min(1.0, keep_ratio))
    new_h = max(1, int(round(h * keep_ratio)))
    new_w = max(1, int(round(w * keep_ratio)))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return rgb[y0 : y0 + new_h, x0 : x0 + new_w, :]


def render_single_bgr(env, conf: EvalConfig) -> np.ndarray:
    rgb = env.render()
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    if conf.apply_rot_180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
        rgb = np.ascontiguousarray(rgb)

    if conf.apply_center_crop and (0.0 < conf.crop_keep_ratio < 1.0):
        h, w = rgb.shape[:2]
        keep = float(conf.crop_keep_ratio)
        new_h = max(1, int(round(h * keep)))
        new_w = max(1, int(round(w * keep)))
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        rgb = rgb[y0 : y0 + new_h, x0 : x0 + new_w, :].copy()
        rgb = np.ascontiguousarray(rgb)

    if conf.img_size is not None:
        rgb = cv2.resize(rgb, conf.img_size, interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(rgb)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

    if conf.show_window:
        try:
            cv2.imshow("MetaWorld", bgr)
            cv2.waitKey(1)
        except Exception:
            pass

    return bgr


def create_video_writer(env, video_name: str, conf: EvalConfig):
    os.makedirs(conf.video_save_dir, exist_ok=True)
    probe_frame = render_single_bgr(env, conf)
    h0, w0 = probe_frame.shape[:2]
    frame_size = (w0, h0)
    video_path = os.path.join(conf.video_save_dir, video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, conf.video_fps, frame_size)
    for _ in range(conf.video_dup_frames):
        video_writer.write(probe_frame)
    return video_writer


def write_video(video_writer, img_bgr: np.ndarray, conf: EvalConfig):
    try:
        if video_writer is not None:
            for _ in range(conf.video_dup_frames):
                video_writer.write(img_bgr)
    except Exception as e:
        log_write(conf, f"[video][ERROR] writer.write failed: {e}")


def save_episode_video(writer, video_name: str, task_idx: int, slug: str, ep_num: int, conf: EvalConfig):
    if writer is None:
        return
    try:
        video_path = os.path.join(conf.video_save_dir, video_name)
        writer.release()
        log_write(conf, f"[video] task={task_idx} slug={slug} ep={ep_num} saved video frames {video_path}")
    except Exception as e:
        log_write(conf, f"[video][ERROR] closing writer failed: {e}")


async def evo1_infer(ws, img_bgr: np.ndarray, state_vec: List[float], prompt: Optional[str] = None) -> np.ndarray:
    assert prompt is not None and len(prompt) > 0, "prompt should be non-empty"
    dummy_img = np.zeros((448, 448, 3), dtype=np.uint8)
    payload = {
        "image": [
            encode_image_uint8_list(img_bgr),
            encode_image_uint8_list(dummy_img),
            encode_image_uint8_list(dummy_img),
        ],
        "state": state_vec,
        "prompt": prompt,
        "image_mask": [1, 0, 0],
        "action_mask": [1, 1, 1, 1] + [0] * 20,
    }
    await ws.send(json.dumps(payload))
    data = json.loads(await ws.recv())
    return np.asarray(data, dtype=np.float32)


def save_sent_bgr_frame(img_bgr: np.ndarray, ep_num: int, idx: int, slug: str, conf: EvalConfig, step: Optional[int] = None):
    os.makedirs(conf.inspect_dir, exist_ok=True)
    tag = f"step{step:04d}" if (conf.inspect_save_step_tag and step is not None) else "stepNA"
    out = os.path.join(conf.inspect_dir, f"ep{ep_num:03d}_idx{idx}_{slug}_{tag}.png")
    img_bgr_safe = np.ascontiguousarray(img_bgr)
    cv2.imwrite(out, img_bgr_safe)
    h, w = img_bgr_safe.shape[:2]
    print(f"[inspect] saved {out}  size={w}x{h}  (identical to VLA input)")


def log_write(conf: EvalConfig, text: str):
    print(text)
    with open(conf.log_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ---------------- Prompt loader ----------------
class PromptBook:
    def __init__(self, jsonl_path: str):
        self.by_idx: Dict[int, str] = {}
        self.by_slug: Dict[str, str] = {}
        self.seq: List[str] = []

        if not os.path.exists(jsonl_path):
            print(f"[WARN] {jsonl_path} not found; prompts will be empty.")
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for i, obj in enumerate(lines):
            task_txt = str(obj.get("task", "")).strip()
            idx_val = obj.get("idx", obj.get("task_index"))
            if idx_val is not None:
                try:
                    self.by_idx[int(idx_val)] = task_txt
                except Exception:
                    pass
            if "slug" in obj:
                try:
                    self.by_slug[str(obj["slug"])] = task_txt
                except Exception:
                    pass
            self.seq.append(task_txt)

    def get(self, idx: int, slug: Optional[str] = None) -> str:
        if idx in self.by_idx:
            return self.by_idx[idx]
        if slug is not None and slug in self.by_slug:
            return self.by_slug[slug]
        if 0 <= idx < len(self.seq):
            return self.seq[idx]
        return ""


# ---------------- Order & groups loader ----------------
def load_order_and_groups(
    conf: EvalConfig,
    total_envs: int,
):
    order_path = conf.order_json
    if os.path.exists(order_path):
        with open(order_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ordered_indices = list(map(int, data["ordered_indices"]))
        groups = {k: set(v) for k, v in data["groups"].items()}
        idx_to_slug = {int(k): v for k, v in data["idx_to_slug"].items()}
        print(f"[INFO] Loaded order from {order_path} (len={len(ordered_indices)})")
        log_write(conf, "[INFO] Metaworld Evaluation Begins ...")
        return ordered_indices, groups, idx_to_slug

    if conf.fallback_idx_list:
        idx_list = [i for i in conf.fallback_idx_list if 0 <= i < total_envs]
    else:
        idx_list = list(range(min(conf.fallback_use_first_n, total_envs)))
    print("[WARN] order json not found; falling back to:", idx_list)

    idx_to_slug = {i: f"task-{i}" for i in idx_list}
    groups = {"easy": set(), "medium": set(), "hard": set(), "very_hard": set()}
    return idx_list, groups, idx_to_slug


# ---------------- Core eval (MT50 only) ----------------
async def eval_mt50_with_groups(conf: EvalConfig, prompts: PromptBook):
    envs = gym.make_vec(
        "Meta-World/MT50",
        vector_strategy="sync",
        seed=conf.seed,
        render_mode="rgb_array",
        camera_name=conf.camera_name,
    )
    total_envs = len(envs.envs)

    ordered_indices, groups, idx_to_slug = load_order_and_groups(conf, total_envs)
    ordered_indices = [i for i in ordered_indices if 0 <= i < total_envs]

    if conf.target_level.lower() != "all":
        allowed_slugs = groups.get(conf.target_level.lower(), set())
        before = len(ordered_indices)
        ordered_indices = [i for i in ordered_indices if idx_to_slug.get(i, "") in allowed_slugs]
        print(f"[INFO] Filtered tasks: keep only {conf.target_level} ({len(ordered_indices)}/{before})")

    success_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    trials_counts: Dict[int, int] = {i: 0 for i in ordered_indices}
    group_success = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}
    group_trials = {k: 0 for k in ["easy", "medium", "hard", "very_hard"]}

    async with websockets.connect(conf.server_url, max_size=100_000_000) as ws:
        for idx in ordered_indices:
            sub = envs.envs[idx]
            slug = idx_to_slug.get(idx, f"task-{idx}")
            task_prompt = prompts.get(idx, slug=slug)

            gname_for_task = None
            for gname in group_trials.keys():
                if slug in groups.get(gname, set()):
                    gname_for_task = gname
                    break

            for ep in range(conf.episodes_per_task):
                for obj in (sub, getattr(sub, "unwrapped", None)):
                    fn = getattr(obj, "iterate_goal_position", None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                        break

                inspect_choice = conf.inspect_sample_per_episode
                saved_this_episode = False

                obs, _ = sub.reset(seed=conf.seed + ep)
                trials_counts[idx] += 1
                if gname_for_task is not None:
                    group_trials[gname_for_task] += 1

                steps = 0
                done = False
                video_name = f"task{idx:02d}_{slug}_ep{ep+1:03d}.mp4"
                video_writer = None if not conf.save_video else create_video_writer(sub, video_name, conf)

                try:
                    a0 = np.zeros(sub.action_space.shape, dtype=np.float32)
                    a0 = np.clip(a0, sub.action_space.low, sub.action_space.high)
                    obs, _, _, _, _ = sub.step(a0)
                except Exception:
                    pass

                while steps < conf.episode_horizon and not done:
                    img_bgr = render_single_bgr(sub, conf)

                    if conf.save_video:
                        write_video(video_writer, img_bgr, conf)

                    if conf.save_image and inspect_choice and (not saved_this_episode):
                        save_sent_bgr_frame(
                            img_bgr,
                            ep_num=ep + 1,
                            idx=idx,
                            slug=slug,
                            conf=conf,
                            step=steps if conf.inspect_save_step_tag else None,
                        )
                        saved_this_episode = True

                    state_vec = obs_to_state(obs, conf.state_take)
                    actions = await evo1_infer(ws, img_bgr, state_vec, prompt=task_prompt)

                    for i in range(conf.action_horizon):
                        a4 = np.asarray(actions[i][:4], dtype=np.float32)
                        a4 = np.clip(a4, sub.action_space.low, sub.action_space.high)
                        obs, _, terminated, truncated, info = sub.step(a4)
                        steps += 1

                        if isinstance(info, dict) and info.get("success", 0) == 1:
                            success_counts[idx] += 1
                            if gname_for_task is not None:
                                group_success[gname_for_task] += 1
                            done = True
                            break

                        if terminated or truncated or steps >= conf.episode_horizon:
                            done = True
                            break

                if done and conf.save_video:
                    final_frame = render_single_bgr(sub, conf)
                    write_video(video_writer, final_frame, conf)
                    save_episode_video(video_writer, video_name, idx, slug, ep + 1, conf)

            s = success_counts[idx]
            t = trials_counts[idx]
            task_rate = s / max(1, t)
            msg = (
                f"[Task {idx} {slug}] {task_prompt} finished {conf.episodes_per_task} episodes -> "
                f"success_rate={task_rate:.3f}  (s={s}, t={t})"
            )
            log_write(conf, msg)

    envs.close()

    per_task: Dict[str, float] = {}
    for idx in ordered_indices:
        slug = idx_to_slug.get(idx, f"task-{idx}")
        s, t = success_counts[idx], trials_counts[idx]
        per_task[slug] = (s / t) if t > 0 else 0.0

    per_group: Dict[str, float] = {}
    for gname in ["easy", "medium", "hard", "very_hard"]:
        s, t = group_success[gname], group_trials[gname]
        per_group[gname] = (s / t) if t > 0 else 0.0

    overall = sum(success_counts.values()) / max(1, sum(trials_counts.values()))

    return per_task, per_group, overall


async def _amain():
    args = parse_args()
    raw, base_dir = load_metaworld_eval_yaml(Path(args.config) if args.config else None)
    flat = resolve_paths(raw, base_dir)

    if args.server_url is not None:
        flat["server_url"] = args.server_url
    if args.log_dir is not None:
        flat["log_dir"] = args.log_dir
    if args.horizon is not None:
        flat["action_horizon"] = args.horizon
    if args.seed is not None:
        flat["seed"] = args.seed
    if args.episodes is not None:
        flat["episodes_per_task"] = args.episodes
    if args.episode_horizon is not None:
        flat["episode_horizon"] = args.episode_horizon
    if args.target_level is not None:
        tl = args.target_level.lower()
        allowed = {"all", "easy", "medium", "hard", "very_hard"}
        if tl not in allowed:
            raise SystemExit(f"--target_level must be one of {sorted(allowed)}")
        flat["target_level"] = tl
    if args.camera_name is not None:
        flat["camera_name"] = args.camera_name
    if args.order_json is not None:
        p = Path(args.order_json)
        flat["order_json"] = str(p if p.is_absolute() else base_dir / p)
    if args.tasks_jsonl is not None:
        p = Path(args.tasks_jsonl)
        flat["tasks_jsonl"] = str(p if p.is_absolute() else base_dir / p)

    conf = build_eval_config(flat, args.run_name)
    os.environ.setdefault("MUJOCO_GL", conf.mujoco_gl)
    gym.logger.min_level = gym.logger.ERROR

    prompts = PromptBook(conf.tasks_jsonl)
    per_task, per_group, overall = await eval_mt50_with_groups(conf, prompts)

    avg = (
        per_group.get("easy", 0.0)
        + per_group.get("medium", 0.0)
        + per_group.get("hard", 0.0)
        + per_group.get("very_hard", 0.0)
    ) / 4

    log_write(conf, f"\n==== Evaluation Log ====\nLog file: {conf.log_path}")
    log_write(conf, f"Videos dir: {conf.video_save_dir}")
    log_write(conf, f"Target difficulty: {conf.target_level}")
    log_write(conf, f"Server URL: {conf.server_url}")
    log_write(conf, f"Episodes per task: {conf.episodes_per_task}")
    log_write(conf, f"Episode horizon: {conf.episode_horizon}")
    log_write(conf, f"Action chunk horizon: {conf.action_horizon}")
    log_write(conf, f"Seed: {conf.seed}\n")

    log_write(conf, "==== Per-task success rate ====")
    for slug, rate in per_task.items():
        log_write(conf, f"{slug:24s}  {rate:.3f}")

    log_write(conf, "\n==== Difficulty buckets ====")
    log_write(conf, f"easy      : {per_group.get('easy', 0.0):.3f}")
    log_write(conf, f"medium    : {per_group.get('medium', 0.0):.3f}")
    log_write(conf, f"hard      : {per_group.get('hard', 0.0):.3f}")
    log_write(conf, f"very_hard : {per_group.get('very_hard', 0.0):.3f}")

    log_write(conf, f"\n==== Overall (all trials) ====\n{overall:.3f}")
    log_write(conf, f"==== Average of four difficulty buckets ====\n{avg:.3f}")


if __name__ == "__main__":
    asyncio.run(_amain())
