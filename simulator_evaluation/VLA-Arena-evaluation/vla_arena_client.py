
'''
python vla_arena/vla_arena_client.py  \
--execution_horizon 10     \
--seed 27    \
--num_episodes_per_task 10  \  
--server_url ws://127.0.0.1:9000  \  
--log_out_dir ./logs/exp_h10_s27  
--save_video_mode all \
--max_episode_steps 300 
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import logging
import math
import random
import asyncio
import traceback
import numpy as np
import draccus
import imageio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import websockets

from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.envs import OffScreenRenderEnv

VLA_ARENA_ENV_RESOLUTION = 256
VLA_ARENA_DUMMY_ACTION = [0.0] * 6 + [-1.0]

SUITE_CATEGORIES = {
    "safety_static_obstacles": "Safety",
    "safety_cautious_grasp": "Safety",
    "safety_hazard_avoidance": "Safety",
    "safety_state_preservation": "Safety",
    "safety_dynamic_obstacles": "Safety",
    "distractor_static_distractors": "Distractor",
    "distractor_dynamic_distractors": "Distractor",
    "extrapolation_preposition_combinations": "Extrapolation",
    "extrapolation_task_workflows": "Extrapolation",
    "extrapolation_unseen_objects": "Extrapolation",
    "long_horizon": "Long Horizon"
}

LEVELS = [0, 1, 2]
LEVEL_NAMES = {0: "L0", 1: "L1", 2: "L2"}


def get_suite_category(suite_name: str) -> str:
    return SUITE_CATEGORIES.get(suite_name, "Unknown")


def get_task_count(suite_name: str, level: int) -> int:
    if suite_name == "long_horizon" and level == 0:
        return 10
    return 5


@dataclass
class Args:
    server_url: str = "ws://0.0.0.0:8000"
    execution_horizon: int = 8
    task_suite_names: Optional[List[str]] = None
    num_episodes_per_task: int = 10
    max_episode_steps: Optional[int] = None
    seed: int = 1
    
    
    log_out_dir: str = "./logs/arena"
    video_out_path: str = None
    log_file: str = None
    
    save_video_mode: str = 'all'
    add_noise: bool = False
    randomize_color: bool = False
    adjust_light: bool = False
    camera_offset: bool = False
    policy_path: str = "remote"

    def __post_init__(self):
        if self.video_out_path is None:
            self.video_out_path = str(Path(self.log_out_dir) / "video")
        if self.log_file is None:
            self.log_file = str(Path(self.log_out_dir) / "log_file.txt")


class StatisticsManager:
    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0})))

    def record(self, category: str, suite: str, level: int, success: bool):
        self.stats[category][suite][level]["total"] += 1
        if success:
            self.stats[category][suite][level]["success"] += 1

    def get_level_rate(self, category: str, suite: str, level: int) -> float:
        data = self.stats[category][suite][level]
        return data["success"] / data["total"] if data["total"] > 0 else 0.0

    def get_suite_rate(self, category: str, suite: str) -> float:
        rates = [self.get_level_rate(category, suite, l) for l in LEVELS]
        return sum(rates) / 3.0

    def get_category_rate(self, category: str) -> float:
        suites = self.stats[category]
        if not suites:
            return 0.0
        return sum(self.get_suite_rate(category, s) for s in suites) / len(suites)

    def get_overall_rate(self) -> float:
        all_suite_rates = []
        for cat_name, suites in self.stats.items():
            for suite_name in suites.keys():
                all_suite_rates.append(self.get_suite_rate(cat_name, suite_name))
        
        return sum(all_suite_rates) / 11.0 if all_suite_rates else 0.0

    def print_summary(self, log_func=print):
        log_func("\n" + "=" * 80)
        log_func("[Overall Statistics]")
        log_func(f"Total Success Rate: {self.get_overall_rate() * 100:.2f}% (Avg over 11 suites)")
        
        log_func(f"\n{'=' * 80}")
        log_func("[Category Statistics]")
        log_func(f"{'Category':<20} {'Success Rate':<15}")
        log_func("-" * 80)
        for cat_name in sorted(self.stats.keys()):
            log_func(f"{cat_name:<20} {self.get_category_rate(cat_name) * 100:>6.2f}%")

        log_func(f"\n{'=' * 80}")
        log_func("[Suite & Level Statistics]")
        log_func(f"{'=' * 80}")
        for cat_name in sorted(self.stats.keys()):
            for suite_name in sorted(self.stats[cat_name].keys()):
                suite_rate = self.get_suite_rate(cat_name, suite_name)
                log_func(f"\n > {cat_name} > {suite_name} (Suite Rate: {suite_rate * 100:.2f}%)")
                level_strs = [f"{LEVEL_NAMES[l]}: {self.get_level_rate(cat_name, suite_name, l) * 100:.2f}%" for l in LEVELS]
                log_func(f"   {' | '.join(level_strs)}")
        log_func("\n" + "=" * 80)

    def save_to_json(self, filepath: str):
        data = {
            "overall_success_rate": self.get_overall_rate(),
            "categories": {}
        }
        for cat_name, suites in self.stats.items():
            cat_data = {
                "category_success_rate": self.get_category_rate(cat_name),
                "suites": {}
            }
            for suite_name, levels in suites.items():
                suite_data = {
                    "suite_success_rate": self.get_suite_rate(cat_name, suite_name),
                    "levels": {LEVEL_NAMES[l]: self.get_level_rate(cat_name, suite_name, l) for l in levels}
                }
                cat_data["suites"][suite_name] = suite_data
            data["categories"][cat_name] = cat_data

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def _get_env(task, args, level):
    bddl = Path(get_vla_arena_path('bddl_files')) / task.problem_folder / f'level_{level}' / task.bddl_file
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=VLA_ARENA_ENV_RESOLUTION,
        camera_widths=VLA_ARENA_ENV_RESOLUTION,
        camera_offset=args.camera_offset,
        color_randomize=args.randomize_color,
        add_noise=args.add_noise,
        light_adjustment=args.adjust_light
    ), task.language


def construct_payload(obs, task_description):
    agentview = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
    wrist = np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1])
    dummy_img = np.zeros_like(agentview)

    return {
        "image": [agentview.tolist(), wrist.tolist(), dummy_img.tolist()],
        "state": np.concatenate((
            obs['robot0_eef_pos'],
            _quat2axisangle(obs['robot0_eef_quat']),
            obs['robot0_gripper_qpos'],
        )).astype(np.float32).tolist(),
        "prompt": task_description,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17
    }


def _quat2axisangle(quat):
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


async def eval_single_episode(env, task, task_suite, websocket, args, logger, category_name, suite_name, level, task_id, ep_idx, rng):
    obs = env.reset()
    env.seed(args.seed + ep_idx)
    init_states = task_suite.get_task_init_states(level, task_id)
    init_state_idx = (ep_idx + rng.integers(0, 10)) % len(init_states)
    obs = env.set_init_state(init_states[init_state_idx])

    for _ in range(10):
        obs, _, _, _ = env.step(VLA_ARENA_DUMMY_ACTION)

    total_steps = 0
    frames = []
    done = False
    max_allowed_steps = args.max_episode_steps or (600 if suite_name == 'long_horizon' else 300)

    while total_steps < max_allowed_steps:
        try:
            if args.save_video_mode != 'none':
                frames.append(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))

            payload = construct_payload(obs, task.language)
            await websocket.send(json.dumps(payload))
            actions = json.loads(await websocket.recv())
            
            steps_to_execute = 1 if len(actions) == 1 else min(len(actions), args.execution_horizon)

            for i in range(steps_to_execute):
                if total_steps >= max_allowed_steps:
                    break
                action = actions[i][:7]
                obs, _, done, info = env.step(action)
                total_steps += 1
                if done:
                    break
            if done or total_steps >= max_allowed_steps:
                break
        except Exception as e:
            logger.error(f"Error in episode {ep_idx}: {e}\n{traceback.format_exc()}")
            break

    success = done
    logger.info(f"Episode {ep_idx + 1}: {'Success' if success else 'Failure'}")
    return success, frames


async def eval_vla_arena(args: Args) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    
    Path(args.log_out_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(args.log_file, mode='w'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    stats_mgr = StatisticsManager()

    websocket = await websockets.connect(args.server_url, max_size=200_000_000)
    logger.info(f"Connected to Server: {args.server_url}")
    logger.info(f"Config: Horizon={args.execution_horizon}, Episodes/Task={args.num_episodes_per_task}")

    try:
        benchmark_dict = benchmark.get_benchmark_dict()
        suite_names = args.task_suite_names if args.task_suite_names else [name for name in SUITE_CATEGORIES.keys() if name in benchmark_dict]
        logger.info(f"Suites to evaluate ({len(suite_names)}): {suite_names}")

        for suite_name in suite_names:
            if suite_name not in benchmark_dict:
                logger.warning(f"Suite {suite_name} not found, skipping.")
                continue

            category_name = get_suite_category(suite_name)
            logger.info(f"\n{'=' * 60}\nStarting Suite: {suite_name}\n{'=' * 60}")
            task_suite = benchmark_dict[suite_name]()

            for level in LEVELS:
                task_count = get_task_count(suite_name, level)
                logger.info(f"Level {LEVEL_NAMES[level]} ({task_count} tasks)")

                for task_id in range(task_count):
                    try:
                        task = task_suite.get_task_by_level_id(level, task_id)
                        logger.info(f"Task {task_id}: {task.language[:60]}...")
                        env, _ = _get_env(task, args, level)
                        
                        for ep_idx in range(args.num_episodes_per_task):
                            success, frames = await eval_single_episode(
                                env=env, task=task, task_suite=task_suite, websocket=websocket,
                                args=args, logger=logger, category_name=category_name,
                                suite_name=suite_name, level=level, task_id=task_id, ep_idx=ep_idx, rng=rng
                            )
                            stats_mgr.record(category_name, suite_name, level, success)

                            if args.save_video_mode != 'none' and frames:
                                video_dir = Path(args.video_out_path) / suite_name / LEVEL_NAMES[level]
                                video_dir.mkdir(parents=True, exist_ok=True)
                                vid_path = video_dir / f"task{task_id}_ep{ep_idx}_{'success' if success else 'failure'}.mp4"
                                imageio.mimsave(vid_path, frames, fps=10)
                        env.close()
                    except Exception as e:
                        logger.error(f"Task {task_id} failed: {e}\n{traceback.format_exc()}")
                        continue

        stats_mgr.print_summary(logger.info)
        json_path = Path(args.log_out_dir) / "statistics.json"
        stats_mgr.save_to_json(str(json_path))
        logger.info(f"Statistics saved to: {json_path}")

    finally:
        if websocket:
            await websocket.close()


if __name__ == '__main__':
    args = draccus.parse(Args)
    try:
        asyncio.run(eval_vla_arena(args))
    except KeyboardInterrupt:
        print("\nUser interrupted")
