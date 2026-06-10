import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random
import argparse
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["MUJOCO_GL"] = "osmes"
from tqdm import tqdm
LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

steps = {
    "libero_spatial": 660,
    "libero_goal": 900,
    "libero_object": 840,
    "libero_10": 1560,
}

HORIZON_BY_TASK = {
    "libero_spatial": 8,
    "libero_goal": 15,
    "libero_object": 15,
    "libero_10": 20,
}


def get_args():
    parser = argparse.ArgumentParser(description="LIBERO-PLUS client")
    parser.add_argument("--task_suites", nargs="+", default=["libero_spatial"])
    parser.add_argument("--horizon", type=int, default=None, help="Override (default: from HORIZON_BY_TASK)")
    parser.add_argument("--server_url", type=str, default="ws://127.0.0.1:9000", dest="SERVER_URL")
    parser.add_argument("--filter_category", type=str, default="background")
    parser.add_argument(
        "--filter_json_path",
        type=str,
        default="./LIBERO-plus/libero/libero/benchmark/task_classification.json",
        help="From your LIBERO-plus clone root (see README).",
    )
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, dest="SEED")
    parser.add_argument("--ckpt_name", type=str, default="YOUR_CKPT_NAME")
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--video_log_dir", type=str, default=None)
    p = parser.parse_args()
    p.horizons = []
    p.max_steps = []
    for suite in p.task_suites:
        h = p.horizon if p.horizon is not None else HORIZON_BY_TASK.get(suite, 15)
        p.horizons.append(h)
        p.max_steps.append(steps.get(suite, 900))
    p.horizon = p.horizons[0] if p.horizon is None else p.horizon
    if p.log_file is None:
        p.log_file = os.path.join("./logs", "libero_plus", f"{p.ckpt_name}.txt")
    if p.video_log_dir is None:
        p.video_log_dir = os.path.join("./logs", "libero_plus", p.ckpt_name, "videos")
    return p


args = get_args()

log_dir = os.path.dirname(args.log_file)
os.makedirs(log_dir, exist_ok=True)


os.makedirs(args.video_log_dir, exist_ok=True)

def fliter(filter_category, task_suite_name, json_path=None):
    if json_path is None:
        json_path = "./LIBERO-plus/libero/libero/benchmark/task_classification.json"
    with open(json_path, 'r', encoding='utf-8') as file:
        task_json = json.load(file)
    if filter_category != "":
        task_lists = task_json[task_suite_name]
        filter_ids = [t['id'] - 1 for t in task_lists if any(cat in t['category'].lower() for cat in filter_category.split(','))]
        #log_message(f'Evaluating {len(filter_ids)} tasks in perturbation category [{filter_category}]", log_file)
        return filter_ids
    else:
        task_lists = task_json[task_suite_name]
        all_ids = [t['id'] - 1 for t in task_lists]
        return all_ids
    
    

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        
        logging.FileHandler(args.log_file, mode='a'),
        logging.StreamHandler()
    ]

)
log = logging.getLogger(__name__)


def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    data = {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }
    return data


def get_libero_env(task, resolution=448, seed=args.SEED):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        print(f"💾 Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"⚠️ No frame data, video was not generated: {filepath}")







async def run(SERVER_URL: str, max_steps: int = None, num_episodes: int = None, horizon=None, task_suite_name=None, filter_ids=None):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    print(f"Numbers of tasks: {num_tasks_in_suite}")

    total_success = 0
    total_episodes = 0
    total_steps = 0
   
    async with websockets.connect(SERVER_URL) as ws:
       
        log.info(f"===========================Start task suite {task_suite_name}========================")

        for idx, task_id in enumerate(tqdm(filter_ids, desc=f"{task_suite_name} tasks")):

            print(f"task_id{task_id}")
            

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=448, seed=args.SEED)

           

            task_success = 0
            task_episodes = min(num_episodes, len(initial_states))

            for ep in range(task_episodes):
                

                env.reset()


                obs = env.set_init_state(initial_states[ep])
                t = 0
                while t < 10:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        

                prompt = str(task_description)
                
                episode_done = False
                max_step = 0
                frames = []

                for step in range(max_steps // horizon):
                    max_step += horizon

                    send_data = obs_to_json_dict(obs, prompt)
                    await ws.send(json.dumps(send_data))
                    

                    result = await ws.recv()
                    try:
                        action_list = json.loads(result)
                        actions = np.array(action_list)
                        
                    except Exception as e:
                        print(f"❌ Failed to parse action: {e}, content: {result}")
                        break

                    
                    for i in range(horizon):
                        action = actions[i].tolist()
                       
                        if action[6]>0.5:
                            action[6] = -1
                        else:
                            action[6] = 1
                        
                        
                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            print(f"❌ Environment step failed: {ve}")
                            episode_done = False
                            break

                        
                        frame = np.hstack([
                            np.rot90(obs["agentview_image"], 2),
                            np.rot90(obs["robot0_eye_in_hand_image"], 2)
                        ])
                        frames.append(frame)

                        
                        if done:
                            print("🎯 Task completed.")
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += max_step
                            break
                    if episode_done:
                        break

                
                save_video(frames, f"task{task_id}_episode{ep+1}.mp4", fps=30, save_dir = f"{args.video_log_dir}/{task_suite_name}/{args.filter_category}")

                if episode_done:
                    log.info(f"Task {task_id} | {idx}task | Episode {ep+1}: ✅ Success")
                else:
                    log.info(f"Task {task_id} | Episode {ep+1}: ❌ Fail")

                # exit(0)

            
            total_episodes += task_episodes

        
        log.info("\n========= Overall Summary =========")
        log.info(f"✅ Total successful episodes: {total_success}/{total_episodes} | Success Rate: {total_success / total_episodes * 100:.2f}%")
        if total_episodes > 0:
            log.info(f" Average steps: {total_steps / total_episodes:.2f}")


if __name__ == "__main__":
    
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    
    for name, max_steps, horizon in zip(args.task_suites, args.max_steps, args.horizons):
        filter_ids = fliter(filter_category=args.filter_category, task_suite_name=name, json_path=args.filter_json_path)
        num_tasks_in_suite = len(filter_ids)
        asyncio.run(run(SERVER_URL=args.SERVER_URL,
                        max_steps=max_steps,
                        num_episodes=args.num_episodes,
                        horizon=horizon,
                        task_suite_name=name,
                        filter_ids=filter_ids))