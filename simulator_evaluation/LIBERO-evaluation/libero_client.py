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
from tqdm import tqdm
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["MUJOCO_GL"] = "osmes"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]


steps = {
    "libero_spatial": 220,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_10": 520,
}

HORIZON_BY_TASK = {
    "libero_spatial": 8,
    "libero_goal": 15,
    "libero_object": 15,
    "libero_10": 20,
}

############################################
# ========= CLI  =========
############################################

def get_args():
    parser = argparse.ArgumentParser(description="LIBERO client")

    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override horizon (default: task-specific: libero_spatial=8, libero_goal/object=15, libero_10=20)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=["libero_spatial"],
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="ws://0.0.0.0:9000",
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        dest="denosing_steps",
    )
    
    parser.add_argument(
        "--test_time",
        type=int,
        default=1,
    )
    parser.add_argument("--log_file", type=str, default=None, help="Log file path")
    parser.add_argument("--video_log_dir", type=str, default=None, help="Video save directory")
    parser.add_argument(
        "--stream_port",
        type=int,
        default=None,
        help="Enable MJPEG web stream on this port (e.g. 8080). Open http://<host>:<port>/ in browser.",
    )
    parser.add_argument(
        "--stream_host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the web stream server (default: 0.0.0.0)",
    )

    parsed = parser.parse_args()

    parsed.horizons = []
    parsed.max_steps = []
    for suite in parsed.task_suites:
        h = parsed.horizon if parsed.horizon is not None else HORIZON_BY_TASK.get(suite, 15)
        parsed.horizons.append(h)
        parsed.max_steps.append(steps[suite] // h)
    parsed.SEED = parsed.seed
    parsed.SERVER_URL = parsed.server_url
    parsed.num_episodes = parsed.episodes
    parsed.ckpt_name = (
        f"S{parsed.SEED}_h{parsed.horizons[0]}_d{parsed.denosing_steps}_test{parsed.test_time}"
    )
    if parsed.log_file is None:
        parsed.log_file = f"log_file/{parsed.ckpt_name}.txt"
    if parsed.video_log_dir is None:
        parsed.video_log_dir = f"video_log_file/{parsed.ckpt_name}"

    return parsed


args = get_args()

########################################
# ========= Logging  =========
########################################

os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(args.log_file, mode="a"),
        logging.StreamHandler(),
    ],
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

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }

    env = OffScreenRenderEnv(**env_args)

    env.seed(seed)

    np.random.seed(seed)
    random.seed(seed)

    return env, task_description


def build_frame(obs):
    return np.hstack([
        np.rot90(obs["agentview_image"], 2),
        np.rot90(obs["robot0_eye_in_hand_image"], 2),
    ])


def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        print(f"💾 Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"⚠️ No frame data, video was not generated: {filepath}")


async def run(
    SERVER_URL: str,
    max_steps: int = None,
    num_episodes: int = None,
    horizon=None,
    task_suite_name=None,
    streamer=None,
):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Numbers of tasks: {num_tasks_in_suite}")

    total_success = 0
    total_episodes = 0
    total_steps = 0

    async with websockets.connect(SERVER_URL) as ws:
        log.info(f"===========================Start task suite {task_suite_name}========================")
        log.info(f"Horizon is {horizon}")
        total_false = 0
        for task_id in tqdm(
            range(num_tasks_in_suite),
            desc=f"[Suite {task_suite_name}] Tasks",
            leave=True
        ):


            

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=448, seed=args.SEED)

        

            log.info(f"\n========= Start Task {task_id+1}: {task_description} =========")

            task_success = 0
            task_episodes = min(num_episodes, len(initial_states))
            rng = np.random.RandomState(args.SEED)

            state_ids = rng.choice(
                len(initial_states),   
                task_episodes,         
                replace=False
            )
            logging.info(f"Random selected episode indices for Task {task_id+1}: {state_ids}")
            for ep in tqdm(
                range(task_episodes),
                desc=f"Task {task_id+1} Episodes",
                leave=False
            ):

                print(f"\n===== Task {task_id+1} | Episode {ep+1} =====")
                episode_seed = args.SEED  

                env.seed(episode_seed)
                np.random.seed(episode_seed)
                random.seed(episode_seed)


                obs = env.reset()
                

                state_id = state_ids[ep]
                obs = env.set_init_state(initial_states[state_id])
                env.sim.forward()
                t = 0
                while t < 10:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        

                prompt = str(task_description)
                print(prompt)
                episode_done = False
                max_step = 0
                frames = []

                for step in range(max_steps):
                    max_step += 1

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
                        # print(action[:7])
                        if action[6]>0.5:
                            action[6] = -1
                        else:
                            action[6] = 1
                        
                        
                       
                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            print(f"❌ Failed to execute action in environment: {ve}")
                            episode_done = False
                            break

                        
                        frame = build_frame(obs)
                        frames.append(frame)
                        if streamer is not None:
                            streamer.update(
                                frame,
                                task_id=task_id + 1,
                                episode=ep + 1,
                                step=max_step,
                                status="success" if done else "running",
                            )

                        # print(f"[Step {step}] reward={reward:.2f}, done={done}")
                        if done:
                            print(" Task completed.")
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += max_step
                            break
                    if episode_done:
                        break

                # 保存视频（文件名带 task_id）
                save_video(frames, f"task{task_id+1}_episode{ep+1}.mp4", fps=30, save_dir=f"{args.video_log_dir}/{task_suite_name}")

                if episode_done:
                    log.info(f"Task {task_id+1} | Episode {ep+1}: ✅ Success")
                else:
                    log.info(f"Task {task_id+1} | Episode {ep+1}: ❌ Fail")
                    total_false+=1
                    # if total_false>=11:
                    #     break
                


            log.info(f"========= Task {task_id+1} Summary: {task_success}/{task_episodes} Success =========")
            total_episodes += task_episodes
            

        # ======= 全部总结 =======
        log.info("\n========= Overall Summary =========")
        log.info(f"✅ Total Successful Episodes: {total_success}/{total_episodes}")
        if total_episodes > 0:
            log.info(f"Average Steps: {total_steps / total_episodes:.2f}")

# ========= 启动入口 =========
if __name__ == "__main__":
    # 全局随机种子
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    os.environ["PYTHONHASHSEED"] = str(args.SEED)

    streamer = None
    if args.stream_port is not None:
        from web_stream import WebStreamServer

        streamer = WebStreamServer(host=args.stream_host, port=args.stream_port)
        streamer.start()
        log.info(
            f"Web stream enabled: http://127.0.0.1:{args.stream_port}/ "
            f"(MJPEG: /video)"
        )

    for name, max_steps, horizon in zip(args.task_suites, args.max_steps, args.horizons):
        asyncio.run(
            run(
                SERVER_URL=args.SERVER_URL,
                max_steps=max_steps,
                num_episodes=args.num_episodes,
                horizon=horizon,
                task_suite_name=name,
                streamer=streamer,
            )
        )