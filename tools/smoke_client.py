import asyncio, json, sys, time
import numpy as np
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:9000"


def fake_obs(seed=0, mode="dual"):
    rng = np.random.default_rng(seed)
    imgs = [rng.integers(0, 255, (448, 448, 3), dtype=np.uint8).tolist() for _ in range(3)]
    return {
        "image": imgs,
        "state": [0.0] * 8,
        "prompt": "pick up the black bowl and place it on the plate",
        "image_mask": [1, 0, 0] if mode == "mono" else [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }


async def main():
    obs = fake_obs()
    payload = json.dumps(obs)
    print(f"payload {len(payload)/1e6:.1f} MB -> {URL}")
    async with websockets.connect(URL, max_size=100_000_000, open_timeout=30) as ws:
        t0 = time.time()
        await ws.send(payload)
        reply = await ws.recv()
        dt = time.time() - t0
        actions = json.loads(reply)
        arr = np.array(actions, dtype=np.float32)
        print(f"round-trip {dt:.3f} s")
        print(f"actions shape {arr.shape} dtype {arr.dtype}")
        print(f"first action[:8] {arr[0][:8].round(4).tolist()}")
        print(f"finite={np.isfinite(arr).all()} min={arr.min():.4f} max={arr.max():.4f}")


asyncio.run(main())
