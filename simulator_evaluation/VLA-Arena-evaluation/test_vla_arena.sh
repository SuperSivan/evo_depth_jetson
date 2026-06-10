python vla_arena/vla_arena_client.py  \
--execution_horizon 10     \
--seed 27    \
--num_episodes_per_task 10  \  
--server_url ws://127.0.0.1:9000  \  
--log_out_dir ./logs/exp_h10_s27  
--save_video_mode all \
--max_episode_steps 300 
