# 板子访问链路（2026-11-10 实测）

## 链路

    Windows 本机 (liuqi)
      | ssh -i C:\Users\liuqi\.ssh\id_rsa, User root
      v
    124.71.225.34  华为云 ECS, hostname hcss-ecs-e47e, 内网 172.31.15.45
      | 本机侧监听 127.0.0.1:2222，由板子的反向隧道建立
      v ssh board  (= HostName 127.0.0.1, Port 2222, User nvidia,
      |              IdentityFile /root/.ssh/id_ed25519_board, IdentitiesOnly yes)
    板子 tegra-ubuntu (nvidia), 内网 192.168.101.13, JP7 R39.2.1

方向是板子主动拨出去，不是被拨入。板端 board-tunnel.service
（User=nvidia, Restart=always, RestartSec=10, MainPID 用 systemctl show -p MainPID 查）：

    /usr/bin/ssh -NT -o ExitOnForwardFailure=yes -o BatchMode=yes -o IdentitiesOnly=yes \
      -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/nvidia/.ssh/known_hosts \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o TCPKeepAlive=yes \
      -i /home/nvidia/.ssh/board_tunnel_ed25519 \
      -R 127.0.0.1:2222:127.0.0.1:22 \
      board-tunnel@124.71.225.34

## 常用命令

单条：

    ssh 124.71.225.34 'ssh board "uname -a"'

多行，用 stdin 传脚本，避免三层引号：

    ssh 124.71.225.34 "ssh board 'bash -s'" <<'INNER'
    cd ~/evo_depth_jetson && ls
    INNER

传文件，必须两跳，124 到板子约 230 KB/s：

    scp file 124.71.225.34:/tmp/ && ssh 124.71.225.34 'scp /tmp/file board:/tmp/'

大文件别走隧道，让板子自己从公网取，HF 走 hf-mirror.com 更稳。

## 不能走的路

板子直连办公网 IP。当前 ip -4 addr 只有 192.168.101.13 (wifi)、192.168.55.1 (USB)、
172.17.0.1 (docker)，过去记录的 172.25.80.139 已不在网卡上。就算同网段可达，
板端 authorized_keys 只认 124 那把 key，本机 -J 直连会被拒。

从 120 出发。120 到 124:22 的 TCP 是通的，但 120 没有 124 的公钥，
报 Permission denied (publickey,password)。

板子连 120/121/216。按既定要求禁止。

## 已知限制

sudo 需要密码，nvpmodel 和 jetson_clocks 跑不了。测速前需另行确认时钟状态，
看 /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 与 scaling_max_freq。

统一内存 7.5 GB，Evo-Depth 服务常驻约 3.6 GB，同时只能跑一份模型。

隧道断开会表现为 Connection timed out during banner exchange。板端有 Restart=always
和 RestartSec=10，通常 10 秒后自愈；若不自愈且无入口，只能现场处理。
