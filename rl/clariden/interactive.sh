zellij --session openfront
srun --time=11:00:00 -A infra01 --container-writable --environment=/users/alexpadula/projects/openfront-rl/rl/clariden/torch.toml --pty bash
cd ~/projects/openfront-rl/rl
python -m venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -r requirements.txt -c clariden/constraints.txt

python train.py \
    --curriculum \
    --num-envs 16 \
    --rollout-steps 1024 \
    --num-updates 10000 \
    --save-dir ./checkpoints \
    --anneal-lr

zellij attach openfront

srun --jobid= --overlap --environment=/capstor/store/cscs/swissai/infra01/reasoning/imgs/projects/verl_swiss:1/env.toml --pty bash

# unset SSL_CERT_FILE
