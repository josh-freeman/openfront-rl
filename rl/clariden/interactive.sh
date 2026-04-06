zellij --session openfront
srun --time=6:00:00 -A infra01 --container-writable --environment=/users/alexpadula/projects/openfront-rl/rl/clariden/torch.toml --pty bash
cd ~/projects/openfront-rl/rl
python -m venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -r requirements.txt -c clariden/constraints.txt

python -u train.py \
    --num-agents-per-env 8 \
    --curriculum \
    --num-envs 16 \
    --rollout-steps 1024 \
    --minibatch-size 2048 \
    --num-updates 4000 \
    --lr 3.4e-4 \
    --vf-coef 1 \
    --hidden-sizes 512,512,256 \
    --save-interval 10 \
    --log-interval 5 \
    --potential-alpha 1 \
    --save-dir ./checkpoints/multiagent4

python -u train.py \
    --num-agents-per-env 1 \
    --num-nations 2 \
    --num-tribes 2 \
    --difficulty Medium \
    --maps plains,big_plains,ocean_and_land,half_land_half_ocean \
    --max-steps 10000 \
    --win-threshold 0.80 \
    --num-envs 16 \
    --rollout-steps 1024 \
    --num-updates 4000 \
    --lr 3.4e-4 \
    --vf-coef 1 \
    --hidden-sizes 512,512,256 \
    --save-interval 10 \
    --log-interval 5 \
    --potential-alpha 1 \
    --save-dir ./checkpoints/multiagent4

zellij attach openfront

srun --jobid= --overlap --environment=/capstor/store/cscs/swissai/infra01/reasoning/imgs/projects/verl_swiss:1/env.toml --pty bash

# unset SSL_CERT_FILE
