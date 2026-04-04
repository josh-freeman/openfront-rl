"""Run one headless training game and log obs vectors to /tmp/training_obs.log"""
import sys
import numpy as np
import torch
sys.path.insert(0, "/Users/joshua/openfront-rl/rl")
from env import OpenFrontEnv
from train import ActorCritic

def load_model(model_path, obs_dim, max_neighbors=16):
    device = torch.device("cpu")
    state = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    hidden_sizes = []
    i = 0
    while f"backbone.{i}.weight" in state_dict:
        hidden_sizes.append(state_dict[f"backbone.{i}.weight"].shape[0])
        i += 2
    if not hidden_sizes:
        hidden_sizes = [256, 256, 128]
    model = ActorCritic(obs_dim, max_neighbors, hidden_sizes=hidden_sizes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

env = OpenFrontEnv(map_name="World", num_opponents=5, difficulty="medium", ticks_per_step=10)
model, device = load_model("/Users/joshua/openfront-rl/rl/checkpoints/best_model.pt", 80)

obs, info = env.reset()
print(f"Game started. obs_dim={obs.shape}")

with open("/tmp/training_obs.log", "w") as f:
    step = 0
    while True:
        # Log the obs vector
        player = obs[:16]
        f.write(f"[step={step}] VEC: [{', '.join(f'{v:.5f}' for v in player)}]\n")
        f.write(f"  tiles={player[0]:.5f} troops={player[1]:.4f} gold={player[2]:.5f} pct={player[3]:.5f} inAtk={player[4]:.2f} outAtk={player[5]:.2f} units={player[6]:.3f} nNeigh={player[7]:.3f} silo={player[8]:.0f} port={player[9]:.0f} sam={player[10]:.0f} ships={player[11]:.2f} nukes={player[12]:.2f} tick={player[13]:.5f} bldRes={player[14]:.0f} actOk={player[15]:.0f}\n")
        # Log first 3 neighbors
        for ni in range(min(3, 16)):
            base = 16 + ni * 4
            f.write(f"  n{ni}: tiles={obs[base]:.5f} troops={obs[base+1]:.4f} rel={obs[base+2]:.2f} alive={obs[base+3]:.0f}\n")
        f.write("\n")
        f.flush()

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, value = model.get_action_and_value(obs_t)
        action_np = action.squeeze(0).cpu().numpy()

        f.write(f"  → type={int(action_np[0])} target={int(action_np[1])} bucket={int(action_np[2])} value={value.item():.3f}\n\n")
        f.flush()

        obs, reward, done, truncated, info = env.step(action_np)
        step += 1

        if step % 50 == 0:
            pct = info.get("territoryPct", obs[3])
            print(f"  step={step} territory={pct:.1%} reward={reward:.3f}")

        if done or truncated:
            print(f"Game over at step {step}. Won={info.get('weWon', False)}")
            break

env.close()
print(f"Training obs logged to /tmp/training_obs.log ({step} steps)")
