#!/usr/bin/env python3

import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Inspect model")
parser.add_argument("--best", action="store_true",
                    help="Will inspect best-status.pt instead of status.pt")
args = parser.parse_args()

# 1. Load the state dictionary
dict_path = "best-status.pt" if args.best else "status.pt"
pretrained_dict = torch.load(dict_path, map_location='cpu', weights_only=False)['model_state']

# 2. Channel layout
# v2 conv input: 19 env bit-planes (EnvironmentFlag bit positions 0..18)
#                + 9 monster attribute channels (compute_monster_attrs, indices 19..27)
ENV_CHANNEL_NAMES = [
    "Player",       # bit 0
    "Wall",         # bit 1
    "PrevTrigger",  # bit 2
    "NextTrigger",  # bit 3
    "WarpTrigger",  # bit 4
    "Door",         # bit 5
    "Missile",      # bit 6
    "Monster",      # bit 7
    "UnknownObject",# bit 8
    "Crucifix",     # bit 9
    "Barrel",       # bit 10
    "Chest",        # bit 11
    "Sarcophagus",  # bit 12
    "Item",         # bit 13
    "Explored",     # bit 14
    "Visible",      # bit 15
    "Interactable", # bit 16
    "Open",         # bit 17
    "Goal",         # bit 18
]
NR_ENV_CHANNELS = len(ENV_CHANNEL_NAMES)  # 19

MONSTER_ATTR_NAMES = [
    "hp_ratio",     # 19
    "level",        # 20
    "unique",       # 21
    "walk_speed",   # 22
    "atk_speed",    # 23
    "fire_resist",  # 24
    "light_resist", # 25
    "magic_resist", # 26
    "is_ranged",    # 27
]

# 3. Scalar encoder input names (46-float base + optional env-appended; see compute_scalars docstring)
SCALAR_NAMES = [
    "dungeon_level",        # 0
    "char_level",           # 1
    "hp",                   # 2
    "mana",                 # 3
    "hero_dir",             # 4
    "strength",             # 5
    "magic",                # 6
    "dexterity",            # 7
    "vitality",             # 8
    "weapon_dam_min",       # 9
    "weapon_dam_max",       # 10
    "armor_class",          # 11
    "fire_resist",          # 12
    "lightning_resist",     # 13
    "magic_resist",         # 14
    "mana_shield",          # 15
    "pot_small_hp",         # 16
    "pot_scroll_heal",      # 17
    "pot_full_hp",          # 18
    "pot_small_mana",       # 19
    "pot_full_mana",        # 20
    "pot_rejuv",            # 21
    "pot_full_rejuv",       # 22
    "setlvl_none",          # 23
    "setlvl_skelking",      # 24
    "setlvl_bonechamber",   # 25
    "setlvl_maze",          # 26
    "setlvl_poisonwater",   # 27
    "setlvl_vampirenest",   # 28
    "setlvl_arena1",        # 29
    "setlvl_arena2",        # 30
    "setlvl_arena3",        # 31
    "avail_firebolt",       # 32
    "avail_chargedbolt",    # 33
    "avail_firewall",       # 34
    "avail_stonecurse",     # 35
    "avail_manashield",     # 36
    "avail_phasing",        # 37
    "avail_fireball",       # 38
    "lvl_firebolt",         # 39
    "lvl_chargedbolt",      # 40
    "lvl_firewall",         # 41
    "lvl_stonecurse",       # 42
    "lvl_manashield",       # 43
    "lvl_phasing",          # 44
    "lvl_fireball",         # 45
    # v4 env appends one extra scalar:
    "stuck_frac",           # 46
]

# 4. Locate first conv layer
if 'image_conv.network.0.weight' in pretrained_dict:
    weight_key = 'image_conv.network.0.weight'
    bias_key   = 'image_conv.network.0.bias'
elif 'image_conv.trunk_a.0.weight' in pretrained_dict:
    weight_key = 'image_conv.trunk_a.0.weight'
    bias_key   = 'image_conv.trunk_a.0.bias'
else:
    raise KeyError("Cannot find first conv layer in state dict")

weights = pretrained_dict[weight_key]
biases  = pretrained_dict[bias_key] if bias_key in pretrained_dict else None

print(f"--- Conv layer '{weight_key}' ---")
print(f"Shape: {weights.shape}  (out_channels, in_channels, kH, kW)")

# 5. Env bit-plane channel stats
hdr = f"\n{'idx':<5} {'name':<16} {'mean':>10} {'std':>10} {'rms':>10} {'min':>10} {'max':>10}"
print(f"\n--- Env bit-plane channel stats (0..{NR_ENV_CHANNELS-1}) ---{hdr}")
for idx, name in enumerate(ENV_CHANNEL_NAMES):
    w = weights[:, idx, :, :]
    rms = w.pow(2).mean().sqrt().item()
    print(f"{idx:<5} {name:<16} {w.mean().item():>10.6f} {w.std().item():>10.6f} "
          f"{rms:>10.6f} {w.min().item():>10.6f} {w.max().item():>10.6f}")

# 6. Monster attribute channel stats
start = NR_ENV_CHANNELS
print(f"\n--- Monster attribute channel stats ({start}..{start+len(MONSTER_ATTR_NAMES)-1}) ---{hdr}")
for i, name in enumerate(MONSTER_ATTR_NAMES):
    idx = start + i
    w = weights[:, idx, :, :]
    rms = w.pow(2).mean().sqrt().item()
    print(f"{idx:<5} {name:<16} {w.mean().item():>10.6f} {w.std().item():>10.6f} "
          f"{rms:>10.6f} {w.min().item():>10.6f} {w.max().item():>10.6f}")

# 7. Dead ReLU bias check
if biases is not None:
    dead = (biases < -1e-2).sum().item()
    print(f"\n--- Conv bias trap check ---")
    print(f"Biases < -0.01: {dead} / {len(biases)}")

# 8. CNN32Expert attention gammas
gamma_keys = {
    'self_attn':  'image_conv.self_attn.gamma',
    'cross_attn': 'image_conv.cross_attn.gamma',
    'film':       'image_conv.film.gamma',
}
if any(k in pretrained_dict for k in gamma_keys.values()):
    print(f"\n--- CNN32Expert attention gammas ---")
    for name, key in gamma_keys.items():
        if key in pretrained_dict:
            print(f"  {name:12s}: {pretrained_dict[key].item():.8f}")

# 9. Scalar encoder weight stats (scalars_enc.0.weight shape [64, n_scalars])
scalar_key = 'scalars_enc.0.weight'
if scalar_key in pretrained_dict:
    sw = pretrained_dict[scalar_key]
    n_cols = sw.shape[1]
    print(f"\n--- Scalar encoder '{scalar_key}' shape {list(sw.shape)} ---")
    print(f"{'idx':<5} {'name':<22} {'mean':>10} {'std':>10} {'rms':>10} {'min':>10} {'max':>10}")
    for col in range(n_cols):
        name = SCALAR_NAMES[col] if col < len(SCALAR_NAMES) else f"scalar_{col}"
        c = sw[:, col]
        rms = c.pow(2).mean().sqrt().item()
        print(f"{col:<5} {name:<22} {c.mean().item():>10.6f} {c.std().item():>10.6f} "
              f"{rms:>10.6f} {c.min().item():>10.6f} {c.max().item():>10.6f}")
else:
    print(f"\n[scalars_enc.0.weight not found in checkpoint]")
    scalar_keys = [k for k in pretrained_dict if 'scalar' in k.lower()]
    if scalar_keys:
        print(f"  Scalar-related keys: {scalar_keys}")

# 10. Scalar encoder SVD: effective rank
if scalar_key in pretrained_dict:
    W = pretrained_dict[scalar_key].detach().numpy()   # [out, in]
    out_dim, in_dim = W.shape
    sv = np.linalg.svd(W, compute_uv=False)            # min(out,in) values, descending
    sv_norm = sv / (sv.sum() + 1e-12)
    eff_rank = float(np.exp(-np.sum(sv_norm * np.log(sv_norm + 1e-12))))

    print(f"\n--- Scalar encoder SVD ({in_dim} -> {out_dim}) ---")
    print(f"Effective rank: {eff_rank:.2f} / {in_dim}  "
          f"(1 = rank-1 collapse, {in_dim} = full rank)")
    print("Singular values (desc):")
    for row in range(0, len(sv), 10):
        chunk = sv[row:row + 10]
        idx_str  = "  ".join(f"{row+i:>2}" for i in range(len(chunk)))
        val_str  = "  ".join(f"{s:6.3f}"   for s in chunk)
        print(f"  [{idx_str}]")
        print(f"   {val_str}")

# 10b. Automap CNN
# automap_enc.network.8 is the final Linear(64, output_dim):
#   output_dim = 64 -> fed into LSTM input (use_memory=True) or concat to embedding (use_memory=False)
AUTOMAP_CHANNEL_NAMES = ["explored", "frontier", "player_pos"]
AUTOMAP_LAYERS = [
    ("conv0",  "automap_enc.network.0.weight", "automap_enc.network.0.bias"),
    ("conv1",  "automap_enc.network.2.weight", "automap_enc.network.2.bias"),
    ("conv2",  "automap_enc.network.4.weight", "automap_enc.network.4.bias"),
    ("linear", "automap_enc.network.8.weight", "automap_enc.network.8.bias"),
]
if "automap_enc.network.0.weight" in pretrained_dict:
    lin_w = pretrained_dict.get("automap_enc.network.8.weight")
    if lin_w is not None:
        out_dim, in_dim = lin_w.shape
        wiring_str = f"  linear {in_dim}->{out_dim} (LSTM input)"
    else:
        wiring_str = ""
    print(f"\n--- Automap branch ---")
    if wiring_str:
        print(wiring_str)

    # Input channel stats on first conv
    key0 = "automap_enc.network.0.weight"
    w0 = pretrained_dict[key0]   # (16, 3, 3, 3)
    hdr = f"\n  {'idx':<5} {'name':<12} {'mean':>10} {'std':>10} {'rms':>10} {'min':>10} {'max':>10}"
    print(f"\n  AutomapCNN conv0 input-channel stats:{hdr}")
    for idx, name in enumerate(AUTOMAP_CHANNEL_NAMES):
        w = w0[:, idx, :, :]
        rms = w.pow(2).mean().sqrt().item()
        print(f"  {idx:<5} {name:<12} {w.mean().item():>10.6f} {w.std().item():>10.6f} "
              f"{rms:>10.6f} {w.min().item():>10.6f} {w.max().item():>10.6f}")

    # Layer-by-layer weight norms
    print(f"\n  AutomapCNN layer norms:")
    for lname, wkey, bkey in AUTOMAP_LAYERS:
        if wkey not in pretrained_dict:
            continue
        w = pretrained_dict[wkey]
        rms = w.pow(2).mean().sqrt().item()
        dead_str = ""
        if bkey in pretrained_dict:
            b = pretrained_dict[bkey]
            dead = (b < -1e-2).sum().item()
            dead_str = f"  bias<-0.01: {dead}/{len(b)}"
        print(f"  {lname:<8} shape={list(w.shape)}  rms={rms:.6f}{dead_str}")

# 11. Scalar encoder dead-neuron estimate (static, no rollout needed)
#
# For each output neuron i: the maximum possible pre-activation is
#   dot(W[i], x_max) where x_max clips negative weights to 0 (inputs >= 0)
#   or sum(|W[i]|) for inputs in [-1,1].
# If bias[i] < -max_preact[i], the neuron can never fire -> hard dead.
# If bias[i] / ||W[i]|| < -1, same thing for unit-sphere inputs -> soft dead.
bias_key_enc = 'scalars_enc.0.bias'
if scalar_key in pretrained_dict and bias_key_enc in pretrained_dict:
    W  = pretrained_dict[scalar_key].detach().numpy()       # [out, in]
    b  = pretrained_dict[bias_key_enc].detach().numpy()     # [out]
    out_dim = W.shape[0]

    w_norms    = np.linalg.norm(W, axis=1)                  # ||W[i,:]||
    ratio      = b / (w_norms + 1e-12)                      # b / ||w||

    # Hard bound: inputs in [0,1] -> max pre-act = sum of positive weights
    max_preact = np.sum(np.maximum(W, 0), axis=1)
    hard_dead  = (b < -max_preact).sum()

    # Soft heuristic: ratio < -1 (unit-sphere bound)
    soft_dead  = (ratio < -1.0).sum()

    print(f"\n--- Scalar encoder dead-neuron estimate (static) ---")
    print(f"Hard dead  (b < -sum(W+)):      {hard_dead:3d} / {out_dim}")
    print(f"Soft dead  (b / ||w|| < -1.0):  {soft_dead:3d} / {out_dim}")
    print(f"bias/||w|| percentiles  "
          f"p5={np.percentile(ratio,5):.3f}  "
          f"p25={np.percentile(ratio,25):.3f}  "
          f"p50={np.percentile(ratio,50):.3f}  "
          f"p75={np.percentile(ratio,75):.3f}  "
          f"p95={np.percentile(ratio,95):.3f}")
