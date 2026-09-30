# DevilutionX-AI

A hybrid agent for *Diablo* that combines a reinforcement learning model with an
algorithmic supervisor, playing Ironman - one life, no town returns, 16 dungeon levels
in sequence. Eval results (no stat inflation, no cheats):

```
runs=97  min=2  max=9  mean=5.49  std=1.67

Survival (fraction of runs reaching at least level N):

  level  1: 100.0%  ||||||||||||||||||||||||||||||||||||||||||||||||||||
  level  2: 100.0%  ||||||||||||||||||||||||||||||||||||||||||||||||||||
  level  3:  95.9%  |||||||||||||||||||||||||||||||||||||||||||||||||||
  level  4:  89.7%  ||||||||||||||||||||||||||||||||||||||||||||||||
  level  5:  70.1%  |||||||||||||||||||||||||||||||||||||||||
  level  6:  50.5%  ||||||||||||||||||||||||||||||
  level  7:  27.8%  |||||||||||||||||
  level  8:  12.4%  |||||||
  level  9:   3.1%  ||

Death level histogram:

  level  2:   4 (  4.1%)  ####
  level  3:   6 (  6.2%)  ######
  level  4:  19 ( 19.6%)  ###################
  level  5:  19 ( 19.6%)  ###################
  level  6:  22 ( 22.7%)  ######################
  level  7:  15 ( 15.5%)  ###############
  level  8:   9 (  9.3%)  #########
  level  9:   3 (  3.1%)  ###
```

Diablo himself is not yet dead - but the first half of the game is nailed. Interested? Read on.

## What This Is

A [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based framework for
training reinforcement learning agents in *Diablo*, running on
[DevilutionX](https://github.com/diasurgical/DevilutionX/), an open-source port of the game.
The agent plays the Warrior class - spells are suppressed because they degrade Warrior performance.

The chosen training approach is to work directly on the internal game state rather than
on screenshots or pixels. The observation space is a structured representation of what the
Diablo engine itself tracks: tile properties, monster attributes, player stats. This is not
entirely human-like, but it is far less resource-intensive than pixel-based training and
makes it straightforward to encode exactly what information matters for decision-making.

### Game State Extraction

Game state is extracted from the running `DevilutionX` engine through a shared memory
region - a large blob that the Python agent accesses via `mmap`. The engine writes dungeon
layout, monster positions and attributes, player state, and event data into this region each
game tick. All actions are keyboard inputs that the agent sends back to the engine through a
ring buffer in the same shared memory. This architecture means the agent and the engine run
as separate processes with no modifications to the core game loop beyond what is needed to
expose the state and accept inputs.

### Headless Mode

`DevilutionX` supports a headless mode that runs the game without displaying graphics.
For training this is the primary mode: dozens of game instances run simultaneously,
each driven by its own environment runner, and their collected experience is fed to the
optimizer in parallel. For evaluation the agent can run headless as well, but it is also
possible to attach a GUI session to a running headless instance and watch the hero play
in real time - the engine state is shared, so both views reflect the same game.

## Ironman Rules

Ironman is the natural mode for an agent. No town returns means no need to write
town navigation code - the agent just descends. No second life means success and
failure are unambiguous. And frankly, Ironman sounds a lot cooler than "single-level
benchmark with resets." The agent runs from dungeon level 1 with no assistance:

- `--no-quest`: quests are disabled. The Butcher and other quest-triggered content require
  significant algorithmic handling that is not the focus of this project.
- Town is bypassed entirely. The hero starts at level 1 and never returns.
- Gear dropped on the floor stays there. The agent does not revisit specific items.
- The agent does revisit rooms within a level freely - there is no mechanism to prevent
  that. It explores until the per-level step budget (5000 steps) is exhausted or until
  all monsters are killed (the latter rarely happens in practice).
- One life. Death ends the run.

## Agent

The agent is a hybrid of an algorithmic supervisor and a trained RL model. The design
philosophy: anything that *can* be algorithmized *should* be algorithmized. Only the
genuinely hard problem - how to fight monsters and explore efficiently - goes to the
RL model.

Inventory management is pure algorithm: compare item effectiveness scores, equip the
better one, drop the rest. Stat allocation is a fixed strategy: dex-rush is known to
be optimal for Warrior in the early game, no need to learn it. Navigation to the stairs
at end of floor is BFS: the position is known, the path is computable. None of this needs
a neural network.

What the neural network does handle: which direction to move when monsters are nearby,
when to attack vs retreat, when to use a potion, which tiles to explore next. These
decisions depend on context that is hard to hand-code and easy to learn from experience.

### Algorithmic Supervisor

The agent is a state machine that wraps the RL model and handles everything
the model should not have to learn:

- **DUNGEON**: hands control to the RL model. This is the default state on every floor.
- **PATHFIND**: fires when the per-level step budget is exhausted or the kill threshold
  is met. Algorithmically walks to the last sighted stairs using BFS - only if stairs
  were discovered during DUNGEON; if not, the run fails.
- **DEAD / DONE**: terminal states.

Town navigation is not needed - the run starts at level 1 directly.

Additional logic that runs on every step regardless of state:

- **Gear management**: evaluates new inventory items against per-level expected monster
  stats, equips upgrades, drops junk.
- **Stat allocation**: each level-up distributes attribute points automatically.
  The current strategy (`--stat-strategy dex-rush`) front-loads Dexterity to maximize
  hit chance early, when enemy armor class is still low. In the original game this is
  a known player-optimal path for Warrior: Dex scales attack rating, which directly
  determines whether hits land.
- **Repair**: repairs equipped gear when durability drops below 25%, only when no
  monsters are within 2 tiles.
- **Warp masking**: staircase triggers (ascend / descend) are masked as walls in the
  model's observation to prevent accidental level transitions before the kill threshold
  is met.

The gear management logic is visible in the agent log:

```
agent 26004: inv[+] cii=8 seed=1363456912 slot=Weapon [dmg=6-15] 'Bastard Sword'
agent 26004: queue equip 'Bastard Sword' seed=1363456912 [eff=20.03] over 'Broad Sword' [eff=17.04]
agent 26005: equip 'Bastard Sword' seed=1363456912
agent 26006: equipped 'Bastard Sword' seed=1363456912 lvl=6 gear: slots=7 clvl=12 hp=86/141 str=40(+12) dex=60(+0) mag=9(+0) vit=31(+9) ac=21 dmin=6 dmax=15
agent 26006: queue drop 'Broad Sword' seed=1296653445 [eff=17.04] - worse than 'Bastard Sword' [eff=20.03]
agent 26007: drop 'Broad Sword' seed=1296653445
agent 26008: masked drop 'Broad Sword' seed=1296653445 at (56, 71)
agent 26045: queue drop gold seed=150527310 val=41 - excess gold piles (>0)
agent 26047: queue drop 'Robe' seed=404106678 [eff=18.49] - worse than 'Quilted Armor' [eff=20.03]
agent 26049: masked drop 'Robe' seed=404106678 at (54, 69)
```

`eff` is an effectiveness score computed from the item's stats against expected
monsters for the current floor. Items the agent decides to drop are masked in
the model's observation immediately, so the model does not try to pick them up again.

### RL Model

Architecture: `CNN32Expert` + LSTM, 20 discrete actions.

The RL model drives all dungeon combat and exploration decisions. Different model
checkpoints are used per level band. The `--model` flag accepts a comma-separated
list of `RANGE=NAME` pairs, where `RANGE` is either a single level or a `LOW-HIGH`
span (with `*` meaning "to the end"), and `NAME` is a model directory under
`ai/models/`. For example:

```
./diablo-ai.py agent-ai \
   --embedding-dim 512 --cnn-arch cnn32expert --best-train \
   --model 1-4=ClearAllLevels-1-4,5-8=ClearAllLevels-5-8,9-*=ClearAllLevels-1-4 \
   --env Diablo-ClearAllLevels-v17 --dungeon-level 1 --seeds 1-100 \
   --max-steps-per-level 5000 --kill-threshold 1 --stat-strategy dex-rush
```

The `9-*` band reuses the L1-4 model: the agent reaches L9+ in roughly 3% of runs,
which is not enough to justify a dedicated training run. `ClearAllLevels-5-8` shows
solid performance for the mid-game; L1-4 is a reasonable fallback for anything deeper.

The model switches checkpoint at each level transition; the LSTM state resets.

## Observation Space

The agent observes only a local window of the dungeon - a 21x21 tile region centered on
the hero, covering a radius of 10 cells. This mirrors how a human plays: most of the map
is unknown at any given moment and must be explored.

The observation is a `gym.spaces.Dict` with three keys.

### `env` - (21, 21, 19) uint32

Each tile in the local view is encoded as a set of flags representing its properties:
whether it contains the player, a monster, a wall, a door, a chest, an item, and so on,
as well as whether the tile has been explored or is currently in line-of-sight. Rather
than passing this bitfield directly, the environment exposes each flag as a separate
channel, giving the model a clean one-hot representation. The result is a
`21 x 21 x 19` array where each of the 19 channels corresponds to one tile property:

| Channel | Flag | Description |
|---------|------|-------------|
| 0 | Player | Hero position |
| 1 | Wall | Solid obstacle |
| 2 | PrevTrigger | Staircase to previous level / town |
| 3 | NextTrigger | Staircase to next level |
| 4 | WarpTrigger | Warp portal |
| 5 | Door | Door tile |
| 6 | Missile | Projectile in flight |
| 7 | Monster | Monster present |
| 8 | UnknownObject | Misc interactive object |
| 9 | Crucifix | Crucifix |
| 10 | Barrel | Barrel / pod / urn |
| 11 | Chest | Chest |
| 12 | Sarcophagus | Sarcophagus |
| 13 | Item | Dropped item |
| 14 | Explored | Tile has been visited |
| 15 | Visible | Tile is in current line-of-sight |
| 16 | Interactable | Object can be interacted with |
| 17 | Open | Door is open |
| 18 | Goal | Episode goal tile (unused in current env) |

### `monster_attrs` - (21, 21, 9) float32

Per-tile monster attributes, filled only for visible tiles. Nine normalized channels:

| Index | Attribute |
|-------|-----------|
| 0 | HP ratio (current / max) |
| 1 | Monster level / max monster level |
| 2 | Is unique (0 / 1) |
| 3 | Walk speed score (higher = faster) |
| 4 | Attack speed score (higher = faster) |
| 5 | Fire resistance (0 / 0.5 / 1.0) |
| 6 | Lightning resistance |
| 7 | Magic resistance |
| 8 | Is ranged attacker (0 / 1) |

### `scalars` - 46 float32

Flat vector covering player and episode state:

- Dungeon level / 16, character level / 50
- HP ratio, mana ratio
- Hero direction (0-7)
- Str / magic / dex / vit normalized by class cap
- Weapon damage min/max, armor class, fire/lightning/magic resistances
- Mana shield active
- Potion counts (small HP, full HP, scroll heal, small mana, full mana, rejuv, full rejuv)
- Spell availability bits and spell levels (7 spells; included in obs but unused for Warrior)

## Action Space

20 discrete actions:

- Walk N / NE / E / SE / S / SW / W / NW (8)
- Stand
- Primary action (attack monster, interact with towner, lift/place item)
- Secondary action (open chest / door, pick up item)
- Restore HP (uses best available HP potion)
- Restore mana (uses best available mana potion)
- Cast Firebolt / Charged Bolt / Firewall / Stone Curse / Mana Shield / Phasing / Fireball (7)

The 7 spell actions are part of the action space but never trained for Warrior -
the hero loadout never includes spells, so the model never sees a spell succeed.

## Reward Function

Current version: `Diablo-ClearAllLevels-v17`. The reward function is continuously
revised as training progresses; this version is current but not final.

**Terminal:**

| Event | Reward |
|-------|--------|
| Death | -10.0 |
| Diablo killed | +20.0 |
| Level cleared (threshold / budget met) | +20.0 |
| Escape to previous level / town | -10.0 |
| Stuck (no progress for 300 steps) | -10.0 |
| Timeout (3000 steps) | -10.0 |

**Per-step shaping:**

| Event | Reward |
|-------|--------|
| New tile explored | +0.05 |
| Wasted step (no combat, no new tiles) | -0.025 |
| Attack monster (per target hit) | +0.02 |
| Kill monster | +0.10 |
| Spell successful | +0.15 |
| Spell first use (per type per episode) | +0.10 |
| Open door (no monsters visible) | +0.02 |
| Activate object | +0.05 |
| Collect item | +0.02 |
| HP potion used correctly (HP < 90%) | +0.05 |
| HP potion wasteful / no potion | -0.10 |
| Mana potion used correctly | +0.05 |
| Mana potion wasteful / no potion | -0.10 |
| Primary action with no target | -0.05 |
| Secondary action with no target | -0.05 |
| Spell unavailable / wasteful | -0.10 |

The movement penalty (-0.025) and exploration reward (+0.05) are calibrated so that
a step to a new tile has positive expected value, discouraging the agent from standing still.

The reward function is a long history of trial and error. The principle is simple: give a
signal for every meaningful action so the model is never flying blind. Each entry in the
table above was added because the agent was doing something wrong - standing still, spamming
actions with no target, ignoring potions, walking past enemies. The current v17 is the
result of iterating on observable failures; it is not final.

## Training

Training is intentionally separated from agent eval and is not directly comparable to
ironman performance.

### How the current model came to be

Pure reinforcement learning from scratch failed to make progress on exploration. The
solution was to bootstrap with imitation learning: an algorithmic bot collected 50k
demonstration episodes, and the agent was trained to imitate it for 150M frames. This
gives the agent a navigation foundation before any RL starts.

After imitation learning the policy is reasonable but the critic (value function) is
essentially uninitialized. Starting PPO at this point causes catastrophic forgetting
within a few updates - the critic's poor estimates generate bad gradients that overwrite
everything the agent just learned. The fix is to train the critic in isolation first,
then bring the policy back in gradually.

Architecture was also a blocker. When standing monsters were introduced the agent simply
ignored them and performance stayed flat. Switching to the `CNN32Expert` architecture -
which adds self-attention over the spatial map and FiLM conditioning that modulates
spatial features based on the LSTM memory - unblocked learning. The agent started
engaging monsters and navigating around them instead of ignoring them.

### Setup

The hero is dropped into a single dungeon level with stats artificially bumped by
`scripts/diablo-sim.py` to match expected character progression at that level. Without
this, the hero dies within seconds on hard levels before the model can learn anything.

Models are trained per level chapter rather than across all 16 levels at once - chapter-specific
training consistently outperforms uniform training. Training success rate is roughly 0.7 on
average across all 16 levels, but this number reflects artificial conditions: boosted stats,
single-level episodes, no gear accumulation. It does not translate directly to ironman depth.

Current training command:

```shell
./diablo-ai.py train-ai \
   --no-butcher --no-spells \
   --hero-hp-at-start 0.4-1 \
   --hero-mana-at-start 0.4-1 \
   --hero-potions-at-start 0-12 \
   --dungeon-level 1-16 \
   --cnn-arch cnn32expert \
   --embedding-dim 512 \
   --env Diablo-ClearAllLevels-v17 \
   --gpus 2 --env-runners 256 \
   --frames +100M \
   --batch-size 40960 \
   --frames-per-env-runner 320 \
   --lr 0.0001 \
   --entropy-coef 0.015 \
   --recurrence 160 \
   --eval-episodes 250 \
   --eval-dungeon-level 1-16 \
   --model $MODEL
```

Key parameters:

- `--env Diablo-ClearAllLevels-v17` - environment the agent trains in. The task is to
  clear dungeon floors by killing monsters and finding stairs, repeating across all 16 levels.
- `--dungeon-level 1-16` - dungeon levels sampled during training. Models can also be
  trained per chapter (L1-4, L5-8, etc.) for better per-chapter performance.
- `--cnn-arch cnn32expert` - CNN architecture with self-attention and FiLM conditioning.
  Self-attention lets the model reason about spatial relationships across the full 21x21 view.
  FiLM conditions the spatial features on the LSTM memory, so the model can interpret the
  same tile differently depending on what it has seen earlier in the episode.
- `--embedding-dim 512` - size of the latent embedding produced by the CNN, fed into the LSTM.
- `--recurrence 160` - length of temporal sequences used for LSTM training (BPTT window).
- `--frames +100M` - total frame budget. The `+` prefix means "add 100M to the current
  frame count", so the command can be re-run as-is to extend training incrementally.
- `--env-runners 256` - parallel game instances collecting experience simultaneously.
- `--frames-per-env-runner 320` - steps each runner collects before sending data to the optimizer.
- `--batch-size 40960` - number of steps per gradient update.
- `--entropy-coef 0.015` - weight of the entropy regularization term. Too low and the agent
  collapses to a single strategy and stops exploring; too high and it acts randomly.
- `--no-butcher --no-spells` - disable quest content and suppress spell drops for Warrior.

## Sprout: Model Version Control

Managing many training runs with different hyperparameters quickly becomes chaotic.
Sprout treats model checkpoints like a version control system. Each training run is a node
in a tree that versions everything - all model files, parameters, training metrics, and eval
results. By default it shows only which parameters changed from the parent, making it easy to spot what
was different between runs.

It is also a sanity tool. When you have hundreds of runs it becomes impossible to remember
what you tried, what helped, and what the command line for a given checkpoint was. Sprout
solves this: every run is fully reproducible from its stored parameters, the tree shows
the full experiment history at a glance, and the last training and eval stats are always
attached. Browsing the history to understand why one run outperformed another takes seconds
instead of digging through log files. Each run can also carry a short alias (e.g. "BEST TO
CONTINUE" or "wrong params, RM ASAP") and a longer description note - useful when you need
to leave instructions for yourself or flag a run for follow-up.

Key commands (all require a model name as the working target):

```shell
# Show full training history tree
./diablo-ai.py sprout tree

# Show the exact command line used for a specific run
./diablo-ai.py sprout show --run <RUNID>

# Move the active head to any past run and resume training from there
./diablo-ai.py sprout switch <HEAD> <TO_RUN>

# Roll back the head to its parent state
# --persist saves the current state as a branch before rewinding
./diablo-ai.py sprout rewind <HEAD> [--persist]
```

`sprout switch` and `sprout rewind` make it practical to try risky experiments -
architecture changes, direct weight surgery, reward reshaping - without losing a good
checkpoint. Branching from any run is a single command.

## Docker

Two images are available on [Docker Hub](https://hub.docker.com/r/romanpen/):

**Full environment** - `romanpen/devilutionx-ai-ubuntu24.04`

Includes CUDA 12.9, build tools, compiled DevilutionX binary, Diablo Shareware asset,
and a pre-configured Python virtualenv. This is the image for training and evaluation.

NVIDIA Container Toolkit must be installed first. See the
[NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```shell
docker run \
   --runtime=nvidia --gpus all \
   -dit \
   --name devilutionx-ai \
   romanpen/devilutionx-ai-ubuntu24.04:latest
```

For GUI evaluation (agent running with graphics):

```shell
xhost +local:root

docker run \
   --runtime=nvidia --gpus all \
   -dit \
   --name devilutionx-ai \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   romanpen/devilutionx-ai-ubuntu24.04:latest
```

Both commands start the container in the background with a `tmux` session:

```shell
docker exec -it devilutionx-ai tmux -u attach
```

**Model checkpoints only** - `romanpen/devilutionx-ai-models`

A minimal image (`FROM scratch`) containing only the trained model files.
Useful for restoring checkpoints without pulling the full environment:

```shell
docker create --name tmp romanpen/devilutionx-ai-models:latest
docker cp tmp:/models/. ai/models/
docker rm tmp
```

## Building and Running

Build the DevilutionX binary:

```shell
cmake -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=OFF \
    -DDEBUG=ON \
    -DUSE_SDL1=OFF \
    -DHAS_KBCTRL=1 \
    -DPREFILL_PLAYER_NAME=ON \
    \
    -DKBCTRL_BUTTON_DPAD_LEFT=SDLK_LEFT \
    -DKBCTRL_BUTTON_DPAD_RIGHT=SDLK_RIGHT \
    -DKBCTRL_BUTTON_DPAD_UP=SDLK_UP \
    -DKBCTRL_BUTTON_DPAD_DOWN=SDLK_DOWN \
    -DKBCTRL_BUTTON_X=SDLK_y \
    -DKBCTRL_BUTTON_Y=SDLK_x \
    -DKBCTRL_BUTTON_B=SDLK_a \
    -DKBCTRL_BUTTON_A=SDLK_b \
    -DKBCTRL_BUTTON_RIGHTSHOULDER=SDLK_RIGHTBRACKET \
    -DKBCTRL_BUTTON_LEFTSHOULDER=SDLK_LEFTBRACKET \
    -DKBCTRL_BUTTON_LEFTSTICK=SDLK_TAB \
    -DKBCTRL_BUTTON_START=SDLK_RETURN \
    -DKBCTRL_BUTTON_BACK=SDLK_LSHIFT

make -C build -j$(nproc)
```

Download the Diablo Shareware asset:

```shell
wget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq -P build
```

Set up the Python environment:

```shell
cd ai
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Run the game in headless TUI mode:

```shell
./diablo-ai.py play
```

Attach a TUI to a running game instance (including a GUI session):

```shell
./diablo-ai.py play --attach 0
```

List all running instances:

```shell
./diablo-ai.py list
```

## DevilutionX Engine

The DevilutionX engine source is modified from the upstream project: several bugs fixed,
engine optimized for parallel headless training, `--no-quest` mode added, and artificial
stat injection added to support curriculum training. None of these modifications affect
the agent eval path - eval runs the game as close to the original as `--no-quest` allows,
with no stat cheating.

## Status and Contributing

The agent currently reaches level 9 in roughly 3% of ironman runs and dies somewhere
between levels 4 and 7 in most of them. Diablo on level 16 is still very much alive.
The model keeps improving - better reward shaping, better curriculum, better architecture
choices all move the survival curve to the right.

If this interests you: the framework is fully open, the training code runs in Docker,
and there is plenty of unsolved ground. PRs, ideas, and experiments welcome.
