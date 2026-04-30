<div style="width:100%;">
  <a href="https://youtu.be/6KuSlT9EOec" target="_blank">
    <img src="https://github.com/user-attachments/assets/85967261-4fe9-43db-86dd-1c20c8a641d4" style="width:100%; height:auto;" />
  </a>
</div>

# What is DevilutionX-AI

`DevilutionX-AI` is a
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) - based
framework for training reinforcement learning (RL) agents in the game
*Diablo*. The game runs on
[DevilutionX](https://github.com/diasurgical/DevilutionX/), an
open-source port of *Diablo*, with some extra patches added to make it
usable for RL.

The framework includes a Gymnasium environment,
[patches](#devilutionx-patches) for DevilutionX, a runner, and a
training pipeline. The PPO training pipeline is built on
[torch_ac](https://github.com/lcswillems/torch-ac), with imitation
learning components adapted from the [BabyAI
project](https://github.com/mila-iqia/babyai).

The goal is to train an agent (the Warrior) to clear the first dungeon
level. That means exploring the dungeon, fighting monsters, picking up
items, opening chests, activating other objects, and finding the stairs
to the next level - basically what a human would do when just starting
the game.

The short video at the top of this README demonstrates the agent
fighting monsters and clearing a randomly generated dungeon level
(more on replicating the results below) with the pre-trained model,
which achieved a success rate of 0.98 during evaluation.

This project is not about training an agent to beat the entire
game. At first, I just wanted to see "signs of life": an RL agent that
can explore the first dungeon level without worrying about more
complex behaviors like going back to town, casting spells, or swapping
gear.

I am not an RL expert, and AI is not part of my daily work, so I
started with a small and simple goal. Hopefully the framework can be
useful to others with more RL experience. Maybe together we will see
an agent one day that plays *Diablo* in a way that looks a lot like a
human.

## Results

Training progressed through three stages, each building on the previous one.

**Stage 1: Finding the stairs (monsters disabled)**

The first goal was simple: train the agent to find the stairs to the
next dungeon level with all monsters disabled. Despite the apparent
simplicity, the agent had to explore a large partially-observable
dungeon without any map.

The agent reached a **0.967 success rate** and showed some unexpected
behavior: it learned to exploit structural regularities in the dungeon
generator, since stairs are not placed entirely at random -- they tend
to appear in larger halls. It also learned to backtrack when a path
leads nowhere, which gives the impression of episodic memory, even
though the agent only has a local view and a recurrent state.

**Stage 2: Finding a random goal (monsters still disabled)**

The next task was harder: find a truly random goal placed anywhere in
the dungeon. Unlike stairs, random goals have no spatial bias, so the
agent had to develop systematic exploration rather than exploiting
structural patterns.

Pure reinforcement learning from scratch failed to make progress. The
solution was a multi-phase training pipeline: first bootstrap the
agent with imitation learning from a scripted bot, then carefully
warm up the critic before switching to PPO. Starting PPO directly
after imitation learning with an uninitialized critic causes
catastrophic forgetting in just a few updates -- the agent quickly
forgets everything it learned. The warm-up step provides a stable
bridge.

The agent reached a **0.97 success rate** on finding a randomly placed
goal.

**Stage 3: Clearing the level (monsters enabled)**

Enabling monsters revealed a new problem: the agent completely ignored
them. Switching to a more expressive CNN architecture (CNN32Expert),
which adds self-attention and FiLM conditioning on the agent's memory,
unblocked learning and the agent quickly started engaging with
monsters.

Ablation experiments on 3000 episodes confirmed that FiLM is the
load-bearing component -- zeroing it drops success rate from 0.98 to
0.91 -- while self-attention and cross-attention contribute
marginally. All three blocks are kept in the architecture.

The current model achieves a **0.98 success rate** on 3000 randomly
generated dungeon levels.

## Docker Container

A prebuilt docker image is available on [Docker Hub](https://hub.docker.com/r/romanpen/devilutionx-ai-ubuntu24.04).

First, the NVIDIA Container Toolkit must be installed. For a detailed guide, please follow the [NVIDIA instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

As described by [NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html), you can run the image with CUDA support as follows:

```
docker run \
   --runtime=nvidia --gpus all \
   -dit \
   --name devilutionx-ai \
   romanpen/devilutionx-ai-ubuntu24.04:latest
```

If the X11 application (Diablo GUI) needs to be executed from Docker
(e.g., when the model is evaluated in graphics mode), the X11 socket
must be shared with Docker using the following command:

```
# Let root inside Docker connect to your X session
xhost +local:root

# Run docker with a shared X11 socket
docker run \
   --runtime=nvidia --gpus all \
   -dit \
   --name devilutionx-ai \
   -e DISPLAY=$DISPLAY \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   romanpen/devilutionx-ai-ubuntu24.04:latest
```

Previous `docker run` commands start the container in the background
with a default `tmux` session available for attaching. To attach to
the `tmux` session, please execute:

```
docker exec -it devilutionx-ai tmux -u attach
```

## Training Peculiarities

The chosen training method is the least resource-intensive: training
on the internal state of the game rather than on screenshots and
pixels. This means the observation space is represented as a
two-dimensional matrix of the dungeon (see details about the
[observation space](#observation-space) below), which is the
structured game state the Diablo engine itself uses. Although this
approach is not entirely human-like, it allows you to save
computational or RAM resources and quickly adapt the training
strategy. Having trained on structured data, in the future it is
possible to separately train another CNN-based layer, which will be
able to represent screenshots of the game in the same structured
state.

### Game State Extraction

For reinforcement learning training purposes, data from the
DevilutionX engine implementation is extracted as a two-dimensional
21x21 array representation of a section of a dungeon. This array
represents the agent's view, which covers a radius of 10 cells
surrounding the agent. Additionally, descriptor arrays for dungeon
objects, states for non-player characters, various counters, and the
player's state (including hit points, current dungeon level, position
in the dungeon, status, etc.) are included. All state structures are
shared by the engine through a memory file, a large blob which the AI
agent can access using Linux APIs such as `mmap`. All actions are
keyboard presses that the agent sends to the game engine through a
ring buffer and the same shared memory. To get everything working, it
was necessary to make a set of [changes](#devilutionx-patches) to the
original `DevilutionX` project.

## Observation Space

The observation space in reinforcement learning represents the domain
of various experiments, trials, and errors. Currently, a radius of 10
cells around the agent is observed by the RL agent. This means the
agent sees only part of the whole dungeon, similar to how a human
would play the game.

Each tile in the two-dimensional dungeon map is encoded as a set of
bits, where each bit denotes a specific property of the tile. These
properties include, for example, the presence of the player, a
monster, a wall, a closed or open door, a chest, an item, as well as
whether the tile has been explored or is currently visible to the
player. Instead of passing this bitset directly, the environment
provides the agent with a one-hot style representation: each bit is
exposed as a separate channel. As a result, the agent's observation
takes the form of a three-dimensional array of size `21 × 21 × N`,
where `N` equals the total number of encoded tile properties.

## Action Space

The choice of action space is simpler: the player can stand still
or move in eight cardinal directions: north, northeast, east,
southeast, south, southwest, west, and northwest. Additionally, the
player can perform exactly two types of actions: primary and secondary
action, where primary action includes attacking monsters, talking to
towners, lifting and placing inventory items. Meanwhile, a secondary
action involves opening chests, interacting with doors, and picking up
items.

Since there are only 11 possible discrete actions, the action space is
defined using `gym.spaces.Discrete` type.

## Reward Function

The reward function guides the agent toward clearing the dungeon level
while surviving combat:

**Terminal rewards**:

- **Death** - penalty (-10), episode ends.

- **Escaping back to town** - neutral (0), episode ends.

- **Reaching the goal** - strong reward (+20), episode ends.

**Shaping rewards**:

- **Damage taken** - penalty proportional to health lost (scaled by max HP).

- **Attacking a monster** - small reward (+0.02) for dealing damage.

- **Killing a monster** - reward (+0.1) per kill.

- **Unproductive movement** - small penalty (-0.01) for moving without
  any combat or progress, to discourage aimless wandering.

- **Getting stuck** - early truncation with no penalty if the agent
  repeats useless actions or times out.

## Headless Mode

`DevilutionX` already supports a `headless` mode, which allows the
game to run without displaying graphics. For RL training, this is the
primary mode because dozens of game instances and envrionemt runners
can be run simultaneously, and states from each is collected for
training in parallel. While evaluating (when a pre-trained AI agent
interacts with the Diablo environment without further learning), it is
possible to attach to the game with a graphics session and have the
player navigate the dungeon according to the trained strategy.

## Agent Training

### Training Pipeline

Training the agent to clear the level required several stages rather
than a single reinforcement learning run.

**Stage 1: Imitation learning bootstrap (no monsters)**

An algorithmic bot that knows how to explore the dungeon was used to
collect 50k demonstration episodes. The agent was then trained to
imitate the bot's behavior for 150M frames, reaching 0.95 action
accuracy. This gives the agent a solid navigation foundation before
any RL starts.

After imitation learning, the policy is well-formed but the critic
(value function) is essentially uninitialized. Starting PPO at this
point immediately destabilizes learning: the critic's poor estimates
produce bad gradient updates that overwrite the policy in just a few
steps. To avoid this, the critic is trained in isolation for 50M
frames, then jointly with the policy for another 100M frames.

PPO fine-tuning in the same no-monsters environment then brought the
agent to a **0.97 success rate** on finding a randomly placed goal.

**Stage 2: Standing still monsters, new architecture**

Introducing standing non-attacking monsters had no effect: the agent
simply ignored them and performance stayed flat. Switching to the
CNN32Expert architecture -- adding self-attention and FiLM
conditioning on the agent's memory -- unblocked progress. The agent
started navigating around standing monsters and occasionally engaging
them when they blocked the path.

**Stage 3: Moving and attacking monsters (invincible player)**

With the player made invincible, monsters were enabled with full
movement and attacks. The agent reached **>0.9 success rate** in
roughly 50M frames, learning to navigate a dungeon full of actively
pursuing monsters.

**Stage 4: Full combat with damage**

Enabling monster damage and shaping the reward function around
combat produced a brief drop from 0.9 to 0.6, but the agent
recovered quickly -- faster than expected. It developed strategies
for killing monsters and avoiding damage on its own, eventually
reaching the current **0.98 success rate** on 3000 randomly generated
dungeon levels.

### Training Command

Choosing the right parameters and their combinations for effective RL
training is an art and essentially a path of endless trial and
error. For example, I use the following command line:

```shell
./diablo-ai.py train-ai \
   --harmless-barrels \
   --cnn-arch cnn32expert \
   --embedding-dim 512 \
   --env Diablo-ClearTheLevel-v0 \
   --env-runners 256 \
   --frames 100M \
   --batch-size 40960 \
   --frames-per-env-runner 320 \
   --lr 0.0001 \
   --entropy-coef 0.001 \
   --recurrence 160 \
   --eval-episodes 250 \
   --model Diablo-ClearTheLevel-v0
```

Where:

- `--env Diablo-ClearTheLevel-v0` - The environment the agent
  interacts with. The task is to explore the dungeon, fight monsters,
  and find the goal.

- `--model Diablo-ClearTheLevel-v0` - Name of the model used for
  training. Essentially, it's a folder where the model files are
  located.

- `--cnn-arch cnn32expert` - The convolutional neural network
  architecture used to process observations. The `cnn32expert` variant
  extends the base CNN with self-attention (for deeper spatial
  understanding of the dungeon layout) and FiLM conditioning (for
  modulating spatial features based on the agent's memory,
  helping to differentiate between objects depending on current
  context such as combat or exploration). The name reflects iterative
  experimentation with several architectures.

- `--harmless-barrels` - Makes exploding barrels harmless. Since the
  agent cannot use potions to restore health, an accidental barrel
  explosion would end the episode early and obscure the training
  signal.

- `--frames 900M` - Total number of environment frames (steps) the
  agent will be trained on.

- `--frames-per-env-runner 320` - Number of steps each environment
  instance runs before sending data to the optimizer.

- `--env-runners 256` - Number of parallel environment instances used
  for training, allowing faster experience collection.

- `--batch-size 40960` - Number of frames (steps) collected before
  performing a gradient update.

- `--recurrence 160` - Length of temporal sequences used for recurrent
  policy updates (for RNN/LSTM agents, representing a memory).

- `--embedding-dim 512` - Size of the latent embedding vector produced
  by the CNN.

- `--lr 0.0001` - Learning rate for the optimizer.

- `--entropy-coef 0.001` - Weight of the entropy regularization term,
  encouraging exploration.

- `--eval-episodes 250` - Number of episodes used for periodic
  evaluation during training.

Hyperparameters are the subject of many experiments. For example, a
low entropy coefficient can result in a Diablo RL agent getting stuck
in one room without taking any further actions, or wandering from
corner to corner.

This list of game and training parameters used in my experiments is by
no means optimal. I am continually exploring the behavior of an RL
agent and frequently adjust parameters or introduce new ones to
achieve the desired results.

## Agent Evaluation

The video at the very beginning of this README can be replicated with
the following command:

```shell
./diablo-ai.py play-ai \
   --harmless-barrels \
   --cnn-arch cnn32expert \
   --embedding-dim 512 \
   --env Diablo-ClearTheLevel-v0 \
   --env-runners 1 \
   --model Diablo-ClearTheLevel-v0 \
   --seed-base 5 \
   --game-ticks-per-step 12 \
   --gui
```

As soon as the Diablo GUI window appears, select "Single Game" and
proceed with the "Warrior" character, using the default name and
normal difficulty. Once the first level is loaded, the agent resets
the environment a few times and starts exploring the dungeon, fighting
monsters, and searching for the goal. The episode ends when the agent
reaches the goal or gets stuck.

To attach a terminal ASCII representation to the running game
instance, use the following command:

```shell
./diablo-ai.py play --attach 0
```

## Sprout: Model Version Control

Managing dozens of training runs with different hyperparameters,
architectures, and results quickly becomes chaotic. Sprout is a
lightweight tool included in the repository that treats model
checkpoints like a version control system.

Each time training starts, Sprout takes a snapshot of the current
model state. The full training history is stored as a tree where each
node records only the parameters that changed from its parent, along
with training metrics such as success rate and total frames. Returning
to any previous state -- including before a risky surgery or a bad
hyperparameter choice -- is a single command:

```shell
# Show the full training history tree
./diablo-ai.py sprout tree

# Show details for the current head
./diablo-ai.py sprout show --head Diablo-ClearTheLevel-v0

# Jump the active head back to any specific run
./diablo-ai.py sprout switch --head Diablo-ClearTheLevel-v0 --to-run d9ca5ecd

# Undo the last training run and return to the parent state
./diablo-ai.py sprout rewind Diablo-ClearTheLevel-v0

# Branch off a new experiment from the current head
./diablo-ai.py sprout clone --from-head Diablo-ClearTheLevel-v0 Diablo-ClearTheLevel-experiment
```

This made it practical to try experiments such as architecture changes
or direct weight surgery without fear of losing a good checkpoint, and
to compare different training strategies side by side by branching
from the same base run.

The training history for the ClearTheLevel model shows the full
evolution from the cloned FindRandomGoal baseline through monster
introduction and gradual recovery:

```
▶ Diablo-ClearTheLevel-v0
└─ dd6fe9af (CLONED--Diablo-FindRandomGoal-v0--cnn32-best)
   │ ≡ best/success_rate: 0.968
   │ ≡ last/duration: 1d14h
   └─ d9ca5ecd
      │ ⇾ no_monsters: True -> False
      │ ⇾ cnn_arch: cnn32 -> cnn32expert
      │ ⇾ env: Diablo-FindRandomGoal-v0 -> Diablo-ClearTheLevel-v0
      │ ⇾ entropy_coef: 0.01 -> 0.001
      │ ≡ last/success_rate: 0.244
      └─ ...
         └─ c7a414fb
            │ ⇾ invincible_player: False -> True
            │ ⇾ blind_monsters: True -> False
            │ ≡ last/success_rate: 0.780
            └─ 7edc9fad
               │ ⇾ invincible_player: True -> False
               │ ≡ last/success_rate: 0.816
               └─ ...
                  └─ ● Diablo-ClearTheLevel-v0
                       ≡ best/success_rate: 0.968
                       ≡ last/duration: 3d00h
                       ≡ last/success_rate: 0.916
```

Each node shows only the parameters that changed from its parent. The
`●` marker indicates the current active head. The `⇾` prefix marks
parameter changes, `≡` marks recorded metrics.

Sprout is available as `./diablo-ai.py sprout` (which automatically
sets the working directory) or directly as a [single Python
file](ai/sprout.py) with `--working models`.

## Building and Running

The RL training pipeline is written in Python and retrieves
environment states from the running `DevilutionX` game
instance. `DevilutionX` must be compiled, as it is written in
C++. First, build the `DevilutionX` binary in the `build` folder:

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

Once the binary is successfully built, the entry point for all RL
tasks is the `diablo-ai.py` script located in the `ai/` folder. This
script includes everything needed to attach to an existing
`DevilutionX` game instance, run RL training from scratch or evaluate
a pre-trained agent.

Before executing `diablo-ai.py` there are a few things left to be
done: the Shareware original Diablo content should be downloaded and
placed alongside the `devilutionx` binary, i.e., in the `build`
folder:

```shell
wget -nc https://github.com/diasurgical/devilutionx-assets/releases/download/v2/spawn.mpq -P build
```

Once the download is finished, the required Python modules need to be
installed in the `virtualenv` folder which can be named as `myenv`:

```shell
cd ai
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Now, as a hello-world example, the Diablo game can be launched
directly in the terminal in `headless` mode, but with TUI (text-based user
interface) frontend:

```shell
./diablo-ai.py play
```

And the game will look on your terminal as follows:
```
        Diablo ticks:    263; Kills: 000; HP: 4480; Pos: 83:50; State: PM_STAND
                    Animation: ticksPerFrame  1; tickCntOfFrame  0; frames  1; frame  0
                   Total: mons HP 14432, items 4, objs 94, lvl 1 ⠦  . . . . . . ↓ ↓ ↓ ↓






                                                   # #
                                             # # # $ . # # # #
                                     .     # . . . . . . . . . #
                                   . . . . # . . . . . . . . . #
                                   . . . . . . . . . . . . . . #
                                 . . . o . @ @ . . . . . . . . #
                                 . . . . . . . . . . . . . . . #
                                 . . . . . . . . . . ↓ . . . . #
                                 . . . . . . . . . . . . . . . #
                                   # D # # # . . . . . . . . . #
                                           # . . . . . . . . . #
                                             # # . # . # . # #
                                               # .   .   . #
                                               #     C     #
                                               #     .     #
                                                   . . .
                                                   . . .
                                                   C . .

                                           Press 'q' to quit
```

This shows a top-down view of a Diablo dungeon on the level 1 (town is
skipped) where the arrow `↓` in the center represents the player, `#`
represents walls, `.` represents visible part of the dungeon (or the
player vision), `@` represents monsters, `o` represents objects, `C`
represents unopened chests, and so on. TUI mode accepts keyboard input
only: regular arrows for movement and exploring the dungeon, `a` for
the primary action, `x` for the secondary action, `s` for quick save,
`l` for quick load, and `p` for game pause.

A similar text-based output can be achieved by attaching to an
existing game instance, even when graphic session is active in another
window:

```shell
./diablo-ai.py play --attach 0
```

Where `0` represents the first available Diablo instance. A list of
all running instances can be retrieved by calling the

```shell
./diablo-ai.py list
```
## `DevilutionX` Patches

For game state extraction to a third-party application (the RL agent,
specifically `diablo-ai.py`) and submitting keyboard inputs outside
the UI loop, several changes to the original `DevilutionX` were
necessary:

### AI-Oriented Gameplay Changes

- Shared memory implementation for reinforcement learning
  agents. Supports external key inputs and game event monitoring.

- Added a `headless` mode option to start the game in non-windowed
  mode (already supported by the `DevilutionX` engine, but see the
  list of [fixes](#various-fixes) below)

- Added an option to launch the game directly into a specified dungeon
  level.

- Enables deterministic level and player generation for reproducible
  training by setting a seed.

- Added an option to remove all monsters from the dungeon level to
  ease the exploration training task.

- Added an option to skip most animation ticks to accelerate training
  speed.

- Added an option to run the game in step mode, i.e., the game does not
  proceed without a step from an agent (player).

- Added an option to disable monster auto-pursuit behavior when
  pressing a primary action button does not lead to the pursuit of a
  nearby monster.

### Various Fixes

- Fixed missing events in the main event loop when running in headless
  mode, which was causing the AI agent to get stuck after an event had
  been sent, but no reaction occurred.

- Fixed access to graphics and audio objects in `headless` mode. A few
  bugs were causing random crashes of the `DevilutionX` instance.

- Fixed long-standing bug where objects aligned with X/Y axis became
  invisible under certain light conditions. Improved raycasting logic
  with adjacent tile checks.

- Fixed light rays leaking through diagonally adjacent corners,
  further refining the lighting model.

The listed changes made it possible to monitor and manage the state of
the Diablo game from an RL agent, and also added stability during
parallel AI training.
