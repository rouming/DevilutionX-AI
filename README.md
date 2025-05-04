<div style="width:100%;">
  <a href="https://www.youtube.com/watch?v=JKrBJXbmbjQ" target="_blank">
    <img src="https://github.com/user-attachments/assets/400cbd5c-9b56-4208-8680-1d68f56fd29d" style="width:100%; height:auto;" />
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
training pipeline. The RL part is based on the [BabyAI
project](https://github.com/mila-iqia/babyai), with its PPO
implementation and CNN architecture modified for this setup.

The goal is to train an agent (the Warrior) to explore the first
dungeon level. That means exploring the dungeon, fighting monsters,
picking up items, opening chests, activating other objects, or finding
the stairs to the next level - basically what a human would do when
just starting the game.

The short video at the top of this README demonstrates the agent
successfully locating a randomly placed portal in an environment where
monsters are disabled. The agent explores 10 randomly generated levels
using the default seed 0 (more on replicating the results below) with
the pre-trained model, which achieved a success rate of 0.967 during
training.

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

## Training Pecularitites

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

At the current stage, the reward function is sparse: the agent
receives a positive reward when it reaches the goal (the stairs to the
next level) and `0` otherwise.

As the project progresses, the reward function will be extended to
account for additional events such as the agent's death, monster
defeats, opening chests, and item collection. A more detailed design
may look as follows:

**Major rewards**:

- **Death** - large penalty (-100) and episode ends.

- **Escaping back to town** - moderate penalty (-10), episode ends.

- **Descending to the next level** - strong reward (+50), episode ends.

**Shaping rewards**:

These are smaller rewards that guide the agent toward productive
behavior:

- **Damage taken** - penalty proportional to health lost.

- **Exploration** - reward for visiting previously unseen tiles.

- **Interactions** - small rewards for opening doors, activating
  objects (e.g., chests, barrels), or collecting items.

- **Combat**

   - small reward for damaging enemies.
   - larger reward for killing them (+20 per kill).

- **Inactivity** - small penalty (-0.1) for unproductive actions.

- **Getting stuck** (e.g., by repeating useless actions) - early
  truncation of an episode with a minor penalty (-5).

Internally, the reward function may track metrics such as monster
health, number of opened doors, explored tiles, and collected
items. This allows the agent's behavior to be guided not only by
long-term goals but also by immediate, meaningful interactions with
the environment.

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

Choosing the right parameters and their combinations for effective RL
training is an art and essentially a path of endless trial and
error. For example, I use the following command line:

```shell
./diablo-ai.py train-ai \
   --env Diablo-FindNextLevel-v0 \
   --model Diablo-FindNextLevel-v0--cnn32-best \
   --cnn-arch cnn32 \
   --no-monsters \
   --frames 50M \
   --frames-per-env 320 \
   --env-runners 32 \
   --batch-size 10240 \
   --recurrence 20 \
   --embedding-dim 512 \
   --gae-lambda 0.99 \
   --lr 5e-05 \
   --optim-eps 1e-8 \
   --entropy-coef 0.01 \
   --epochs 5
```

Where:

- `--env Diablo-FindNextLevel-v0` - The environment the agent
  interacts with. Here, the task is to navigate the dungeon and find
  the next level.

- `--model Diablo-FindNextLevel-v0--cnn32-best` - Name of the model
  used for training. Essentially, it's a folder where the model files
  are located.

- `--cnn-arch cnn32` - The convolutional neural network architecture
  used to process observations. `cnn32` is just a name, meaning the
  3rd model, 2nd version.

- `--no-monsters` - Disables monsters in the environment, simplifying
  training by focusing on navigation.

- `--frames 50M` - Total number of environment frames (steps) the
  agent will be trained on.

- `--frames-per-env 320` - Number of steps each environment instance
  runs before sending data to the optimizer.

- `--env-runners 32` - Number of parallel environment instances used
  for training, allowing faster experience collection.

- `--batch-size 10240` - Number of frames (steps) collected before
  performing a gradient update.

- `--recurrence 20` - Length of temporal sequences used for recurrent
  policy updates (for RNN/LSTM agents, representing a memory).

- `--embedding-dim 512` - Size of the latent embedding vector produced
  by the CNN.

- `--gae-lambda 0.99` - Lambda parameter for Generalized Advantage
  Estimation, controlling bias-variance tradeoff in advantage
  calculation.

- `--lr 5e-05` - Learning rate for the optimizer.

- `--optim-eps 1e-8` - Small epsilon added to the optimizer for
  numerical stability.

- `--entropy-coef 0.01` - Weight of the entropy regularization term,
  encouraging exploration.

- `--epochs 5` - Number of optimization passes over each collected
  batch of experience.

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
   --env Diablo-FindRandomGoal-v0 \
   --cnn-arch cnn32 \
   --embedding-dim 512 \
   --model Diablo-FindRandomGoal-v0--cnn32-best \
   --episodes 10 \
   --harmless-barrels \
   --no-monsters \
   --seed 0 \
   --game-ticks-per-step 12 \
   --gui \
   --best
```

As soon as the Diablo GUI window appears, select "Single Game" and
proceed with the "Warrior" character, using the default name and
normal difficulty (monsters will be disabled anyway). Once the first
level is loaded, the agent resets the environment a few times and
starts looking for a randomly placed portal. The episode ends if the
task is completed, meaning the agent finds a goal, or if the agent is
stuck, resulting in task failure. The agent will be traversing ten
randomly generated dungeons (controlled by the `--episodes 10`
option).

To attach a terminal ASCII representation to the running game
instance, use the following command:

```shell
./diablo-ai.py play --attach 0
```

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
