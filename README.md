# What is DevilutionX-AI

A [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)-based
framework for reinforcement learning agent in the environment of the
game Diablo. The game itself is a patched open-source port of Diablo,
called [DevilutionX](https://github.com/diasurgical/DevilutionX/),
with additional patches for RL needs. The RL framework includes a
Gymnasium environment, [patches](#devilutionx-patches) for
DevilutionX, a DevilutionX runner, and an
[RLlib](https://docs.ray.io/en/latest/rllib/index.html) training
pipeline.

The goal of the `DevilutionX-AI` is to train an RL agent, the Warrior
player, to clear the first level by exploring the dungeon. This
includes finding the descent to the next level, killing all the
monsters along the way, picking up items, opening chests, or
activating other objects. Essentially, the aim is for the agent to
emulate what a human player typically does when starting to play
Diablo.

I would like to clarify that this project does not aim to train an
agent capable of completing the entire game. Initially, I would like
to observe "signs of life" from an RL agent exploring the first level
of the dungeon, without adding complex behaviors such as returning to
town, using spells, changing clothes, etc.

The fact that I am not an expert in reinforcement learning and that my
main daily duties are not related to AI complicates the chosen
task. That is why I started with a straightforward and minimal goal.

Ultimately, I hope that the framework will attract people with
experience in RL. Perhaps it will be possible to achieve Diablo
gameplay performed by the RL agent that is indistinguishable from that
of a human player.

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

For RL training purposes, data from the DevilutionX engine
implementation is extracted as a two-dimensional array 112x112
representation of the whole dungeon, along with descriptor arrays for
dungeon objects, states for non-player characters, various counters,
and of course, the player's own state: hit points, current dungeon
level, position in the dungeon, status, etc. All state structures are
shared by the engine through a memory file, a large blob which the AI
agent can access using Linux APIs such as `mmap`. All actions are
keyboard presses that the agent sends to the game engine through a
ring buffer and the same shared memory. To get everything working, it
was necessary to make a set of [changes](#devilutionx-patches) to the
original `DevilutionX` project.

## Observation Space

The observation space in reinforcement learning represents the domain
of various experiments, trials, and errors. Currently, the entire
environment is observed by the RL agent, meaning the entire state of
the dungeon is fetched, which is a two-dimensional array of size
112x112 of type uint16. Each tile in the two-dimensional dungeon is a
set of 16 bits, each of which describes a specific state of the tile,
such as the presence of a player, a monster, a wall, a closed or open
door, a chest, an item, and whether the tile has already been explored
or is visible to the player. In addition to the environment, the
status of the environment is also provided, namely an array consisting
of the following counter variables: the number of monsters on the
level, player hit points, the player's position in the dungeon’s
two-dimensional space, and the player's mode (idle, movement, attack,
etc.).

To combine two data arrays (the entire 2D environment and the
environment status defined as `gym.spaces.Box`) into a single
observation space, `gym.spaces.Dict` is used:

```python
self.observation_space = gym.spaces.Dict({
        "env-status":  gym.spaces.Box(low=0,
                                      high=0xffffff,
                                      shape=env_status.shape,
                                      dtype=env_status.dtype),
        "environment": gym.spaces.Box(low=0,
                                      high=env_high,
                                      shape=env.shape,
                                      dtype=env.dtype),
    }
)
```

I'm not confident that selecting these specific data and data types
for the observation space is the optimal and correct choice, but at the
moment, all my tests and experiments are based on this selection.

## Action Space

The choice of action space is much simpler: the player can stand still
or move in eight cardinal directions: north, northeast, east,
southeast, south, southwest, west, and northwest. Additionally, the
player can perform exactly two types of actions: primary and secondary
action, where primary action includes attacking monsters, talking to
towners, lifting and placing inventory items. Meanwhile, a secondary
action involves opening chests, interacting with doors, and picking up
items.

Since there are only 11 possible discrete actions, the action space is
defined using `gym.spaces.Discrete` type:

```python
self.action_space = gym.spaces.Discrete(len(DiabloEnv.ActionEnum))
```

where the length of `DiabloEnv.ActionEnum` is 11.

## Reward Function

The agent uses a shaped reward function (see the detailed description
about [reward shaping](#reward-shaping) below this section) to learn efficient behavior
in the dungeon environment. The function combines major outcome
rewards with smaller intermediate rewards that provide continuous
feedback during training.

**Major rewards**:

- Death results in a large penalty (-100) and ends the episode.

- Escaping back to town gives a moderate penalty (-10) and also ends
  the episode.

- Descending to the next dungeon level gives a strong reward (+50) and
  ends the episode.

**Shaping rewards**:

These are smaller rewards that guide the agent toward productive
behavior:

- Taking damage reduces reward in proportion to health lost.

- Exploring new areas in dungeon increases reward, either with a flat
  value or scaled by proximity to unexplored doors (if door-attraction
  mode is enabled with the `--exploration-door-attraction` option).

- Opening doors, activating objects (like chests or barrels), and
  collecting items each give small positive rewards.

- Fighting monsters provides:

   - A small reward for damaging enemies.
   - A larger reward for killing them (+20 per kill).

- No meaningful activity results in a small penalty (-0.1).

- Getting stuck (e.g., by repeating useless actions) triggers early
  truncation of the episode with a minor penalty (-5).

Internally, the function tracks several metrics such as monster
health, number of opened doors, explored tiles, and items collected to
determine whether the agent is actively interacting with the
environment.

### Reward Shaping

Reward shaping is a way to make learning easier for an RL agent by
adding extra rewards that guide it toward the final goal, especially
when the main reward is delayed or sparse.

As numerous tests have shown, it is by no means an easy task to train
an RL agent to explore a dungeon, even without monsters (you can use
the option `--no-monsters` to disable monsters, yay). It is necessary
to guide the agent toward doors that open unexplored areas of the
dungeon.

Small rewards serve as motivational signals to guide the agent in
exploring the dungeon. These rewards are given when the agent
approaches a door leading to the final goal (or to another room that
leads to the final goal, and so on). The final goal is the descent to
the next level of the dungeon. The mode, where more reward is given
the closer the player is to the correct door, can be enabled with the
`--exploration-door-attraction` option.

To account for the doors leading through the dungeon to the final
goal, a graph with the shortest path to the destination is built
at the very start of the level. In this graph, the nodes represent
regions (rooms), while the leaves represent adjacent regions (rooms)
that can be reached. The regions (rooms) are connected by doors, whose
coordinates are known, making it possible to construct a function that
gives a reward inversely proportional to the proximity to the correct
door: the closer to the correct door, the more reward is received.

Given:

- $\lambda\$ (lambda factor) is a constant factor controlling the
  decay smoothness if proximity to a door increases (e.g., $\lambda = 0.3\$).
- $\text{discount}(door)$ is a function that returns 0 if the door is
  on the path to the goal, and 2 otherwise (number chosen empirically).
- $\text{dist}(pos, door)$ is the Euclidean distance between the
  current player position and the door.

The distance calculation for each door is:

$$
\text{dists} = \left[ (\text{dist}(pos, door) + \text{discount}(door)) \times \lambda \mid \text{for each door} \right]
$$

And the reward function for exploration is:

$$
\text{reward} = \frac{1}{1 + \min(\text{dists})}
$$

Which is inversely proportional to the smallest distance, ensuring the
agent gets a higher reward for being closer to a door (with
adjustments based on the discount, because not all doors matter).

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
               --train-batch-size=10000 \
               --train-iters=1000 \
               --env-runners=30 \
               --gpus=1 \
               --seed=0 \
               --same-seed \
               --no-monsters \
               --save-to-checkpoint \
               --restore-from-checkpoint \
               --exploration-door-attraction \
               --exploration-door-backtrack-penalty
```

Where:

- `--train-batch-size` - determines how many samples (environment
  steps) are collected across all workers before an optimization step
  (gradient update) is performed.

- `--train-iters` - determines how many times training loop runs, or
  how many times RLlib collects experience, computes gradients, and
  updates the policy.

- `--env-runners` - determines how many workers interact with the
  environment and how many `DevilutionX` game instances start in
  parallel.

- `--gpus` - number of GPUs used for calculating gradint update.

- `--seed` - the initial seed for the `DevilutionX` environment sets
  the randomization controls for Diablo's gameplay. This seed ensures
  that the dungeon's layout and features generated by the
  `DevilutionX` engine can be replicated consistently across different
  runners.

- `--same-seed` - the same initial seed will be used for every runner
  during training. If the `--same-seed` option is not provided, each
  `DevilutionX` instance will use a seed equal to `seed +
  runner_index`, resulting in different dungeons across
  environments. Using `--same-seed` often accelerates learning, but it
  can also lead to overfitting, where the model performs poorly on
  new, unseen dungeons.

- `--no-monsters` - disables monsters in the dungeon, which makes the
  agent learn only basic exploration actions.

- `--save-to-checkpoint` - the learning state is saved to the
  checkpoint folder after each training iteration.

- `--restore-from-checkpoint` - the learning state is restored from
  the checkpoint folder before training starts.

- `--exploration-door-attraction` - enables the mode where the reward
  increases as the agent moves closer to unexplored doors, which lead
  to the final goal: descent to the next level (see the [reward
  shaping](#reward-shaping) section for details).

- `--exploration-door-backtrack-penalty` - enables the mode where the
  agent receives a negative reward if the proximity to the unexplored
  door starts increasing, basically when agent moves in the other
  direction (see the [reward shaping](#reward-shaping) section for
  details).

For reinforcement learning, I use RLlib's default **PPO** (Proximal
Policy Optimization) algorithm. There are also a few hardcoded
hyperparameters (settings that control how an RL model is
trained). Hyperparameters are not learned by the model itself but must
be set manually:

- `lr=0.0001` - learning rate determines the step size at each update;
  too high can cause the model to overshoot; too low can make training
  very slow or get stuck.

- `entropy_coeff=0.03` - entropy coefficient controls the balance
  between exploration and exploitation.

Hyperparameters are the subject of many experiments. For example, a
low entropy coefficient can result in a Diablo RL agent getting stuck
in one room without taking any further actions, or wandering from
corner to corner.

This list of game and training parameters used in my experiments is by
no means optimal. I am continually exploring the behavior of an RL
agent and frequently adjust parameters or introduce new ones to
achieve the desired results.

## Agent Evaluation

I would like to show visual results from evaluating a pre-trained RL
model, based on the training parameters outlined in the
[previous](#agent-training) section.

To run the RL agent in evaluation mode, simply execute the following
command:

```shell
./diablo-ai.py play-ai --no-monsters
```

As soon as the agent starts, you need to attach to the `DevilutionX`
instance with the TUI (text-based) frontend from another shell:

```shell
./diablo-ai.py play attach 0
```

The following are the recordings of the RL agent exploring the dungeon
(at least a few rooms). The first recording is the TUI (text-based)
frontend:

<p align="center">
   <img src="https://github.com/user-attachments/assets/24e164e8-c2c8-4ab1-8516-3604654699cc">
</p>

The other recording was made from the `DevilutionX` graphics window
during the second evaluation run. The RL agent's decisions are
slightly different, but the overall direction of the movement is
similar:

https://github.com/user-attachments/assets/56ac988c-2c6e-4895-bbf0-c9ea5dac4022

After watching this video, it's quite clear that there are many random
actions in the RL agent's behavior. However, the Warrior did manage to
advance three rooms ahead, even though it got stuck in the left
corner, which requires further analysis and/or tuning of the reward
function.

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
        Diablo ticks: 263 (missed: 0); Kills: 000; HP: 4480; Pos: 83:50; State: PM_STAND
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
./diablo-ai.py attach 0
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

- Enabled line-buffered stdout (`setlinebuf`) and additional game
  event logs (pause, save, load) for the AI to consume the log from
  the `DevilutionX` engine.

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
