import datetime
import numpy as np
import os
import shutil
import time
import torch

from rl import utils
from rl.evaluate import ManyEnvs
from rl.evaluate import batch_evaluate
from rl.model import ACModel
from rl.utils import device
import sprout


class BotEnv:
    def __init__(self, bot):
        self.bot = bot

    def reset(self, seed=None):
        # Bot should be reset right after environment reset
        self.bot.reset(seed=seed)
        return None, None

    def step(self, dummy_action):
        done, action = self.bot.step()
        # Be aware of differences from the original environment step
        # API call convention: dummy action is not used (bot knows how
        # to act), but a true action from bot is returned as a last
        # tuple element. Bot `done` flag is returned as a termination
        # flag.
        return None, None, done, False, None, action

class EpochIndexSampler:
    """
    Generate smart indices for epochs that are smaller than the dataset size.

    The usecase: you have a code that has a strongly baken in notion of an epoch,
    e.g. you can only validate in the end of the epoch. That ties a lot of
    aspects of training to the size of the dataset. You may want to validate
    more often than once per a complete pass over the dataset.

    This class helps you by generating a sequence of smaller epochs that
    use different subsets of the dataset, as long as this is possible.
    This allows you to keep the small advantage that sampling without replacement
    provides, but also enjoy smaller epochs.
    """
    def __init__(self, n_examples, epoch_n_examples):
        self.n_examples = n_examples
        self.epoch_n_examples = epoch_n_examples

        self._last_seed = None

    def _reseed_indices_if_needed(self, seed):
        if seed == self._last_seed:
            return

        rng = np.random.RandomState(seed)
        self._indices = list(range(self.n_examples))
        rng.shuffle(self._indices)

        self._last_seed = seed

    def get_epoch_indices(self, epoch):
        """Return indices corresponding to a particular epoch.

        Tip: if you call this function with consecutive epoch numbers,
        you will avoid expensive reshuffling of the index list.

        """
        seed = epoch * self.epoch_n_examples // self.n_examples
        offset = epoch * self.epoch_n_examples % self.n_examples

        indices = []
        while len(indices) < self.epoch_n_examples:
            self._reseed_indices_if_needed(seed)
            n_lacking = self.epoch_n_examples - len(indices)
            indices += self._indices[offset:offset + min(n_lacking, self.n_examples - offset)]
            offset = 0
            seed += 1

        return indices

def compute_mc_returns(rewards, dones, next_value, discount=0.99):
    T = len(rewards)
    returns = torch.zeros((T,), device=device, dtype=torch.float32)
    # Monte Carlo return
    R = next_value
    for t in reversed(range(T)):
        R = rewards[t] + discount * R * (1 - dones[t])
        returns[t] = R

    return returns

def compute_gae_returns(rewards, dones, lam, values, gamma=0.99):
    T = len(rewards)
    returns = torch.zeros((T,), device=device, dtype=torch.float32)
    # GAE return (lambda between 0 and 1)
    advantages = torch.zeros((T,), device=device, dtype=torch.float32)
    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]

    return returns

class ImitationLearning(object):
    def __init__(self, args, spr, penv_pool, pbot_pool, model_dir, phase_no,
                 tb_writer, txt_logger, csv_logger,
                 train_policy=True, train_critic=False):
        self.args = args
        self.spr = spr
        self.penv_pool = penv_pool
        self.pbot_pool = pbot_pool
        self.tb_writer = tb_writer
        self.txt_logger = txt_logger
        self.csv_logger = csv_logger
        self.val_seed = self.args.val_seed
        self.train_policy = train_policy
        self.train_critic = train_critic

        if phase_no is None:
            model_phase_dir = model_dir
        else:
            model_phase_dir = os.path.join(model_dir, f"phase-{phase_no}")
            # Ensure that the model directory exists
            os.makedirs(model_phase_dir, exist_ok=True)

        self.model_dir = model_dir
        self.model_phase_dir = model_phase_dir

        if getattr(args, 'multi_env', None):
            self.train_demos = []
            for demos, episodes in zip(args.multi_demos, args.multi_episodes):
                demos_path = utils.get_demos_path(model_dir, None, valid=False)
                txt_logger.info('loading {} of {} demos'.format(episodes, demos))
                train_demos = utils.load_demos(demos_path)
                txt_logger.info('loaded demos')
                if episodes > len(train_demos):
                    raise ValueError("there are only {} train demos in {}".
                                     format(len(train_demos), demos))
                self.train_demos.extend(train_demos[:episodes])
                txt_logger.info('So far, {} demos loaded'.format(len(self.train_demos)))

            txt_logger.info('Loaded all demos')

            observation_space = self.penv_pool.envs[0].observation_space
            action_space = self.penv_pool.envs[0].action_space

        else:
            demos_path = utils.get_demos_path(model_dir, args.env, valid=False)
            demos_path_valid = utils.get_demos_path(model_dir, args.env, valid=True)

            self.train_demos = utils.load_demos(demos_path)
            txt_logger.info(f'Loaded train demos {len(self.train_demos)}')
            if args.episodes_int:
                if args.episodes_int > len(self.train_demos):
                    raise ValueError("there are only {} train demos".format(len(self.train_demos)))
                self.train_demos = self.train_demos[:args.episodes_int]

            observation_space = self.penv_pool.envs[0].observation_space
            action_space = self.penv_pool.envs[0].action_space

        # Generate demos for validation if needed
        if self.args.val_interval > 0:
            seed = args.val_seed
            seeds = list(range(seed, seed + args.val_episodes))
            txt_logger.info(f'Generating {args.val_episodes} validation demos')
            self.val_demos, _, _ = ImitationLearning.generate_demos(
                self.pbot_pool, seeds)

        # Load training status
        need_save = False
        try:
            status = utils.get_status(model_phase_dir)
        except OSError:
            status = {'num_frames': 0, 'update': 0, 'patience': 0}
            need_save = True

        txt_logger.info("Training status loaded\n")

        # Load observations preprocessor
        obs_space, preprocess_obss = utils.get_obss_preprocessor(observation_space)
        if "vocab" in status:
            preprocess_obss.vocab.load_vocab(status["vocab"])
        txt_logger.info("Observations preprocessor loaded")

        self.preprocess_obss = preprocess_obss

        # Load model
        self.acmodel = ACModel(obs_space, action_space, args.cnn_arch,
                               embedding_dim=args.embedding_dim,
                               use_memory=True, use_text=False)
        assert self.acmodel.recurrent, "Currently, non-recurrent models are not supported."

        # Training the critic only requires special attention
        if not self.train_policy and self.train_critic:
            # Freeze all parameters in the entire model
            for param in self.acmodel.parameters():
                param.requires_grad = False

            # Unfreeze critic parameters only
            for param in self.acmodel.critic.parameters():
                param.requires_grad = True

        if "model_state" in status:
            self.acmodel.load_state_dict(status["model_state"])
        self.acmodel.to(device)

        # In case we train both - separate parameters in order to have
        # granular control over the LR
        if self.train_policy and self.train_critic:
            # Everything shared or belonging to the actor is "delicate",
            # we should not destroy the policy by the critic
            delicate_params = []
            # Only specific layers of the critic head are "robust"
            # since we still want to push the critic to higher accuracy
            critic_params = []

            for name, param in self.acmodel.named_parameters():
                if 'critic' in name:
                    critic_params.append(param)
                else:
                    # Everything else (encoder, memory, actor, etc.) is 'delicate'
                    delicate_params.append(param)

            self.optimizer = torch.optim.Adam([
                {'params': delicate_params, 'lr': self.args.lr_delicate},
                {'params': critic_params,   'lr': self.args.lr}
            ], eps=self.args.optim_eps)
        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(),
                                              self.args.lr,
                                              eps=self.args.optim_eps)

        train_mode = status.get('il_train_mode')
        self.train_mode_changed = (train_mode != (train_policy, train_critic))
        status['il_train_mode'] = (train_policy, train_critic)
        # Remove previous RL optimizer state if any
        status.pop('optimizer_state', None)
        if not self.train_mode_changed :
            # We load the optimizer state if this is a continuation of
            # the training
            if "il_optimizer_state" in status:
                self.optimizer.load_state_dict(status["il_optimizer_state"])
                txt_logger.info("Optimizer loaded from the state\n")

        # Create exponential decay LR scheduler, so every N steps LR
        # reduced by gamma
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_steps,
            gamma=args.lr_gamma)

        if need_save:
            # Model saved initially for the first validation step
            status.update({"model_state": self.acmodel.state_dict(),
                           "il_optimizer_state": self.optimizer.state_dict()})
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_phase_dir)

    def starting_indexes(self, num_frames):
        if num_frames % self.args.recurrence == 0:
            return np.arange(0, num_frames, self.args.recurrence)
        else:
            return np.arange(0, num_frames, self.args.recurrence)[:-1]

    def run_epoch_recurrence(self, demos, is_training=False, indices=None):
        if not indices:
            indices = list(range(len(demos)))
            if is_training:
                np.random.shuffle(indices)
        batch_size = min(self.args.batch_size, len(indices))
        num_batches = (len(indices) + batch_size - 1) // batch_size
        offset = 0

        if not is_training:
            self.acmodel.eval()

        # Log dictionary
        log = {"entropy": [],
               "policy_loss": [],
               "value_loss": [],
               "policy_accuracy": [],
               "value_accuracy": [],
               "grad_norm": [],
               }

        total_frames = 0
        for i in range(num_batches):
            batch_size = min(batch_size, len(indices) * batch_size - offset)
            batch = [demos[i] for i in indices[offset: offset + batch_size]]
            frames = sum([len(actions) for seed, actions in batch])
            total_frames += frames
            offset += batch_size
            ts = time.time()

            _log = self.run_epoch_recurrence_one_batch(batch, is_training=is_training)

            diff = time.time() - ts

            if num_batches > 1:
                self.txt_logger.info("batch {}/{} | steps {} | FPS {:.0f} | took {:.2f}s".format(
                    i + 1, num_batches, frames, frames / diff, diff))

            log["entropy"].append(_log["entropy"])
            log["policy_loss"].append(_log["policy_loss"])
            log["value_loss"].append(_log["value_loss"])
            log["policy_accuracy"].append(_log["policy_accuracy"])
            log["value_accuracy"].append(_log["value_accuracy"])
            log["grad_norm"].append(_log["grad_norm"])

        log['total_frames'] = total_frames

        if not is_training:
            self.acmodel.train()

        return log

    def run_epoch_recurrence_one_batch(self, batch, is_training=False):
        batch = utils.demos.transform_demos(batch)
        by_actions_len = lambda episode: len(episode[1])
        batch.sort(key=by_actions_len, reverse=True)

        # Constructing flat batch and indices pointing to start of
        # each demonstration
        flat_batch = []
        inds = [0]

        num_envs = min(len(self.penv_pool.envs), len(batch))
        env = ManyEnvs(self.penv_pool)

        # Generate observations based on true actions from demos.
        # Similar to collect experiences step for regular PPO
        for offset in range(0, len(batch), num_envs):
            size = min(len(batch) - offset, num_envs)
            episodes = batch[offset: offset + size]

            seeds, true_actions = zip(*episodes)
            obs, _ = env.reset(seeds=seeds)

            obss = [[] for _ in range(size)]
            dones = [[] for _ in range(size)]
            rewards = [[] for _ in range(size)]
            steps = np.zeros((size,), dtype=int)
            not_yet_done = np.ones((size,), dtype=bool)
            ts = time.time()

            while np.any(not_yet_done):
                active_indices = np.flatnonzero(not_yet_done)
                actions = [true_actions[i][steps[i]] for i in active_indices]
                new_obss, reward, terminated, truncated, _ = \
                    env.step(actions, active_indices)
                done = np.asarray(terminated) | np.asarray(truncated)

                for i, o, r, d in zip(active_indices, obs, reward, done):
                    obss[i].append(o)
                    dones[i].append(d)
                    rewards[i].append(r)
                    steps[i] += 1

                obs = new_obss
                just_done_indices = active_indices[done]
                not_yet_done[just_done_indices] = False

            diff = time.time() - ts
            sum_steps = np.sum(steps)
            if False:
                self.txt_logger.info(f"generated OBSs for episodes {offset + size}/{len(batch)}: steps {sum_steps} | MAX steps {np.max(steps)} | AVG steps {np.mean(steps):.0f} | {sum_steps / diff:.0f} FPS, took {diff:.0f}s")

            for obss_, actions_, dones_, rewards_, seed_ in \
                    zip(obss, true_actions, dones, rewards, seeds):
                total_reward = np.sum(rewards_)
                if total_reward <= 0:
                    self.txt_logger.warning(f"Environment was not able to get positive reward for demos generated by seed {seed_}")
                    continue

                assert len(obss_) == len(actions_) == len(dones_) == len(rewards_)
                flat_batch += [(o, a, d, r) for o, a, d, r in
                               zip(obss_, actions_, dones_, rewards_)]
                inds.append(inds[-1] + len(actions_))

        # Do training with collected experiences

        flat_batch = np.array(flat_batch, dtype=object)
        inds = inds[:-1]
        num_frames = len(flat_batch)

        mask = np.ones([len(flat_batch)], dtype=np.float64)
        mask[inds] = 0
        mask = torch.tensor(mask, device=device, dtype=torch.float).unsqueeze(1)

        # Observations, true action and done for each of the stored demostration
        obss, actions_true, dones, rewards = (flat_batch[:, 0], flat_batch[:, 1],
                                              flat_batch[:, 2], flat_batch[:, 3])
        actions_true = torch.as_tensor(actions_true.astype(dtype=int, copy=False),
                                       device=device, dtype=torch.long)

        # Episodes always reach the end, so next_value is zero
        returns = compute_mc_returns(rewards, dones, next_value=0, discount=0.99)

        # Memory to be stored
        memories = torch.zeros([len(flat_batch), self.acmodel.memory_size], device=device)
        episode_ids = np.zeros(len(flat_batch))
        memory = torch.zeros([len(batch), self.acmodel.memory_size], device=device)

        # Loop terminates when every observation in the flat_batch has
        # been handled
        while True:
            # taking observations and done located at inds
            obs = obss[inds]
            done_step = dones[inds]
            with torch.no_grad():
                preprocessed_obs = self.preprocess_obss(obs, device=device)
                # taking the memory till len(inds), as demos beyond
                # that have already finished
                dist, value, new_memory = self.acmodel(preprocessed_obs,
                                                       memory[:len(inds), :])
            memories[inds, :] = memory[:len(inds), :]
            memory[:len(inds), :] = new_memory
            episode_ids[inds] = range(len(inds))

            # Updating inds, by removing those indices corresponding
            # to which the demonstrations have finished
            inds = inds[:len(inds) - sum(done_step)]
            if len(inds) == 0:
                break

            # Incrementing the remaining indices
            inds = [index + 1 for index in inds]


        # Here, actual backprop upto args.recurrence happens
        final_loss, final_entropy = 0.0, 0.0
        final_value_loss, final_policy_loss = 0.0, 0.0
        value_accuracy, policy_accuracy = 0.0, 0.0
        grad_norm = 0.0

        indexes = self.starting_indexes(num_frames)
        memory = memories[indexes]

        for _ in range(self.args.recurrence):
            obs = obss[indexes]
            preprocessed_obs = self.preprocess_obss(obs, device=device)
            action_step = actions_true[indexes]
            mask_step = mask[indexes]

            dist, value, memory = self.acmodel(preprocessed_obs, memory * mask_step)
            r = returns[indexes]

            # Standard MSE loss for critic
            value_loss = torch.mean((value - r) ** 2)
            # Pearson correlation for value accuracy
            stacked = torch.stack([value, r], dim=0).detach()
            corr = torch.corrcoef(stacked)[0, 1]
            corr = torch.nan_to_num(corr, nan=0.0)
            value_accuracy += float(corr)

            # Actor
            policy_loss = -dist.log_prob(action_step).mean()
            # Entropy is needed for calculating entropy bonus, which
            # trains the policy to be stochastic and encourages
            # exploration. By subtracting an entropy penalty for low
            # entropy, the model is encouraged to increase its
            # entropy, making it more random. This helps maintain a
            # healthy amount of randomness and is not specifically
            # intended to aid IL. Instead, it prepares the policy PPO
            # fine-tuning
            entropy = dist.entropy().mean()

            action_pred = dist.probs.argmax(dim=1)
            policy_accuracy += (action_pred == action_step).float().mean().item()

            if self.train_policy and not self.train_critic:
                final_loss += policy_loss - self.args.entropy_coef * entropy
            elif not self.train_policy and self.train_critic:
                final_loss += value_loss
            elif self.train_policy and self.train_critic:
                final_loss += (value_loss * self.args.value_loss_coef +
                               policy_loss - self.args.entropy_coef * entropy)
            else:
                assert 0, "Unknown training mode"

            # Accumulate
            final_entropy += entropy
            final_policy_loss += policy_loss
            final_value_loss += value_loss

            indexes += 1

        final_loss /= self.args.recurrence

        if is_training:
            self.optimizer.zero_grad()
            final_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(),
                                                       self.args.max_grad_norm).item()
            self.optimizer.step()

        log = {
            "entropy": float(final_entropy / self.args.recurrence),
            "value_loss": float(final_value_loss / self.args.recurrence),
            "policy_loss": float(final_policy_loss / self.args.recurrence),
            "value_accuracy": float(value_accuracy / self.args.recurrence),
            "policy_accuracy": float(policy_accuracy / self.args.recurrence),
            "grad_norm": grad_norm
        }

        return log

    def validate(self, episodes):
        self.txt_logger.info("Validating the model's {} demo episodes with {} environments".format(
            episodes, min(episodes, len(self.penv_pool.envs))))

        num_envs = min(len(self.penv_pool.envs), episodes)
        # Create an agent using the current model
        agent = utils.Agent.from_external_model(
            self.acmodel,
            self.penv_pool.envs[0].observation_space,
            argmax=True, num_envs=num_envs)

        env_names = [self.args.env] if not getattr(self.args, 'multi_env', None) \
            else self.args.multi_env

        # Please FIXME
        assert getattr(self.args, 'multi_env', None) is None, \
            "multi-env is not supported right now"

        logs = []

        agent.acmodel.eval()
        for _ in env_names:
            logs += [batch_evaluate(agent, self.penv_pool, self.val_seed, episodes)]
            self.val_seed += episodes
        agent.acmodel.train()

        return logs

    def train(self, header):
        if self.args.batch_size > len(self.train_demos):
            raise ValueError("batch size ({}) is bigger than the number of demo episodes ({})".
                             format(self.args.batch_size, len(self.train_demos)))

        status = utils.get_status(self.model_phase_dir)

        # Load best training status if exists
        best_success_rate = 0.0
        if not self.train_mode_changed:
            # We pick up the previously saved best status only if this is a
            # continuation of the training
            try:
                best_status = utils.get_status(self.model_phase_dir, best=True)
                best_success_rate = best_status.get("success_rate", 0.0)
            except OSError:
                pass

        total_start_time = time.time()

        epoch_length = self.args.epoch_length
        if not epoch_length:
            epoch_length = self.args.batch_size
        index_sampler = EpochIndexSampler(len(self.train_demos), epoch_length)

        while True:
            # if for some reason you're fine-tuning with an IL and RL
            # pretrained agent
            if 'patience' not in status:
                status['patience'] = 0
            # Do not learn if using a pre-trained model that already lost patience
            if status['patience'] > getattr(self.args, 'patience', 100):
                break
            if status['num_frames'] > self.args.frames_int:
                break

            update_start_time = time.time()

            indices = index_sampler.get_epoch_indices(status['update'])
            log = self.run_epoch_recurrence(self.train_demos, is_training=True,
                                            indices=indices)

            # Learning rate scheduler
            self.scheduler.step()

            status['num_frames'] += log['total_frames']
            status['update'] += 1

            update_end_time = time.time()
            total_elapsed_time = update_end_time - total_start_time

            # Print logs
            if self.args.log_interval > 0 and (status['update'] % self.args.log_interval == 0 or
                                               status['num_frames'] >= self.args.frames_int):

                fps = log['total_frames'] / (update_end_time - update_start_time)

                # Average everything across the batch
                for key in log:
                    log[key] = np.mean(log[key])

                train_data = [status['update'], status['num_frames'], fps, total_elapsed_time,
                              log['entropy'], log['policy_loss'], log['value_loss'],
                              log['policy_accuracy'], log['value_accuracy'],
                              log['grad_norm']]

                self.txt_logger.info(
                    "U {} | F {:06} | FPS {:04.0f} | D {:.0f} | H {:.3f} | pL {:.3f} | vL {:.3f} | pA {:.3f} | vA {:.3f} | ∇ {:.3f}".
                    format(*train_data))

                # Log the gathered data only when we don't evaluate the
                # validation metrics. It will be logged anyways afterwards
                # when status['update'] % self.args.val_interval == 0
                if self.args.val_interval == 0 or status['update'] % self.args.val_interval != 0:
                    # instantiate a validation_log with empty strings
                    # when no validation is done
                    validation_data = [''] * len([key for key in header if 'valid' in key])
                    assert len(header) == len(train_data + validation_data)
                    for key, value in zip(header, train_data):
                        self.tb_writer.add_scalar(key, float(value), status['num_frames'])
                    self.csv_logger.writerow(train_data + validation_data)

            if self.args.val_interval > 0 and (status['update'] % self.args.val_interval == 0 or
                                               status['num_frames'] >= self.args.frames_int):
                valid_log = self.validate(self.args.val_episodes)
                mean_return = [np.mean(log['return_per_episode']) for log in valid_log]
                success_rate = [np.mean([1 if r > 0 else 0 for r in log['return_per_episode']]) for log in
                                valid_log]
                mean_success_rate = np.mean(success_rate)

                if self.args.log_interval > 0 and (status['update'] % self.args.log_interval == 0 or
                                                   status['num_frames'] >= self.args.frames_int):
                    # Run through all validation demos and calculate
                    # the accuracy
                    start_time = time.time()
                    val_log = self.run_epoch_recurrence(self.val_demos)
                    elapsed_time = time.time() - start_time
                    validation_policy_accuracy = np.mean(val_log["policy_accuracy"])
                    validation_value_accuracy = np.mean(val_log["value_accuracy"])

                    validation_data = \
                        [validation_policy_accuracy, validation_value_accuracy] + \
                        mean_return + success_rate
                    self.txt_logger.info(("Validation: D {:.0f} | pA {:.3f} | vA {:.3f} " +
                                          "| R {:.3f} " * len(mean_return) +
                                          "| S {:.3f} " * len(success_rate) +
                                          "| bS {:.3f}",
                                          ).format(elapsed_time,
                                                   *validation_data,
                                                   best_success_rate))

                    assert len(header) == len(train_data + validation_data)
                    for key, value in zip(header, train_data + validation_data):
                        self.tb_writer.add_scalar(key, float(value), status['num_frames'])
                    self.csv_logger.writerow(train_data + validation_data)

                status.update({"success_rate": success_rate,
                               "model_state": self.acmodel.state_dict(),
                               "il_optimizer_state": self.optimizer.state_dict() })
                if hasattr(self.preprocess_obss, "vocab"):
                    status["vocab"] = self.preprocess_obss.vocab.vocab

                # In case of a multi-env, the update condition would
                # be "better mean success rate" !
                if mean_success_rate <= best_success_rate:
                    status['patience'] += 1
                    if hasattr(self.args, 'patience'):
                        self.txt_logger.info(
                            "Losing patience, new value={}, limit={}".
                            format(status['patience'], self.args.patience))

                utils.save_status(status, self.model_phase_dir)
                self.txt_logger.info("Status saved")

                custom_dict = {"duration": total_elapsed_time,
                               "frames": status["num_frames"],
                               "success_rate": mean_success_rate,
                               "policy_accuracy": log["policy_accuracy"],
                               "value_accuracy": log["value_accuracy"]}
                best = {"best": custom_dict}
                last = {"last": custom_dict}

                if mean_success_rate > best_success_rate:
                    best_success_rate = mean_success_rate
                    status['success_rate'] = best_success_rate
                    status['patience'] = 0

                    src_path = utils.get_status_path(self.model_phase_dir, best=False)
                    dst_path = utils.get_status_path(self.model_phase_dir, best=True)
                    shutil.copyfile(src_path, dst_path)
                    self.txt_logger.info("Success rate {: .2f}; best model is saved".
                                         format(best_success_rate))

                    self.spr.edit(head=self.args.model, custom_dict=last | best,
                                  custom_update=True)
                else:
                    self.spr.edit(head=self.args.model, custom_dict=last,
                                  custom_update=True)



        return best_success_rate


    def evaluate_agent(self, eval_seed, num_eval_episodes, return_obss_actions=False):
        """
        Evaluate the agent on some number of episodes and return the seeds for the
        episodes the agent performed the worst on.
        """

        self.txt_logger.info("Evaluating agent using {} episodes".format(num_eval_episodes))

        agent = utils.Agent.from_internal_model(
            self.penv_pool.envs[0].observation_space,
            self.penv_pool.envs[0].action_space,
            self.model_phase_dir,
            self.args.cnn_arch, argmax=False,
            num_envs=min(len(self.penv_pool.envs), num_eval_episodes),
            embedding_dim=self.args.embedding_dim,
            use_memory=True, use_text=False)

        agent.acmodel.eval()
        logs = batch_evaluate(
            agent,
            self.penv_pool,
            episodes=num_eval_episodes,
            seed=eval_seed,
            return_obss_actions=return_obss_actions
        )
        agent.acmodel.train()

        success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        self.txt_logger.info("success rate: {:.2f}".format(success_rate))

        # Find the seeds for all the failing demos
        fail_seeds = []
        fail_obss = []
        fail_actions = []

        for idx, ret in enumerate(logs["return_per_episode"]):
            if ret <= 0:
                fail_seeds.append(logs["seed_per_episode"][idx])
                if return_obss_actions:
                    fail_obss.append(logs["observations_per_episode"][idx])
                    fail_actions.append(logs["actions_per_episode"][idx])

        self.txt_logger.info("{} fails".format(len(fail_seeds)))

        if not return_obss_actions:
            return success_rate, fail_seeds
        else:
            return success_rate, fail_seeds, fail_obss, fail_actions


    @staticmethod
    def generate_demos(pbot_pool, all_seeds, pause=0.0):
        steps_cnt = 0
        demos = []
        durations = []

        num_envs = min(len(pbot_pool.envs), len(all_seeds))
        env = ManyEnvs(pbot_pool)

        for offset in range(0, len(all_seeds), num_envs):
            size = min(len(all_seeds) - offset, num_envs)
            seeds = all_seeds[offset: offset + size]
            durs = np.zeros((size,), dtype=float)

            _, _ = env.reset(seeds=seeds)

            actions = [[] for _ in range(size)]
            steps = np.zeros((size,), dtype=int)
            not_yet_done = np.ones((size,), dtype=bool)

            ts = time.time()

            while np.any(not_yet_done):
                active_indices = np.flatnonzero(not_yet_done)
                dummy_actions = np.zeros(active_indices.shape, dtype=int)
                _, _, terminated, _, _, action = \
                    env.step(dummy_actions, active_indices)
                done = np.asarray(terminated)

                if pause:
                    time.sleep(pause)

                for i, a, d in zip(active_indices, action, done):
                    # Skip last NOOP action
                    if not d:
                        actions[i].append(a)
                        steps[i] += 1

                just_done_indices = active_indices[done]
                durs[just_done_indices] = time.time() - ts
                not_yet_done[just_done_indices] = False

            durations.extend(durs.tolist())
            for seed_, actions_ in zip(seeds, actions):
                demos.append((seed_, actions_))

            steps_cnt += np.sum(steps)

        return demos, durations, steps_cnt


    def grow_training_set(self, eval_seed):
        """
        Grow the training set of demonstrations by some factor
        We specifically generate demos on which the agent fails
        """

        if self.train_demos:
            new_train_set_size = int(len(self.train_demos) * self.args.demo_grow_factor)
        else:
            new_train_set_size = self.args.start_demos
        num_new_demos = new_train_set_size - len(self.train_demos)

        self.txt_logger.info("Generating {} new demos".format(num_new_demos))

        # Add new demos until we rearch the new target size
        while len(self.train_demos) < new_train_set_size:
            num_new_demos = new_train_set_size - len(self.train_demos)

            # Evaluate the success rate of the model
            success_rate, fail_seeds = self.evaluate_agent(eval_seed, self.args.eval_episodes)
            eval_seed += self.args.eval_episodes

            fail_seeds = fail_seeds[:num_new_demos]

            # Generate demos for the worst performing seeds
            new_demos, _, _ = ImitationLearning.generate_demos(self.pbot_pool, fail_seeds)
            self.train_demos.extend(new_demos)

        return eval_seed
