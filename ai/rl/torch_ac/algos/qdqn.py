"""QDQNAlgo: pure Q-learning with QR-DQN + Dueling + Double Q + Watkins Q(lambda).

Replaces the full PPO actor-critic loop with a single distributional Q network.
No actor head, value head, GAE, entropy term, or PPO clip - just Q-learning.

Key design decisions:
  - QR-DQN: distributional quantile regression avoids fixed atom range
  - Dueling: V stream updates all actions from one death (cold-start robustness)
  - Double Q: live net selects a*, target net evaluates - prevents maximization bias
  - Watkins Q(lambda): multi-step returns without replay, trace cut at exploratory steps
  - Target net EMA: stabilizes bootstrap targets across rollouts
  - UCB exploration with per-runner beta spectrum: permanent diversity without decay
  - eps_warmup=0.5 not 1.0: eps=1.0 cuts Watkins trace ~87.5% of steps with 8 actions

References:
  QR-DQN (Dabney et al. 2017), Dueling DQN (Wang et al. 2016),
  Double DQN (van Hasselt et al. 2015), Watkins Q(lambda) (1989),
  ApeX (Horgan et al. 2018), R2D2 (Kapturowski et al. 2019)
"""

import copy
import numpy
import torch

from rl.torch_ac.algos.base import BaseAlgo, is_ddp
from rl.torch_ac.utils import DictList
from rl.q_critic import qr_loss


class QDQNAlgo(BaseAlgo):

    def __init__(self, penv_pool, global_seed, seeds, acmodel, device=None,
                 num_frames_per_proc=None, discount=0.99, lr=0.0001,
                 max_grad_norm=0.5, recurrence=4, adam_eps=1e-8,
                 epochs=3, batch_size=256,
                 ucb_beta_max=2.0, eps_base=0.02,
                 eps_warmup_start=0.5, eps_warmup_frames=500_000,
                 target_ema_tau=0.05, watkins_lambda=0.95,
                 preprocess_obss=None, reshape_reward=None):

        num_frames_per_proc = num_frames_per_proc or 128

        # Pass dummy gae/entropy/value params to BaseAlgo (unused by QDQNAlgo).
        super().__init__(penv_pool, global_seed, seeds, acmodel, device,
                         num_frames_per_proc, discount, lr,
                         gae_lambda=0.95, entropy_coef=0.0, value_loss_coef=0.0,
                         max_grad_norm=max_grad_norm, recurrence=recurrence,
                         preprocess_obss=preprocess_obss, reshape_reward=reshape_reward)

        self.epochs = epochs
        self.batch_size = batch_size
        self.watkins_lambda = watkins_lambda

        assert self.batch_size % self.recurrence == 0

        _m = acmodel.module if is_ddp(acmodel) else acmodel
        self.n_actions     = _m.q_head.num_actions
        self.num_quantiles = _m.q_head.num_quantiles

        # Target net: EMA lag prevents Q drift from feedback loops.
        # Without this, rollout k+1 computes targets from weights updated during
        # rollout k's epoch loop - moving targets cause unbounded Q escalation.
        # target_model is never DDP-wrapped (main process only, no DDP reduce).
        self.target_model = copy.deepcopy(_m)
        self.target_model.to(self.device)
        self.target_model.eval()
        self.target_ema_tau = target_ema_tau

        # Separate LSTM state for target net: target has different weights from
        # live net (EMA lag), so its LSTM gives different h-states for the same
        # obs sequence. Feeding live h into target Q head would mix state spaces.
        self.target_memory = torch.zeros(self.num_procs, self.memory_size,
                                         device=self.device)

        # ApeX-style per-runner UCB beta spectrum, fixed throughout training.
        # First half: greedy (beta=0) for clean Watkins traces.
        # Second half: UCB with increasing beta for permanent off-policy coverage.
        P = self.num_procs
        self.ucb_beta = torch.cat([
            torch.zeros(P // 2),
            torch.linspace(0.5, ucb_beta_max, P - P // 2),
        ], dim=0).to(self.device)  # (P,)

        self.eps_base = eps_base
        self.eps_warmup_start  = eps_warmup_start
        self.eps_warmup_frames = eps_warmup_frames
        self.total_frames      = 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        self._consecutive_nan_updates = 0

    # ------------------------------------------------------------------
    # Rollout collection with UCB action selection
    # ------------------------------------------------------------------

    def collect_experiences(self):
        for i in range(self.num_frames_per_proc):
            with torch.no_grad():
                preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
                q_all, memory = self.acmodel(
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            # q_all: (P, A, N)

            actions = self._ucb_actions(q_all)  # (P,)

            obs, _, terminated, truncated, info = self.env.step(actions.unsqueeze(1).cpu().numpy())
            done   = numpy.logical_or(terminated, truncated)
            reward = numpy.array([inf["hierarchy/reward"] for inf in info], dtype=float)
            # reward: (P, L); L=1 for this model

            self.obss[i] = self.obs
            self.obs     = obs
            if self.recurrent:
                self.memories[i] = self.memory
                self.memory      = memory
            self.env_counters = torch.tensor(
                [inf["env-counters"] for inf in info], dtype=int, device=self.device)
            self.masks[i]  = self.mask
            self.mask      = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.opt_masks[i] = self.opt_mask
            self.opt_mask    = torch.ones(self.num_procs, device=self.device)
            self.actions[i]  = actions.unsqueeze(1)  # (P,) -> (P, L=1)
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(o, a, r, False, d)
                    for o, a, r, d in zip(obs, actions.cpu().tolist(), reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)

            self.log_episode_return             += torch.tensor(reward, device=self.device,
                                                                dtype=torch.float)
            self.log_episode_reshaped_return    += self.rewards[i]
            self.log_episode_num_frames         += torch.ones(self.num_procs, device=self.device)
            for p, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[p].tolist())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[p].tolist())
                    self.log_num_frames.append(self.log_episode_num_frames[p].item())
                    self.log_success.append(info[p].get("success", False))
            self.log_episode_return          *= self.mask.unsqueeze(1)
            self.log_episode_reshaped_return *= self.mask.unsqueeze(1)
            self.log_episode_num_frames      *= self.mask

        self.total_frames += self.num_frames

        # Build P-major exps (same layout as BaseAlgo.collect_experiences)
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.recurrent:
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            exps.mask   = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        exps.action = self.actions.transpose(0, 1).reshape(-1, *self.actions.shape[2:])
        exps.obs    = self.preprocess_obss(exps.obs, device=self.device)

        keep = max(self.log_done_counter, self.num_procs)
        logs = {
            "return_per_episode":          self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode":      self.log_num_frames[-keep:],
            "success_per_episode":         self.log_success[-keep:],
            "num_episodes":                self.log_done_counter,
            "num_frames":                  self.num_frames,
        }
        self.log_done_counter    = 0
        self.log_return          = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames      = self.log_num_frames[-self.num_procs:]
        self.log_success         = self.log_success[-self.num_procs:]

        return exps, logs

    def _ucb_actions(self, q_all):
        """UCB action selection with warmup epsilon.

        UCB not Thompson Sampling: QR-DQN quantiles model aleatoric uncertainty.
        The distribution stays wide at convergence, so Thompson permanently gambles
        on rare jackpot tails. UCB uses Q_std as a bonus without lottery bias.
        """
        q_mean = q_all.mean(dim=-1)                               # (P, A)
        q_std  = q_all.std(dim=-1)                                # (P, A)
        ucb    = q_mean + self.ucb_beta.unsqueeze(1) * q_std     # (P, A)
        actions = ucb.argmax(dim=-1)                               # (P,)

        # Warmup: max(eps_base, eps_warmup * linear_decay).
        # Start at 0.5 not 1.0: eps=1.0 cuts Watkins trace ~87.5% of steps
        # with 8 actions, making Watkins effectively TD(0) for entire warmup.
        eps_warm = self.eps_warmup_start * max(
            0.0, 1.0 - self.total_frames / max(self.eps_warmup_frames, 1))
        eps = max(self.eps_base, eps_warm)

        rand_mask    = torch.rand(self.num_procs, device=self.device) < eps
        rand_actions = torch.randint(0, self.n_actions, (self.num_procs,),
                                     device=self.device)
        return torch.where(rand_mask, rand_actions, actions)

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def update_parameters(self, exps, apply_update=True):
        T = self.num_frames_per_proc
        P = self.num_procs
        N = self.num_quantiles
        A = self.n_actions

        _m = self.acmodel.module if is_ddp(self.acmodel) else self.acmodel

        # Step 1: live Q from stored h-states (Q head only, no CNN/LSTM re-run).
        # memories[t+1, :, :semi] = h after obs[t] (h_{t+1} = LSTM(obs[t], h_t)).
        # Using stored h avoids re-running the backbone just for target computation.
        with torch.no_grad():
            semi = self.semi_memory_size
            h_seq = torch.cat([
                self.memories[1:, :, :semi],
                self.memory[:, :semi].unsqueeze(0),
            ], dim=0)  # (T, P, D)
            live_q_all = _m.q_head(h_seq.view(T * P, semi)).view(T, P, A, N)

            # Q at boundary state s_T for Watkins bootstrap
            obs_T = self.preprocess_obss(self.obs, device=self.device)
            live_q_final, _ = self.acmodel(obs_T, self.memory * self.mask.unsqueeze(1))
            # live_q_final: (P, A, N)

        # Step 2: target net through full rollout to get Q at next states.
        # Separate LSTM state (R2D2): target net has different weights from live,
        # so its h-states differ even for the same observation sequence.
        with torch.no_grad():
            target_q_all_next, new_target_memory = self._compute_target_q()

        # Step 3: Watkins Q(lambda) targets - fixed for all epochs.
        # Not TD(0): without replay, TD(0) propagates terminal reward only 1 step
        # per rollout. Watkins propagates through full rollout for greedy steps,
        # cuts to 1-step at exploratory steps (Watkins off-policy correction).
        with torch.no_grad():
            watkins = self._compute_watkins_targets(
                live_q_all, live_q_final, target_q_all_next)
            # (T, P, N) -> P-major -> (P*T, N) to match exps P-major layout
            targets_flat = watkins.transpose(0, 1).reshape(P * T, N)

        # Step 4: epoch loop - qr_loss only, no policy/value/entropy terms.
        # 2-3 epochs max: without PPO clip, more epochs shifts Q enough to flip
        # argmax in many states and destabilize the next rollout.
        tau = _m.q_head.tau
        all_nan = True
        log_losses     = []
        log_q_means    = []
        log_q_stds     = []
        log_td_errs    = []
        log_grad_norms = []

        for _ in range(self.epochs):
            for inds in self._get_batches_starting_indexes():
                if self.recurrent:
                    memory = exps.memory[inds]

                batch_loss   = 0
                batch_q_mean = 0.0
                batch_q_std  = 0.0
                batch_td_err = 0.0

                for i in range(self.recurrence):
                    sb = exps[inds + i]

                    if self.recurrent:
                        h = memory * sb.mask
                        q_all, memory = self.acmodel(sb.obs, h)
                    else:
                        q_all, _ = self.acmodel(sb.obs)
                    # q_all: (B, A, N)

                    B = q_all.shape[0]
                    action_idx = sb.action[:, 0]                        # (B,) L=1
                    q_taken    = q_all[torch.arange(B), action_idx]    # (B, N)
                    target     = targets_flat[inds + i].detach()        # (B, N)

                    batch_loss += qr_loss(q_taken, target, tau)

                    with torch.no_grad():
                        batch_q_mean += q_taken.mean().item()
                        batch_q_std  += q_taken.std(dim=-1).mean().item()
                        batch_td_err += (target - q_taken).abs().mean().item()

                    if self.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                batch_loss   /= self.recurrence
                batch_q_mean /= self.recurrence
                batch_q_std  /= self.recurrence
                batch_td_err /= self.recurrence

                if apply_update:
                    self.optimizer.zero_grad()
                    batch_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.acmodel.parameters(), self.max_grad_norm).item()
                    if numpy.isfinite(grad_norm):
                        self.optimizer.step()
                else:
                    grad_norm = 0.0

                if numpy.isfinite(grad_norm):
                    all_nan = False

                log_losses.append(batch_loss.item())
                log_q_means.append(batch_q_mean)
                log_q_stds.append(batch_q_std)
                log_td_errs.append(batch_td_err)
                log_grad_norms.append(grad_norm)

        if all_nan:
            self._consecutive_nan_updates += 1
            if self._consecutive_nan_updates >= 5:
                raise RuntimeError(
                    f"Training diverged: {self._consecutive_nan_updates} consecutive "
                    f"updates with NaN/Inf gradients.")
        else:
            self._consecutive_nan_updates = 0

        # Step 5: EMA target update after all epochs (not per-epoch).
        # Per-epoch update would shift targets mid-training, narrowing the lag
        # that prevents Q drift.
        if apply_update:
            self._ema_update_target()
        self.target_memory = new_target_memory.detach()

        L = self.num_hierarchy_levels  # 1
        return {
            "q_loss":       numpy.mean(log_losses),
            "q_mean":       numpy.mean(log_q_means),
            "q_std":        numpy.mean(log_q_stds),
            "q_td_err":     numpy.mean(log_td_errs),
            "grad_norm":    numpy.mean(log_grad_norms),
            # Zero-fill PPO fields so existing display code does not break
            "entropy":      numpy.zeros(L),
            "value":        numpy.zeros(L),
            "policy_loss":  numpy.zeros(L),
            "value_loss":   numpy.zeros(L),
            "kl":           numpy.zeros(L),
            "clip_frac":    numpy.zeros(L),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_target_q(self):
        """Run target net sequentially through stored rollout.

        Returns target_q_all_next (T, P, A, N) where
        target_q_all_next[t] = Q_target(s_{t+1}).
        Also returns the updated target LSTM state for the next rollout.
        """
        T = self.num_frames_per_proc
        target_mem = self.target_memory.clone()
        q_steps    = []

        for i in range(T):
            obs_i = self.preprocess_obss(self.obss[i], device=self.device)
            # Zero memory at episode boundaries (same convention as live net).
            tm    = target_mem * self.masks[i].unsqueeze(1)
            q_i, target_mem = self.target_model(obs_i, tm)
            q_steps.append(q_i)  # Q(s_i)

        # Boundary: run on the observation after the last rollout step.
        obs_T = self.preprocess_obss(self.obs, device=self.device)
        tm    = target_mem * self.mask.unsqueeze(1)
        q_T, _ = self.target_model(obs_T, tm)

        q_steps = torch.stack(q_steps, dim=0)  # (T, P, A, N) = Q at s_0..s_{T-1}

        # Shift by one: target_q_all_next[t] = Q(s_{t+1}).
        # q_steps[t] = Q_target(s_t); we need Q_target(s_{t+1}):
        #   for t < T-1 : q_steps[t+1]
        #   for t = T-1 : q_T (boundary)
        target_q_all_next = torch.cat([q_steps[1:], q_T.unsqueeze(0)], dim=0)
        return target_q_all_next, target_mem

    @torch.no_grad()
    def _compute_watkins_targets(self, live_q_all, live_q_final,
                                  target_q_all_next):
        """Watkins Q(lambda) with Double Q bootstrap.

        Double Q: live net selects a* (decouples selection from evaluation).
        Watkins trace cut: exploratory steps break the eligibility trace,
        falling back to 1-step Q-learning (no lambda blending).

        Backward recursion (standard Watkins 1989):
          running_q = bootstrap at s_T
          for t = T-1 downto 0:
            eff_lambda = lambda * is_greedy_next[t]
            future_q   = (1 - eff_lambda) * next_q_greedy[t] + eff_lambda * running_q
            running_q  = r[t] + gamma * not_done[t] * future_q
            targets[t] = running_q
        """
        T, P, A, N = live_q_all.shape

        # not_done[t] = 1 if episode was NOT done after step t
        not_done = torch.cat([self.masks[1:], self.mask.unsqueeze(0)], dim=0)  # (T, P)

        rewards = self.rewards[:, :, 0]  # (T, P) - L=1
        actions = self.actions[:, :, 0]  # (T, P) - L=1

        # Double Q: live net selects greedy action at s_{t+1}
        live_q_next = torch.cat([live_q_all[1:], live_q_final.unsqueeze(0)], dim=0)
        next_a_star = live_q_next.mean(dim=-1).argmax(dim=-1)  # (T, P)

        # Double Q: target net evaluates Q at a* (prevents maximization bias)
        idx = next_a_star.view(T, P, 1, 1).expand(T, P, 1, N)
        next_q_greedy = target_q_all_next.gather(2, idx).squeeze(2)  # (T, P, N)

        # Watkins trace cut: was the action taken at t+1 the greedy action?
        # Boundary padding: force is_greedy_next[T-1]=True so the last step
        # always bootstraps cleanly (no spurious trace cut at rollout end).
        next_actions   = torch.cat([actions[1:], next_a_star[-1:]], dim=0)  # (T, P)
        is_greedy_next = (next_actions == next_a_star).float().unsqueeze(-1) # (T, P, 1)

        # Bootstrap from boundary state s_T
        running_q = next_q_greedy[-1]  # (P, N)
        targets   = torch.empty(T, P, N, device=self.device)

        for t in reversed(range(T)):
            nd       = not_done[t].unsqueeze(-1)   # (P, 1)
            r_t      = rewards[t].unsqueeze(-1)    # (P, 1)
            eff_lam  = self.watkins_lambda * is_greedy_next[t]
            future_q = (1.0 - eff_lam) * next_q_greedy[t] + eff_lam * running_q
            running_q = r_t + self.discount * nd * future_q
            targets[t] = running_q

        return targets  # (T, P, N)

    def _ema_update_target(self):
        """Polyak EMA: target_params = (1-tau)*target + tau*live."""
        live = self.acmodel.module if is_ddp(self.acmodel) else self.acmodel
        tau  = self.target_ema_tau
        for p_live, p_tgt in zip(live.parameters(), self.target_model.parameters()):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p_live.data)

    def _get_batches_starting_indexes(self):
        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)
        if self.batch_num % 2 == 1 and self.recurrence < self.num_frames_per_proc:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1
        num_indexes = self.batch_size // self.recurrence
        return [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]
