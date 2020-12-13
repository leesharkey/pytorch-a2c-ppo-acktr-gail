import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.episode_len = num_steps

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.p_dists = torch.zeros(num_steps + 1, num_processes, 2)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.episode_step_idxs = torch.zeros(num_steps)
        self.global_step_idxs  = torch.zeros(num_steps)

        self.num_steps = num_steps
        self.step = 0
        self.episode_step = 0
        self.global_step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.p_dists = self.p_dists.to(device)

        self.episode_step_idxs = self.episode_step_idxs.to(device)
        self.global_step_idxs  = self.global_step_idxs.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, p_dists):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.p_dists[self.step + 1].copy_(p_dists)
        if self.step == 0:
            self.p_dists[self.step].copy_(p_dists)

        self.episode_step_idxs[self.step] = self.episode_step
        self.global_step_idxs[self.step]  = self.global_step

        self.step = (self.step + 1) % self.num_steps
        self.episode_step = (self.episode_step + 1) % 200
        self.global_step += 1

    def save_experimental_data(self, save_dir):
        save_dict = {'obs': self.obs.clone().cpu().numpy(),
                     'recurrent_hidden_states': self.recurrent_hidden_states.clone().cpu().numpy(),
                     'rewards': self.rewards.clone().cpu().numpy(),
                     'value_preds': self.value_preds.clone().cpu().numpy(),
                     'returns': self.returns.clone().cpu().numpy(),
                     'action_log_probs': self.action_log_probs.clone().cpu().numpy(),
                     'actions': self.actions.clone().cpu().numpy(),
                     'masks': self.masks.clone().cpu().numpy(),
                     'bad_masks': self.bad_masks.clone().cpu().numpy(),
                     'p_dists': self.p_dists.clone().cpu().numpy(),
                     'episode_step_idxs': self.episode_step_idxs.clone().cpu().numpy(),
                     'global_step_idxs': self.global_step_idxs.clone().cpu().numpy()}
        torch.save(save_dict, save_dir + '/' + \
                   'step_{}.pt'.format(self.global_step))

    def after_update(self, reset_hxs=False):
        # TODO check whether the new objects lee made (pdists, step_idxs)
        #  are reset in the correct way here
        self.obs[0].copy_(self.obs[-1])

        if reset_hxs and self.masks[-1].sum() == 0.:
            self.recurrent_hidden_states[0].copy_(
                torch.zeros_like(self.recurrent_hidden_states[-1]))
        else:
            self.recurrent_hidden_states[0].copy_(
                self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.p_dists[0].copy_(self.p_dists[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]  # 0 if end of episode
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = (self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    self.returns[step] = self.returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]
