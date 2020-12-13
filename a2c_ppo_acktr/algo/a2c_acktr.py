import torch
import torch.nn as nn
import torch.optim as optim

class A2C():
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None):

        self.num_steps = args.num_steps
        self.shift_len = args.shift_len
        self.actor_critic = actor_critic
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        """Called once per episode"""
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        epi_len, num_processes, _ = rollouts.rewards.size()

        self.optimizer.zero_grad()

        last_segment_start = (200 - self.num_steps)  # epi_len - num_steps
        num_shifts = last_segment_start // self.shift_len
        segment_starts = range(0, last_segment_start, self.shift_len)

        for seg_start in segment_starts:
            seg_stop = seg_start + self.num_steps
            values, action_log_probs, dist_entropy, _ = \
                self.actor_critic.evaluate_actions(
                    rollouts.obs[seg_start:seg_stop].view(-1, *obs_shape),
                    rollouts.recurrent_hidden_states[seg_start].view(
                        -1, self.actor_critic.recurrent_hidden_state_size),
                    rollouts.masks[seg_start:seg_stop].view(-1, 1),
                    rollouts.actions[seg_start:seg_stop].view(-1, action_shape))

            values = values.view(self.num_steps, num_processes, 1)
            action_log_probs = action_log_probs.view(self.num_steps, num_processes, 1)

            advantages = rollouts.returns[seg_start:seg_stop] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(advantages.detach() * action_log_probs).mean()


            (value_loss * self.value_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()

            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()
