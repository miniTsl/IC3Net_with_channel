from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        # 呵呵，policy_net和self.policy_net的地址是一样的，所以self.optimizer和self.params中的包含的参数其实是一样的
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]

    def get_episode(self, epoch):
        '''
        input:epoch
        
        output:
            episode: a list of Transitions \n
            stat: a dict:
                    {
                        'reward':array([  0.  ,  -1.2 , -12.56,   0.  ,   0.  ,   0.  ,   0.  , -11.9 ,
                                0.  ,   0.  ])
                        'num_steps':40
                        'steps_taken':40
                        'success':0
                        'add_rate':0.02
                    }
        '''
        episode = []
        
        # Get the names and default values of a function's parameters.
        reset_args = getargspec(self.env.reset).args
        
        # episode初始化
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)   # state: [1, 10, 61]
        else:
            state = self.env.reset()
        
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)    # [1,10,128]

        for t in range(self.args.max_steps):
            misc = dict()
            
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])   # batch_size = 1
                    # ([10,128], [10,128]), full of zero, [prev_hidden, prev_cell]
                
                x = [state, prev_hid]
                action_out, value, prev_hid = self.policy_net(x, info)

                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value = self.policy_net(x, info)

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            next_state, reward, done, info = self.env.step(actual)
            # next_state: [1, 10, 61]
            # reward: (10,)
            # done: bool
            # info: dic
                    
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]

            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]   # add rewards up
            # D.get(k[,d]) -> D[k] if k in D, else d. d defaults to None.
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            # action_out: [1,10,2]
            # value: [10,1]
            episode.append(trans)
            
            state = next_state
            if done:
                break
        
        stat['num_steps'] = t + 1   # 就是max_steps=40
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

        # 这里获得success
        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        '''
        input:
            batch: a single Transition but each named element is a tuple filled with all corresponding elemtns from each Transition in batch from get_episode()
        output:
            stat: a dictionary that contains action_loss, value_loss, entropy
        what's done in this function: compute loss and let loss backward~
        '''
        stat = dict()
        num_actions = self.args.num_actions # [2]
        dim_actions = self.args.dim_actions # 1

        n = self.args.nagents   # 10
        batch_size = len(batch.state)   # 520

        rewards = torch.Tensor(batch.reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)  # [520, 10, 1]

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]
        # misc中就只有alive_mask
        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        # alive_masks：torch.Size([5200])
        coop_returns = torch.Tensor(batch_size, n)
        ncoop_returns = torch.Tensor(batch_size, n)
        returns = torch.Tensor(batch_size, n)
        deltas = torch.Tensor(batch_size, n)
        advantages = torch.Tensor(batch_size, n)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        loss = action_loss + self.args.value_coeff * value_loss

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy


        loss.backward()

        return stat

    def run_batch(self, epoch):
        '''
        input: epoch
        output:
            batch: list of Transitions from every step in batch_size = 500
            states: a dict
                {
                    
                    'num_episodes':13
                    'reward':array([ -57.08,   -6.48,  -66.39, -132.6 ,   -5.61,   -4.83, -109.06,
                            -0.78,   -9.72,  -63.81])
                    'num_steps':520
                    'steps_taken':520
                    'success':9
                    'add_rate':0.25999999999999995
                }
        '''
        batch = []  # list of Transitions ffrom
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch) # episode中全是Transition
            # episode_stat中包含每个episode的步数num_steps40、steps_taken40、总rewards
            merge_stat(episode_stat, self.stats)
            # self.stats中包含500步内的总步数（520）、steps_taken（500）、总rewards
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)    # 一个batch的总步数,其实是520
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        '''
        input:
            epoch
        output:
            stat: a dict.
                {
                    'num_episodes':13
                    'reward':array([ -2.83,  -5.51,  -4.41, -21.34,  -4.55,  -2.13,  -1.77,  -3.62,
                        -17.14,  -3.81])
                    'num_steps':520
                    'steps_taken':520
                    'success':12
                    'add_rate':0.25999999999999995
                    'action_loss':201.7538625441944
                    'value_loss':407.09509556341595
                    'entropy':3299.564348539202
                }
        '''
        batch, stat = self.run_batch(epoch) 
        # batch is a single Transition but each named element is a tuple filled with all corresponding elemtns from each Transition in the 'batch' from get_episode()
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']   # 最后stat['num_steps'] = 500
        self.optimizer.step()

        return stat # the information for an entire batch_size = 500

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
