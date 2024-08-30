import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch
import os

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

path_ = os.getcwd()
local = path_.find('Simon') > 0
if local:
    path_ = '/Users/Simon/Documents/Projects/EWF/Research/PhD/Ergodicity Economics/IOxEE'
else:
    path_ = 'conglomerate'

action_std = 0.1  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
action_std_init = 0.6

random_seed = 0  # set random seed if required (0 = no random seed)
checkpoint_path = '{}/programs/models'.format(path_)

class ActorCritic(nn.Module):
    def __init__(self, action_std_init=0.001, state_dim=4, action_dim=1, has_continuous_action_space=True):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, action_dim), nn.Tanh())
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, action_dim), nn.Softmax(dim=-1))
        # critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        # action = torch.sigmoid(action)
        action = (torch.tanh(action) + 1) / 2
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()


class Firm:
    def __init__(self, has_continuous_action_space=True, action_std_init=0.001, state_dim=4, action_dim=1):

        self.policy = ActorCritic(action_std_init, state_dim, action_dim, has_continuous_action_space).to(device)

        self.policy_old = ActorCritic(action_std_init, state_dim, action_dim, has_continuous_action_space).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.load(checkpoint_path=checkpoint_path + '/agent_' + '1' + '.pth')

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        return action.detach().cpu().numpy().flatten()

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


actor = Firm()

inc = 1/100
max_level = 1
price_states = np.arange(inc, max_level + inc, inc)
share_states = np.arange(inc, max_level + inc, inc)

combinations = np.array(np.meshgrid(price_states, share_states, price_states, price_states)).T.reshape(-1, 4)
prices = np.zeros(combinations.shape[0])

for price in tqdm(range(combinations.shape[0])):
    prices[price] = actor.select_action(combinations[price, :])

side_length = price_states.shape[0]
tensor_shape = (side_length, side_length, side_length, side_length)
result_tensor = np.full(tensor_shape, -1)

for combo, output in zip(combinations, prices):
    indices = tuple(int(val * 100 - 1) for val in combo)
    result_tensor[indices] = output

np.save(checkpoint_path + '/price_tensor.npy', result_tensor)
