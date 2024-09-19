import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

model_tensor = np.load('models/price_tensor.npy')
device = 'cpu'


def to_index(value, max_val, size):
    return min(int(value / max_val * (size - 1)), size - 1)


class ActorCritic(nn.Module):
    def __init__(self, action_std_init, state_dim=4, action_dim=1, has_continuous_action_space=True):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Tanh())
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, action_dim), nn.Softmax(dim=-1))
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
    def __init__(self, market, number, market_id, steps, lookback, has_continuous_action_space=True, action_std_init=0.001, state_dim=4, action_dim=1):
        self.id = number
        self.market_id = market_id
        self.home_market = market
        self.states = np.ones(steps + 1)
        self.markets = [market]
        self.conglomerate = [self.id]
        self.outside_profits = np.ones(lookback)
        self.entered = None
        self.conglomerate_id = None

        self.policy = ActorCritic(action_std_init, state_dim, action_dim, has_continuous_action_space).to(device)

        self.policy_old = ActorCritic(action_std_init, state_dim, action_dim, has_continuous_action_space).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.load(checkpoint_path='models/agent_1.pth')

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        return action.detach().cpu().numpy().flatten()

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def select_action_matrix(self, state):
        state = np.round(state, decimals=2)
        indices = tuple(int(state_ * 100 - 1) for state_ in state)
        try:
            action = model_tensor[indices]
        except IndexError:
            print('!')
        return action


def select_action_matrix(state):
    state = np.round(state, decimals=2)
    indices = tuple(int(state_ * 100 - 1) for state_ in state)
    try:
        action = model_tensor[indices]
    except IndexError:
        print('!')
    return action
