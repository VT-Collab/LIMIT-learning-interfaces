import numpy as np
import torch
import torch.nn as nn
from scipy.stats import halfnorm


# uniformally distributes network weights at initialization
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


# The LIMIT algorithm (see Section 3).
# For best performance, optimize the decoder and human models
# seperately. The decoder and interface policy *must* be trained
# using the same optimizer.
class LIMIT(nn.Module):
    # @param {theta,state,action,signal}_dim : environment parameters
    # @param hidden_dim1 : size of each hidden layer of the interface
    #                      and human policy networks
    # @param hidden_dim2 : size of the hidden layers of the decoder
    # @param timesteps   : timesteps per interaction
    def __init__(self, theta_dim=2, state_dim=2, action_dim=2, signal_dim=2,
                 hidden_dim1=16, hidden_dim2=64, timesteps=10) -> None:
        super(LIMIT, self).__init__()
        self.interface1 = nn.Linear(state_dim + theta_dim, hidden_dim1)
        self.interface2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.interface3 = nn.Linear(hidden_dim1, signal_dim)

        self.human1 = nn.Linear(state_dim + signal_dim, hidden_dim1)
        self.human2 = nn.Linear(hidden_dim1, hidden_dim1)
        self.human3 = nn.Linear(hidden_dim1, action_dim)

        self.decoder1 = nn.Linear(timesteps * (state_dim + action_dim), hidden_dim2)
        self.decoder2 = nn.Linear(hidden_dim2, hidden_dim2)
        self.decoder3 = nn.Linear(hidden_dim2, theta_dim)

        self.apply(weights_init_)
        self.mse_loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # This is Equation 2 from our paper
    # @param state_theta : concatentated (state, theta)
    def interface_policy(self, state_theta):
        x = self.interface1(state_theta)
        x = self.interface2(x)
        x = self.interface3(x)
        return self.tanh(x)

    # This is Equation 3 from our paper
    # and is only used for training purposes
    # @param state_signal : concatenated (state, signal)
    def human_policy(self, state_signal):
        x = self.human1(state_signal)
        x = self.human2(x)
        x = self.human3(x)
        return self.tanh(x)

    # This is the decoder from Equation 14
    # and is only used for training purposes
    # @param state_actions : concatenated (states, actions)
    #        per timestep, should be of size
    #        n_timesteps * (state_dim + action_dim)
    def decoder(self, state_actions):
        x = self.decoder1(state_actions)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x

    # Outputs prediction of human response to the current
    # (state, theta) input to the interface.
    # @param state : state
    # @param theta : theta, the hidden information
    def forward(self, state, theta):
        state_theta = torch.cat((state, theta), 1)
        signals = self.interface_policy(state_theta)
        state_signal = torch.cat((state, signals), 1)
        return self.human_policy(state_signal)


# This class implements a replay buffer with weighted and
# uniform sampling capabilites.
class ReplayMemory:
    def __init__(self, capacity=1000):
        self.capacity = int(capacity)
        self.position = 0
        self.size = 0
        self.buffer = np.zeros(self.capacity, dtype=tuple)

    # push data to the buffer.
    # @param s : state
    # @param a : action
    # @param x : signal
    # @param theta : hidden information
    def push(self, s, a, x, theta):
        self.buffer[self.position] = (s, a, x, theta)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # randomly sample data from the buffer, sampled from a
    # uniform distribution. duplicate samples are possible
    # @param batch_size : number of datapoints to sample
    def sample(self, batch_size):
        batch = np.random.choice(self.buffer[0:self.size], batch_size)
        states, actions, xs, thetas = map(np.stack, zip(*batch))
        return states, actions, xs, thetas

    # randomly sample data from the buffer, sampled from a
    # halfnorm distribution centered at the current position.
    # duplicate samples are possible.
    # @param batch_size : number of datapoints to sample
    # @param stdev : standard deviation, should be proportional
    #                to the number of items in the buffer
    #                for best performance
    def weighted_sample(self, batch_size, stdev=10.):
        weights = np.array(halfnorm.pdf(np.arange(0, self.capacity), loc=0, scale=stdev))
        weights = weights.take(np.arange(self.position - self.capacity, self.position))[::-1][0:self.size]
        weights /= np.sum(weights)
        batch = np.random.choice(self.buffer[0:self.size], batch_size, p=weights)
        states, actions, xs, thetas = map(np.stack, zip(*batch))
        return states, actions, xs, thetas

    # length function, returns number of datapoints in the buffer
    def __len__(self):
        return self.size


# Implementation of Align Human from Section 5
class AlignHuman():
    def __init__(self) -> None:
        self.angle = np.pi * np.random.rand()
        self.scale = 2 * np.random.rand() - 1

    # helper function, return rotation matrix Rot(angle)
    # @param angle : input in radians
    def _rot_mat(self, angle: float) -> np.ndarray:
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    # Implementation of AlignHuman in Two-DoF environment
    # from Section 5.2. Note that AlignHuman is state-independent
    # @param signal : input signal from interface
    # @param params : when not None, params[0] is used as "scale"
    #                 and params[1] is used as "angle"
    def __call__(self, signal: np.ndarray, params=None):
        if params:
            action = params[0] * (self._rot_mat(params[1]) @ signal)
        else:
            action = self.scale * (self._rot_mat(self.angle) @ signal)
        if np.linalg.norm(action) > 1.0:
            return action / np.linalg.norm(action)
        return action

    # Trivial implementation of an optimizer for the AlignHuman.
    # Given a replay buffer of past interactions, optimize finds the best
    # scale and angle to use to minimize error between final state and theta.
    # @param memory : replay buffer of past interactions. Note that
    #                 this buffer's elements must be an array of states,
    #                 signals, etc. as opposed to single datapoints
    # @param n_scale : determines resolution of the scale search space
    # @param n_angle : determines resolution of the angle search space
    # @param n_timesteps : number of timesteps per interaction
    # @param n_samples : maximum number of interactions to sample from
    #                    the replay buffer
    def optimize(self, memory: ReplayMemory,
                 n_scale=10, n_angle=10, n_timesteps=10, n_samples=10) -> None:
        error = np.inf
        x_f = [self.scale, self.angle]
        angles = np.linspace(0., np.pi, n_angle)
        scales = np.linspace(-1., 1., n_scale)
        for angle in angles:
            for scale in scales:
                param = (scale, angle)
                current_error = 0.
                for _ in range(min(n_samples, len(memory))):
                    states, _, signals, thetas = memory.sample(1)
                    states = states.reshape(n_timesteps, 2)
                    signals = signals.reshape(n_timesteps, 2)
                    thetas = thetas.reshape(n_timesteps, 2)
                    state = np.copy(states[0])
                    theta = thetas[0]
                    for timestep in range(n_timesteps):
                        signal = signals[timestep]
                        a = self.__call__(signal, param)
                        state += a
                    current_error += np.linalg.norm(state - theta)
                if current_error < error:
                    x_f = param
                    error = current_error
        self.scale, self.angle = x_f
        return

