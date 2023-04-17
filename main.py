import numpy as np
import torch
from torch.optim import Adam
from models import LIMIT, ReplayMemory, AlignHuman
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def train_LIMIT(model: LIMIT, memory: ReplayMemory, batch_size: int,
                interface_optim: Adam, human_optim: Adam,
                n_timesteps=10, epochs=50):
    stdev = len(memory) / 5
    net_loss0 = 0.
    net_loss1 = 0.
    for _ in range(epochs):
        states, actions, _, thetas = memory.weighted_sample(batch_size, stdev)
        state_batch = torch.FloatTensor(states)
        s_clone = state_batch.clone()
        theta_batch = torch.FloatTensor(thetas)
        action_batch = torch.FloatTensor(actions)
        states_actions = torch.FloatTensor([])
        for _ in range(n_timesteps):
            actions = model(s_clone, theta_batch)
            states_actions = torch.cat((states_actions,
                                        s_clone, actions), 1)
            s_clone += actions
        theta_hat = model.decoder(states_actions)
        action_hat = model(state_batch, theta_batch)
        loss0 = model.mse_loss(theta_hat, theta_batch)
        loss1 = model.mse_loss(action_hat, action_batch)
        loss = loss1 + loss0

        interface_optim.zero_grad()
        human_optim.zero_grad()
        loss.backward()
        interface_optim.step()
        human_optim.step()

        net_loss0 += loss0.item()
        net_loss1 += loss1.item()
    return (net_loss0, net_loss1)


def main(args) -> None:
    n_episodes = args.episodes
    n_timesteps = 10
    batch_size = 64
    human_memory = ReplayMemory(capacity=30)
    interface_memory = ReplayMemory(capacity=5000)
    human = AlignHuman()
    interface = LIMIT()
    disting_optim = Adam([{"params":interface.decoder1.parameters()},
                          {"params":interface.decoder2.parameters()},
                          {"params":interface.decoder3.parameters()},
                          {"params":interface.interface1.parameters()},
                          {"params":interface.interface2.parameters()},
                          {"params":interface.interface3.parameters()},
                          ],
                         lr=0.001)
    convey_optim = Adam([{"params":interface.human1.parameters()},
                         {"params":interface.human2.parameters()},
                         {"params":interface.human3.parameters()},
                         ],
                        lr=0.001)
    error = []
    # note that the human remembers each interaction and only knows
    # theta after the interaction has finished, therefore they store
    # the (s, a, x, t) tuple as an entire interaction, not a datapoint
    for ep in range(n_episodes):
        states = []
        actions = []
        signals = []
        thetas = []
        theta = 10 * (2 * np.random.rand(2) - 1)
        state = 10 * (2 * np.random.rand(2) - 1)
        for _ in range(n_timesteps):
            state_theta = torch.cat((torch.FloatTensor(state),
                                     torch.FloatTensor(theta)))
            signal = interface.interface_policy(state_theta).detach().numpy()
            action = human(signal)
            interface_memory.push(state, action, signal, theta)
            states.append(state)
            actions.append(action)
            signals.append(signal)
            thetas.append(theta)
            next_state = state + action
            state = next_state
            if args.online:
                train_LIMIT(interface, interface_memory, int(batch_size / 4), disting_optim, convey_optim, epochs=4)
        human_memory.push(states, actions, signals, thetas)
        l_dist, l_conv = train_LIMIT(interface, interface_memory, batch_size, disting_optim, convey_optim)
        human.optimize(human_memory, n_samples=15, n_scale=2, n_angle=12)
        error.append(np.linalg.norm(state - theta))
        print(f"Ep: {ep}, L_distinguish: {np.round(l_dist, 2)}, L_convey: {np.round(l_conv, 2)}, "
            f"Error: {np.round(np.linalg.norm(state - theta), 2)}")
    N = 10
    ma_mean = np.convolve(error, np.ones(N) / N, mode='valid')
    std_error = np.std(ma_mean)
    plt.plot(ma_mean)
    plt.plot([0, n_episodes], [np.mean(error), np.mean(error)])
    plt.xlabel("Episodes, n")
    plt.ylabel("Error")
    plt.title("Moving average of episode error")
    plt.fill_between(np.arange(len(ma_mean)), ma_mean - std_error, ma_mean + std_error, alpha=0.2)
    plt.show()
    return


if __name__ == "__main__":
    np.random.seed(8386)
    torch.manual_seed(8386)
    parser = ArgumentParser()
    parser.add_argument("--online", action="store_true", help="pass this flag to enable online training")
    parser.add_argument("--episodes", type=int, help="number of episodes to play with AlignHuman", default=200)
    main(parser.parse_args())
