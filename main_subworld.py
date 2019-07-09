import neat
import numpy as np
import subworldgym

import run_neat_base


def eval_single_genome(genome, genome_config):
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    for i in range(run_neat_base.n):
        # print("--> Starting new episode")
        observation = run_neat_base.env.reset()

        action = eval_network(net, observation)

        done = False

        while not done:

            # env.render()

            observation, reward, done, info = run_neat_base.env.step(action)

            # print("\t Reward {}: {}".format(t, reward))

            action = eval_network(net, observation)

            total_reward += reward

            if done:
                # print("<-- Episode finished after {} timesteps".format(t + 1))
                print("$$$ Reward was: {}".format(total_reward))
                break

    return total_reward / run_neat_base.n


def eval_network(net, net_input):
    assert (len(net_input == 85))
    net_input = net_input.ravel()
    # print(net_input)
    z = net.activate(net_input)
    # print("Taking max:")

    result = np.argmax(z)
    # print("Result:")
    # print(result)
    # raise False
    # assert (result == 0 or result == 1)

    return result

def main():
    run_neat_base.run(eval_network,
                      eval_single_genome,
                      environment_name="SubWorld_empty_atari-v0")


if __name__ == '__main__':
    main()
