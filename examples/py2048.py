import argparse
import time
import logging
import json

from tensorforce import TensorForceError
from tensorforce.execution import Runner
from tensorforce.agents import Agent, dqn_agent
from tensorforce.contrib.py2048 import Game2048


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--agent-config', help="Agent configuration file")
    parser.add_argument('-n', '--network-spec', default=None, help="Network specification file")
    parser.add_argument('-s', '--summary-spec', default=None, help="Summary tensorboard specification file")

    args = parser.parse_args()

    if args.agent_config is not None:
        with open(args.agent_config, 'r') as fp:
            agent_config = json.load(fp=fp)
    else:
        raise TensorForceError("No agent configuration provided.")

    if args.network_spec is not None:
        with open(args.network_spec, 'r') as fp:
            network_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    if args.summary_spec is not None:
        with open(args.summary_spec, 'r') as fp:
            summary_spec = json.load(fp=fp)
    else:
        network_spec = None
        logger.info("No network configuration provided.")

    logger.info("Start training")

    environment = Game2048()

    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states_spec=environment.states,
            actions_spec=environment.actions,
            network_spec=network_spec,
            summary_spec=summary_spec
        )
    )

    # agent = dqn_agent.DQNAgent(
    #     states_spec=environment.states,
    #     actions_spec=environment.actions,
    #     network_spec=network_spec,
    #     summary_spec=dict(directory="./board/",
    #                       steps=50,
    #                       labels=['losses',
    #                               'variables']
    #                       ),
    # )

    # network_spec = [
    #     dict(type='dense', size=4, activation='tanh'),
    #     dict(type='dense', size=4, activation='tanh')
    # ]
    #
    # agent = PPOAgent(
    #     states_spec=environment.states,
    #     actions_spec=environment.actions,
    #     network_spec=network_spec,
    #     batch_size=4096,
    #     # Agent
    #     states_preprocessing_spec=[dict(type='flatten')],
    #     explorations_spec=dict(
    #         type="epsilon_decay",
    #         initial_epsilon=1.0,
    #         final_epsilon=0.1,
    #         timesteps=50000000
    #     ),
    #     reward_preprocessing_spec=None,
    #     # BatchAgent
    #     keep_last_timestep=True,
    #     # PPOAgent
    #     step_optimizer=dict(
    #         type='adam',
    #         learning_rate=1e-3
    #     ),
    #     optimization_steps=10,
    #     # Model
    #     scope='ppo',
    #     discount=0.99,
    #     # DistributionModel
    #     distributions_spec=None,
    #     entropy_regularization=0.01,
    #     # PGModel
    #     baseline_mode=None,
    #     baseline=None,
    #     baseline_optimizer=None,
    #     gae_lambda=None,
    #     # PGLRModel
    #     likelihood_ratio_clipping=0.2,
    #     summary_spec=dict(directory="./board/",
    #                       steps=50,
    #                       # Add custom keys to export
    #                       labels=['configuration',
    #                               'gradients_scalar',
    #                               'gradients_histogram',
    #                               'regularization',
    #                               'inputs',
    #                               'losses',
    #                               'variables']
    #                       ),
    #     distributed_spec=None
    # )

    runner = Runner(
        agent=agent,
        environment=environment,
        repeat_actions=1
    )

    def episode_finished(r):
        if r.episode % 250 == 0:
            sps = r.timestep / (time.time() - r.start_time)
            logger.info("Finished episode {ep} after {ts} timesteps. Steps Per Second {sps}".format(ep=r.episode, ts=r.timestep, sps=sps))
            logger.info("Episode reward: {}".format(r.episode_rewards[-1]))
            logger.info("Episode timesteps: {}".format(r.episode_timestep))
            logger.info("Episode largest tile: {}".format(r.environment.largest_tile))
            logger.info("Average of last 500 rewards: {}".format(sum(r.episode_rewards[-500:]) / 500))
            logger.info("Average of last 100 rewards: {}".format(sum(r.episode_rewards[-100:]) / 100))
        return True

    runner.run(
        timesteps=5000000,
        episodes=10000,
        max_episode_timesteps=500000,
        deterministic=False,
        episode_finished=episode_finished
    )
    runner.close()

if __name__ == '__main__':
    main()