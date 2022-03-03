from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent.parent.parent / "supervised_obs_and_states"


def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10000

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        obs_list = []
        state_list = []
        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward
            obs_list.append(obs)
            state_list.append(state)

        stacked_obs_list = np.stack(obs_list)
        stacked_state_list = np.stack(state_list)
        np.save(DIR / f"episode_{e}_obs.npy", stacked_obs_list)
        np.save(DIR / f"episode_{e}_state.npy", stacked_state_list)
        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()
