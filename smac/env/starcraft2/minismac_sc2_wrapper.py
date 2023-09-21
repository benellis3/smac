from smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np


class MiniSMACStarCraft2Env(StarCraft2Env):
    def __init__(self, **env_kwargs):
        super().__init__(**env_kwargs)
        self.minismac_unit_type_bits = 6
        self.other_unit_features = [
            "health",
            "position_x",
            "position_y",
            "last_action",
            "weapon_cooldown",
        ]
        self.other_unit_features += [
            f"unit_type_bits_{i}" for i in range(self.minismac_unit_type_bits)
        ]
        self.own_features = [
            "health",
            "position_x",
            "position_y",
            "weapon_cooldown",
        ]
        self.own_features += [
            f"unit_type_bits_{i}" for i in range(self.minismac_unit_type_bits)
        ]
        self.time_per_step = 1.0 / 16
        # map from MiniSMAC to SMAC movement actions
        self.actions_map = {
            0: 2,
            1: 4,
            2: 3,
            3: 5,
            4: 1,
        }

    def step(self, actions):
        actions = [int(a) for a in actions]
        # remap the actions, all else is the same
        for i, a in enumerate(actions):
            unit = self.get_unit_by_id(i)
            if unit.health > 0:
                # SMAC has an extra action.
                actions[i] = self.actions_map[a] if a in self.actions_map else a + 1
                # avail_actions = np.array(self.get_avail_agent_actions(i))
                # if avail_actions[actions[i]] != 1:
                #     actions[i] = np.random.choice(self.n_actions, 1, p=avail_actions / avail_actions.sum())
            else:
                actions[i] = 0
        return super().step(actions)

    def get_obs_agent(self, agent_id):
        ally_unit_obs = np.zeros(
            (self.n_agents - 1, len(self.other_unit_features))
        )
        enemy_unit_obs = np.zeros(
            (self.n_enemies, len(self.other_unit_features))
        )
        own_obs = np.zeros((1, len(self.own_features)))
        # iterate through the allies
        own_unit = self.get_unit_by_id(agent_id)
        x = own_unit.pos.x
        y = own_unit.pos.y
        sight_range = self.unit_sight_range(agent_id)
        al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
        if own_unit.health > 0:
            for i, al_id in enumerate(al_ids):
                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
                if dist < sight_range and al_unit.health > 0:
                    ally_unit_obs[i][0] = al_unit.health / al_unit.health_max
                    ally_unit_obs[i][1] = (al_x - x) / sight_range
                    ally_unit_obs[i][2] = (al_y - y) / sight_range
                    # MiniSMAC stores actions just as integers
                    ally_unit_obs[i][3] = np.argmax(self.last_action[al_id])
                    # no idea whether the time_per_step is right.
                    ally_unit_obs[i][4] = (
                        al_unit.weapon_cooldown * self.time_per_step
                    )
                    # TODO assumes that this is a marine! Need to fix if other units
                    # introduced!
                    ally_unit_obs[i][5] = 1
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)
                if dist < sight_range and e_unit.health > 0:
                    enemy_unit_obs[e_id][0] = e_unit.health / e_unit.health_max
                    enemy_unit_obs[e_id][1] = (e_x - x) / sight_range
                    enemy_unit_obs[e_id][2] = (e_y - y) / sight_range
                    # space where the enemy action would be, but we don't
                    # have those cos starcraft
                    enemy_unit_obs[e_id][3] = 0
                    enemy_unit_obs[e_id][4] = (
                        e_unit.weapon_cooldown * self.time_per_step
                    )
                    # TODO again assume that this is a marine
                    enemy_unit_obs[e_id][5] = 1
            unit = self.get_unit_by_id(agent_id)
            own_obs[0][0] = unit.health / unit.health_max
            own_obs[0][1] = x / self.map_x
            own_obs[0][2] = y / self.map_y
            own_obs[0][3] = unit.weapon_cooldown * self.time_per_step
            # TODO make assumption this is a marine
            own_obs[0][4] = 1

        agent_obs = np.concatenate(
            (
                ally_unit_obs.flatten(),
                enemy_unit_obs.flatten(),
                own_obs.flatten(),
            )
        )

        # concatenate them
        return agent_obs

    def get_obs_size(self):
        # add up all the non-me unit features
        # add up all the me unit features
        # add together
        return len(self.other_unit_features) * (
            self.n_agents + self.n_enemies - 1
        ) + len(self.own_features)

    def get_avail_agent_actions(self, agent_id):
        avail_actions_old = super().get_avail_agent_actions(agent_id)
        avail_actions = np.zeros((self.n_actions - 1,))
        for k, v in self.actions_map.items():
            avail_actions[k] = avail_actions_old[v]
        avail_actions[5:] = avail_actions_old[6:]
        return avail_actions