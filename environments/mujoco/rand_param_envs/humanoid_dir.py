raise NotImplementedError("Use humanoid_dir2 instead")
import numpy as np

from environments.mujoco.rand_param_envs.base import MetaEnv
from environments.mujoco.rand_param_envs.gym.envs.mujoco.humanoid import (
    HumanoidEnv,
    mass_center,
)

class HumanoidDirEnv(HumanoidEnv, MetaEnv):

    def __init__(self, tasks=None, n_tasks=2):
        if tasks is None:
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self.reset_task(0)
        super(HumanoidDirEnv, self).__init__()

    def step(self, action):
        pos_before = np.copy(mass_center(self.model, self.sim)[:2])
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[:2]

        alive_bonus = 5.0
        data = self.sim.data
        goal_direction = (np.cos(self._goal), np.sin(self._goal))
        lin_vel_cost = 0.25 * np.sum(goal_direction * (pos_after - pos_before)) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))

        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost,
                                                   reward_quadctrl=-quad_ctrl_cost,
                                                   reward_alive=alive_bonus,
                                                   reward_impact=-quad_impact_cost)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def get_all_task_idx(self):
        return range(0, 50)

    def get_all_eval_task_idx(self):
        return range(50, 55)

    # not sure if I should implement this or set_task..
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector

    def set_task(self, task):
        self._task = task #self.tasks[idx]
        self._goal = self._task['goal'] # assume parameterization of task by single vector

    def get_task(self):
        return self._task

    def sample_tasks(self, num_tasks):
        # velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': d} for d in directions]
        return tasks
