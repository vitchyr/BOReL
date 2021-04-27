import numpy as np

from environments.mujoco.ant_multitask_base import MultitaskAntEnv


class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task=None, n_tasks=2, max_episode_steps=200,
                 forward_backward=False, direction_in_degrees=True, **kwargs):
        if task is None:
            task = {}
        self.forward_backward = forward_backward
        self.direction_in_degrees = direction_in_degrees
        self._max_episode_steps = max_episode_steps

        super(AntDirEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        if self.direction_in_degrees:
            goal = self._goal / 180 * np.pi
        else:
            goal = self._goal
        direct = (np.cos(goal), np.sin(goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            assert num_tasks == 2
            if self.direction_in_degrees:
                directions = np.array([0., 180])
            else:
                directions = np.array([0., np.pi])
        else:
            if self.direction_in_degrees:
                directions = np.random.uniform(0., 360, size=(num_tasks,))
            else:
                directions = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': desired_dir} for desired_dir in directions]
        return tasks

    def set_goal(self, goal):
        self._goal = np.asarray(goal)
