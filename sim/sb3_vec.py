"""Stable-Baselines3 VecEnv over the JAX/MJX envs, so SB3's algorithms
(PPO, SAC, TD3, ...) train on the same GPU-simulated environments as Brax.

    from sb3_vec import JaxVecEnv
    venv = JaxVecEnv(make_env("reach"), num_envs=256)
    PPO("MlpPolicy", venv).learn(1_000_000)
"""
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


class JaxVecEnv(VecEnv):
    def __init__(self, env, num_envs=256, seed=0, episode_length=None):
        self.env = env
        self.T = episode_length or env.episode_length
        obs_space = gym.spaces.Box(-np.inf, np.inf, (env.observation_size,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (env.action_size,), np.float32)
        super().__init__(num_envs, obs_space, act_space)
        self._reset = jax.jit(jax.vmap(env.reset))
        self._step = jax.jit(jax.vmap(env.step))

        def reset_where(state, done, rng):
            fresh = jax.vmap(env.reset)(jax.random.split(rng, num_envs))
            pick = lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), b, a)
            return jax.tree.map(pick, state, fresh)

        self._reset_where = jax.jit(reset_where)
        self._rng = jax.random.PRNGKey(seed)
        self._actions = None
        self.state = None
        self.t = np.zeros(num_envs, int)
        self.ep_ret = np.zeros(num_envs)

    def _split(self):
        self._rng, r = jax.random.split(self._rng)
        return r

    def reset(self):
        self.state = self._reset(jax.random.split(self._split(), self.num_envs))
        self.t[:] = 0
        self.ep_ret[:] = 0
        return np.array(self.state.obs, np.float32)

    def step_async(self, actions):
        self._actions = jnp.asarray(actions, jnp.float32)

    def step_wait(self):
        s = self._step(self.state, self._actions)
        self.t += 1
        rew = np.array(s.reward, np.float32)
        term = np.asarray(s.done) > 0.5
        trunc = self.t >= self.T
        done = term | trunc
        obs = np.array(s.obs, np.float32)
        self.ep_ret += rew
        infos = [{} for _ in range(self.num_envs)]
        if done.any():
            metrics = {k: np.asarray(v) for k, v in s.metrics.items()}
            for i in np.nonzero(done)[0]:
                infos[i] = {"terminal_observation": obs[i], "TimeLimit.truncated": bool(trunc[i] and not term[i]),
                            "episode": {"r": float(self.ep_ret[i]), "l": int(self.t[i])},
                            **{f"m_{k}": float(v[i]) for k, v in metrics.items()}}
            s = self._reset_where(s, jnp.asarray(done), self._split())
            obs = np.array(s.obs, np.float32)
            self.t[done] = 0
            self.ep_ret[done] = 0
        self.state = s
        return obs, rew, done, infos

    def close(self):
        pass

    def get_attr(self, name, indices=None):
        return [getattr(self.env, name)] * self.num_envs

    def set_attr(self, name, value, indices=None):
        setattr(self.env, name, value)

    def env_method(self, name, *a, indices=None, **kw):
        return [getattr(self.env, name)(*a, **kw)] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        self._rng = jax.random.PRNGKey(seed or 0)
