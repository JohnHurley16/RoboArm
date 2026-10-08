"""Compile and step every task for a few batched steps; report obs/reward shapes and timing."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax, jax.numpy as jnp
from envs import make_env, TASKS

names = sys.argv[1:] or list(TASKS)
for name in names:
    env = make_env(name)
    B = 8
    reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
    t = time.time(); s = reset(jax.random.split(jax.random.PRNGKey(0), B)); s.obs.block_until_ready()
    t_reset = time.time() - t
    a = jnp.zeros((B, env.action_size))
    t = time.time(); s = step(s, a); s.obs.block_until_ready(); t_jit = time.time() - t
    t = time.time()
    for i in range(20):
        a = jax.random.uniform(jax.random.PRNGKey(i), (B, env.action_size), minval=-0.3, maxval=0.3)
        s = step(s, a)
    s.obs.block_until_ready(); t_run = (time.time() - t) / 20
    print(f"{name:6s} obs {s.obs.shape} (declared {env.observation_size})  reward {jnp.round(s.reward[:4], 3)}  "
          f"nan {bool(jnp.isnan(s.obs).any())}  done {s.done.sum():.0f}/{B}  "
          f"compile reset {t_reset:.0f}s step {t_jit:.0f}s, step {t_run * 1000:.0f} ms/batch of {B} (CPU)")
    print("   metrics", {k: float(v.mean()) for k, v in s.metrics.items()})
