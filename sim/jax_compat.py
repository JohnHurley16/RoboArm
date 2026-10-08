"""Small shims so Brax 0.14 runs on JAX >= 0.11 (which removed a few pmap-era helpers).
Imported by train.py and play.py before Brax.  Safe to delete once Brax catches up."""
import jax
import jax.numpy as jnp


def _put_replicated(x, devices):
    devices = list(devices)

    def rep(a):
        a = jnp.asarray(a)
        if len(devices) == 1:
            return jax.device_put(a[None], devices[0])
        sharding = jax.sharding.PositionalSharding(devices).reshape((len(devices),) + (1,) * a.ndim) \
            if hasattr(jax.sharding, "PositionalSharding") else None
        stacked = jnp.stack([a] * len(devices))
        return jax.device_put(stacked, sharding) if sharding is not None else stacked
    return jax.tree.map(rep, x)


def _put_sharded(shards, devices):
    devices = list(devices)
    if len(devices) == 1:
        return jax.tree.map(lambda *a: jax.device_put(jnp.stack(a), devices[0]), *shards)
    return jax.tree.map(lambda *a: jnp.stack(a), *shards)


for name, fn in (("device_put_replicated", _put_replicated), ("device_put_sharded", _put_sharded)):
    try:
        getattr(jax, name)
    except AttributeError:
        setattr(jax, name, fn)
