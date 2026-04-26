import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal


class Normal(nn.Module):
    log_std: float = -0.5

    @nn.compact
    def __call__(self, x):
        log_std = self.param("log_std", lambda _: jnp.full((x.shape[-1],), self.log_std))
        return x, log_std


class VectorizedGaussianActor(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, x):
        for size in self.hidden_dims:
            x = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std


class VectorizedQNetwork(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        for size in self.hidden_dims:
            x = nn.Dense(size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
            x = nn.relu(x)
        q = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(q, axis=-1)


class SACState(NamedTuple):
    actor_params: Any
    critic1_params: Any
    critic2_params: Any
    actor_opt_state: Any
    critic1_opt_state: Any
    critic2_opt_state: Any
    target_critic1_params: Any
    target_critic2_params: Any
    step: int


def make_sac(config):
    obs_dim = config["OBS_DIM"]
    action_dim = config["ACTION_DIM"]
    num_envs = config.get("NUM_ENVS", 1)

    actor = VectorizedGaussianActor(action_dim)
    critic1 = VectorizedQNetwork(action_dim)
    critic2 = VectorizedQNetwork(action_dim)

    def init(rng):
        rng_actor, rng_c1, rng_c2 = jax.random.split(rng, 3)
        obs = jnp.zeros((num_envs, obs_dim))
        action = jnp.zeros((num_envs, action_dim))

        actor_params = actor.init(rng_actor, obs)
        critic1_params = critic1.init(rng_c1, obs, action)
        critic2_params = critic2.init(rng_c2, obs, action)

        actor_tx = optax.adam(config["ACTOR_LR"])
        critic_tx = optax.adam(config["CRITIC_LR"])

        actor_opt_state = actor_tx.init(actor_params)
        critic1_opt_state = critic_tx.init(critic1_params)
        critic2_opt_state = critic_tx.init(critic2_params)

        return SACState(
            actor_params=actor_params,
            critic1_params=critic1_params,
            critic2_params=critic2_params,
            actor_opt_state=actor_opt_state,
            critic1_opt_state=critic1_opt_state,
            critic2_opt_state=critic2_opt_state,
            target_critic1_params=critic1_params,
            target_critic2_params=critic2_params,
            step=0,
        )

    @jax.jit
    def select_action(params, obs, rng):
        mean, log_std = actor.apply(params, obs)
        key, rng = jax.random.split(rng)
        noise = jax.random.normal(key, mean.shape)
        action = mean + jnp.exp(log_std) * noise
        action = jnp.tanh(action)
        return action, rng

    @jax.jit
    def log_prob(action, mean, log_std):
        action = jnp.arctanh(jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6))
        log_prob = -0.5 * ((action - mean) ** 2) / (jnp.exp(log_std) ** 2) - jnp.log(jnp.exp(log_std)) - 0.5 * jnp.log(2 * jnp.pi)
        log_prob = log_prob.sum(axis=-1)
        log_prob -= jnp.log(1 - jnp.tanh(action) ** 2 + 1e-6).sum(axis=-1)
        return log_prob

    def update(sac_state, batch, rng):
        obs, actions, rewards, next_obs, dones = batch

        rng, rng_actor = jax.random.split(rng)
        mean, log_std = actor.apply(sac_state.actor_params, next_obs)
        next_actions, rng_actor = select_action(sac_state.actor_params, next_obs, rng_actor)
        next_log_probs = log_prob(next_actions, mean, log_std)

        target_q1 = critic1.apply(sac_state.target_critic1_params, next_obs, next_actions)
        target_q2 = critic2.apply(sac_state.target_critic2_params, next_obs, next_actions)
        target_q = jnp.minimum(target_q1, target_q2)
        target = rewards + config["GAMMA"] * (1 - dones) * (target_q - config["ALPHA"] * next_log_probs)

        def critic_loss_fn(params, obs, actions, target):
            q = critic1.apply(params, obs, actions)
            return jnp.mean((q - target) ** 2)

        grad_fn = jax.grad(critic_loss_fn)
        critic1_grads = grad_fn(sac_state.critic1_params, obs, actions, target)
        critic1_updates, critic1_opt_state = optax.adam(
            config["CRITIC_LR"]
        ).update(critic1_grads, sac_state.critic1_opt_state)
        critic1_params = optax.apply_updates(sac_state.critic1_params, critic1_updates)

        grad_fn2 = jax.grad(critic_loss_fn)
        critic2_grads = grad_fn2(sac_state.critic2_params, obs, actions, target)
        critic2_updates, critic2_opt_state = optax.adam(
            config["CRITIC_LR"]
        ).update(critic2_grads, sac_state.critic2_opt_state)
        critic2_params = optax.apply_updates(sac_state.critic2_params, critic2_updates)

        rng, rng_actor = jax.random.split(rng)
        mean, log_std = actor.apply(sac_state.actor_params, obs)
        new_actions, rng_actor = select_action(sac_state.actor_params, obs, rng_actor)
        log_probs = log_prob(new_actions, mean, log_std)

        q1 = critic1.apply(sac_state.critic1_params, obs, new_actions)
        q2 = critic2.apply(sac_state.critic2_params, obs, new_actions)
        q = jnp.minimum(q1, q2)
        actor_loss = -jnp.mean(q - config["ALPHA"] * log_probs)

        actor_grads = jax.grad(lambda p: actor_loss)(sac_state.actor_params)
        actor_updates, actor_opt_state = optax.adam(
            config["ACTOR_LR"]
        ).update(actor_grads, sac_state.actor_opt_state)
        actor_params = optax.apply_updates(sac_state.actor_params, actor_updates)

        target_critic1_params = jax.tree_util.tree_map(
            lambda a, b: a * (1 - config["TAU"]) + b * config["TAU"],
            sac_state.target_critic1_params,
            critic1_params,
        )
        target_critic2_params = jax.tree_util.tree_map(
            lambda a, b: a * (1 - config["TAU"]) + b * config["TAU"],
            sac_state.target_critic2_params,
            critic2_params,
        )

        return sac_state._replace(
            actor_params=actor_params,
            critic1_params=critic1_params,
            critic2_params=critic2_params,
            actor_opt_state=actor_opt_state,
            critic1_opt_state=critic1_opt_state,
            critic2_opt_state=critic2_opt_state,
            target_critic1_params=target_critic1_params,
            target_critic2_params=target_critic2_params,
            step=sac_state.step + 1,
        ), actor_loss

    return {
        "init": init,
        "select_action": select_action,
        "update": update,
    }


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.done[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
            self.done[idx],
        )


def train_sac(env_fn, config):
    sac_fn = make_sac(config)
    buffer = ReplayBuffer(config["BUFFER_SIZE"], config["OBS_DIM"], config["ACTION_DIM"])

    rng = jax.random.PRNGKey(config["SEED"])
    sac_state = sac_fn["init"](rng)

    returns = []
    total_steps = 0

    for episode in range(config["NUM_EPISODES"]):
        obs, _ = env_fn().reset()
        episode_return = 0.0
        done = False

        while not done:
            if total_steps < config["START_STEPS"]:
                action = env_fn().action_space.sample()
            else:
                rng, _rng = jax.random.split(rng)
                action, rng = sac_fn["select_action"](sac_state.actor_params, obs[None, :], _rng)
                action = action[0]

            next_obs, reward, done, _, _ = env_fn().step(action)
            buffer.add(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_return += reward
            total_steps += 1

            if total_steps >= config["START_STEPS"]:
                for _ in range(config["UPDATES_PER_STEP"]):
                    batch = buffer.sample(config["BATCH_SIZE"])
                    batch_jax = tuple(jnp.array(x) for x in batch)
                    sac_state, loss = sac_fn["update"](sac_state, batch_jax, rng)

            if done:
                returns.append(episode_return)
                mean_return = np.mean(returns[-100:]) if len(returns) >= 1 else 0.0
                print(f"Ep {episode+1}: steps={total_steps}, ret={episode_return:.1f}, mean={mean_return:.1f}")
                break

    return {"sac_state": sac_state, "returns": returns}


if __name__ == "__main__":
    import gymnasium as gym

    config = {
        "OBS_DIM": 11,
        "ACTION_DIM": 2,
        "SEED": 0,
        "ACTOR_LR": 3e-4,
        "CRITIC_LR": 3e-4,
        "GAMMA": 0.99,
        "ALPHA": 0.2,
        "TAU": 0.005,
        "BUFFER_SIZE": 100000,
        "BATCH_SIZE": 256,
        "START_STEPS": 1000,
        "UPDATES_PER_STEP": 1,
        "NUM_EPISODES": 1000,
    }

    def env_fn():
        return gym.make("HalfCheetah-v4")

    result = train_sac(env_fn, config)
    print("Training complete")