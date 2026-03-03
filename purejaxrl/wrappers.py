import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
import navix as nx


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


class MJXGymnaxWrapper:
    """Adapts a raw MJX env object to the Gymnax-style API used by PureJaxRL."""

    def __init__(
        self,
        env,
        observation_size=None,
        action_size=None,
        action_low=-1.0,
        action_high=1.0,
        obs_attr_names=("obs", "observation", "observations"),
        reward_attr_names=("reward",),
        done_attr_names=("done", "terminated", "is_terminal", "is_done"),
    ):
        self._env = env
        self._observation_size = observation_size
        self._action_size = action_size
        self._action_low = action_low
        self._action_high = action_high
        self._obs_attr_names = obs_attr_names
        self._reward_attr_names = reward_attr_names
        self._done_attr_names = done_attr_names

    def _first_attr(self, state, names):
        for name in names:
            if hasattr(state, name):
                value = getattr(state, name)
                return value() if callable(value) else value
        return None

    def _extract_obs(self, state):
        obs = self._first_attr(state, self._obs_attr_names)
        if obs is None:
            raise ValueError(
                "Could not extract observation from MJX state. "
                "Provide one of obs_attr_names in MJXGymnaxWrapper."
            )
        return obs

    def _extract_reward(self, state):
        reward = self._first_attr(state, self._reward_attr_names)
        if reward is None:
            raise ValueError(
                "Could not extract reward from MJX state. "
                "Provide one of reward_attr_names in MJXGymnaxWrapper."
            )
        return reward

    def _extract_done(self, state):
        done = self._first_attr(state, self._done_attr_names)
        if done is None:
            raise ValueError(
                "Could not extract done flag from MJX state. "
                "Provide one of done_attr_names in MJXGymnaxWrapper."
            )
        return done > 0.5

    def _call_with_optional_params(self, fn, *args, params=None):
        if params is None:
            return fn(*args)
        try:
            return fn(*args, params)
        except TypeError:
            return fn(*args)

    def reset(self, key, params=None):
        out = self._call_with_optional_params(self._env.reset, key, params=params)
        if isinstance(out, tuple) and len(out) == 2:
            obs, state = out
            return obs, state
        state = out
        return self._extract_obs(state), state

    def step(self, key, state, action, params=None):
        del key  # Most MJX envs are deterministic given state/action.
        out = self._call_with_optional_params(
            self._env.step, state, action, params=params
        )
        if isinstance(out, tuple) and len(out) == 5:
            obs, next_state, reward, done, info = out
            return obs, next_state, reward, done, info
        next_state = out
        obs = self._extract_obs(next_state)
        reward = self._extract_reward(next_state)
        done = self._extract_done(next_state)
        return obs, next_state, reward, done, {}

    def observation_space(self, params):
        del params
        observation_size = self._observation_size
        if observation_size is None:
            observation_size = getattr(self._env, "observation_size", None)
        if observation_size is None:
            raise ValueError(
                "MJX observation size is unknown. Pass observation_size to "
                "MJXGymnaxWrapper."
            )
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(observation_size,))

    def action_space(self, params):
        del params
        action_size = self._action_size
        if action_size is None:
            action_size = getattr(self._env, "action_size", None)
        if action_size is None:
            raise ValueError(
                "MJX action size is unknown. Pass action_size to MJXGymnaxWrapper."
            )
        return spaces.Box(
            low=self._action_low,
            high=self._action_high,
            shape=(action_size,),
        )

class NavixGymnaxWrapper:
    def __init__(self, env_name):
        self._env = nx.make(env_name)

    def reset(self, key, params=None):
        timestep = self._env.reset(key)
        return timestep.observation, timestep

    def step(self, key, state, action, params=None):
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

    def observation_space(self, params):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=(np.prod(self._env.observation_space.shape),),
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        return spaces.Discrete(
            num_categories=self._env.action_space.maximum.item() + 1,
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count #! total is increasing until reset()

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info
