import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
try:
    import wandb
except ImportError:
    wandb = None
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    MJXGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    if config.get("WANDB_LOG", False) and wandb is None:
        raise ImportError(
            "WANDB_LOG=True, but wandb is not installed. Install wandb or disable WANDB_LOG."
        )

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    gae_scan_unroll = int(config.get("GAE_SCAN_UNROLL", 16))
    env_backend = config.get("ENV_BACKEND", "brax")
    if env_backend == "brax":
        env = BraxGymnaxWrapper(config["ENV_NAME"])
    elif env_backend == "mjx":
        if "MJX_ENV" not in config:
            raise ValueError(
                "When ENV_BACKEND='mjx', pass a built MJX env object in config['MJX_ENV']."
            )
        env = MJXGymnaxWrapper(
            config["MJX_ENV"],
            observation_size=config.get("OBSERVATION_SIZE"),
            action_size=config.get("ACTION_SIZE"),
            action_low=config.get("ACTION_LOW", -1.0),
            action_high=config.get("ACTION_HIGH", 1.0),
            obs_attr_names=config.get(
                "OBS_ATTR_NAMES", ("obs", "observation", "observations")
            ),
            reward_attr_names=config.get("REWARD_ATTR_NAMES", ("reward",)),
            done_attr_names=config.get(
                "DONE_ATTR_NAMES", ("done", "terminated", "is_terminal", "is_done")
            ),
        )
    else:
        raise ValueError(
            f"Unknown ENV_BACKEND='{env_backend}'. Supported values: 'brax', 'mjx'."
        )
    env_params = None

    #! this is wrapped using classes
    #! env = VecEnv(ClipAction(LogWrapper(env))), so when called by env.step()
    #! it goes from out-to-in
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        
        #! optax is the optimizer package with gradient transformation for JAX
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        
        #! TrainState is a training level dataclass
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, global_train_step = runner_state

                # SELECT ACTION
                #! returns two new independent PRNG keys because JAX RNG
                #! is a functional, but the convention here is to use _rng
                #! for the immediate next operation, and rng got carried over
                #! for future
                rng, _rng = jax.random.split(rng)
                #! does not naively deep-copy full parameter trees on every call. 
                #! Think “efficient buffer references with functional semantics,”
                #! not Python-level pointer mutation.
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                #! pi is a distribution over actions conditioned on s, therefore
                #! gives µ and σ. This defines pi(*|s), which evaluates to probability
                #! then after a sample is drawn, plug it into the policy give its 
                #! probability, log_prob gives log pi(a|s) 
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                step_env_params = {"train_step": global_train_step}
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, step_env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )

                #! runner_state is the carry for jax.lax.scan, must keep a 
                #! a fixed structure each iteration
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    rng,
                    global_train_step + jnp.array(1, dtype=jnp.int32),
                )
                return runner_state, transition

            #! scan represents loop control flow. It traces/compiles the loop
            #! body once as a reusable computation, then executes it for 
            #! NUM_STEPS iterations with carries state.
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            #! runner_state from the above scan contains the return of last
            #! iteration
            train_state, env_state, last_obs, rng, global_train_step = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                #! .scan(f, init, xs)
                #! f: the step function f(carry, x) -> (new_carry, y)
                #! init: the initial carry, which goes to the 1st arg of f, i.e. carry, whose
                #!       form must be fixed for scan to work.
                #! xs: the sequence to iterate over, each element goes to the 2nd arg of f

                #! advantages contain the full recursion, and the last carry is ignored by _
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=gae_scan_unroll,
                )
                return advantages, advantages + traj_batch.value #! Vt_target = At + V(st)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("WANDB_LOG", False):

                def wandb_callback(info, global_step):
                    returned_episode = np.asarray(info["returned_episode"]).astype(bool)
                    if not returned_episode.any():
                        return
                    returns = np.asarray(info["returned_episode_returns"])[returned_episode]
                    lengths = np.asarray(info["returned_episode_lengths"])[returned_episode]
                    wandb.log(
                        {
                            "train/episodic_return_mean": float(returns.mean()),
                            "train/episodic_return_max": float(returns.max()),
                            "train/episodic_length_mean": float(lengths.mean()),
                            "train/episodes_finished": int(returned_episode.sum()),
                        },
                        step=int(np.asarray(global_step)),
                    )

                jax.debug.callback(wandb_callback, metric, global_train_step)

            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng, global_train_step)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            _rng,
            jnp.array(0, dtype=jnp.int32),
        )
        if config.get("COLLECT_METRICS", True):
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["NUM_UPDATES"]
            )
            return {"runner_state": runner_state, "metrics": metric}

        # Avoid stacking per-update metrics to keep memory near-constant
        # as NUM_UPDATES grows.
        def _fori_update(_, carry):
            carry, _ = _update_step(carry, None)
            return carry

        runner_state = jax.lax.fori_loop(
            0, config["NUM_UPDATES"], _fori_update, runner_state
        )
        return {"runner_state": runner_state}

    return train


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 512,
        "NUM_STEPS": 200,  # my dt policy = 0.02, this is 4s rollout 
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_BACKEND": "mjx",
        "ENV_NAME": "franka",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
    }
    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
