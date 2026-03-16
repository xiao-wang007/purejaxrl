import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import numpy as np
import optax
import os
import time
from collections.abc import Mapping
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


def _find_tree_shape_mismatches(
    expected_tree: Any, restored_tree: Any, path: str = ""
) -> list[tuple[str, Any, Any]]:
    """Collect shape mismatches between two parameter pytrees."""
    mismatches: list[tuple[str, Any, Any]] = []
    if isinstance(expected_tree, Mapping):
        if not isinstance(restored_tree, Mapping):
            mismatches.append((path or "<root>", "mapping", type(restored_tree).__name__))
            return mismatches
        for key in expected_tree.keys():
            child_path = f"{path}/{key}" if path else str(key)
            if key not in restored_tree:
                mismatches.append((child_path, "missing", None))
                continue
            mismatches.extend(
                _find_tree_shape_mismatches(
                    expected_tree[key], restored_tree[key], child_path
                )
            )
        for key in restored_tree.keys():
            if key not in expected_tree:
                child_path = f"{path}/{key}" if path else str(key)
                mismatches.append((child_path, None, "unexpected"))
        return mismatches

    expected_shape = tuple(expected_tree.shape) if hasattr(expected_tree, "shape") else None
    restored_shape = tuple(restored_tree.shape) if hasattr(restored_tree, "shape") else None
    if expected_shape != restored_shape:
        mismatches.append((path or "<leaf>", expected_shape, restored_shape))
    return mismatches


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
    total_timesteps_target = int(
        config.get("TOTAL_TIMESTEPS_TARGET", config["TOTAL_TIMESTEPS"])
    )
    total_global_train_steps = jnp.array(
        max(total_timesteps_target // config["NUM_ENVS"], 1), dtype=jnp.float32
    )
    gain_schedule_split = jnp.array(
        float(config.get("GAIN_SCHEDULE_SPLIT", 0.6)), dtype=jnp.float32
    )
    gain_schedule_end = jnp.array(
        float(config.get("GAIN_SCHEDULE_END", 1.0)), dtype=jnp.float32
    )
    perf_stats = config.get("PERF_STATS")
    profile_perf = bool(config.get("PROFILE_PERF", False)) and perf_stats is not None
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
        resumed_global_train_step = jnp.array(0, dtype=jnp.int32)
        resume_path = config.get("RESUME_CHECKPOINT_PATH")
        if resume_path:
            if not os.path.exists(resume_path):
                raise FileNotFoundError(
                    f"RESUME_CHECKPOINT_PATH does not exist: {resume_path}"
                )
            with open(resume_path, "rb") as f:
                checkpoint_bytes = f.read()
            checkpoint_template = {
                "train_state": train_state,
                "global_train_step": jnp.array(0, dtype=jnp.int32),
            }
            checkpoint_data = serialization.from_bytes(
                checkpoint_template, checkpoint_bytes
            )
            mismatches = _find_tree_shape_mismatches(
                train_state.params, checkpoint_data["train_state"].params
            )
            if mismatches:
                mismatch_lines = "\n".join(
                    f"  - {param_path}: expected {expected_shape}, checkpoint {restored_shape}"
                    for param_path, expected_shape, restored_shape in mismatches[:8]
                )
                raise ValueError(
                    "Checkpoint is incompatible with current model parameter shapes.\n"
                    f"Checkpoint: {resume_path}\n"
                    "This usually means the observation/action dimensions or network architecture changed.\n"
                    f"First mismatches:\n{mismatch_lines}\n"
                    "Delete or rename the checkpoint (and meta file) to start a fresh run."
                )
            train_state = checkpoint_data["train_state"]
            resumed_global_train_step = jnp.array(
                checkpoint_data["global_train_step"], dtype=jnp.int32
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
                train_progress = jnp.clip(
                    global_train_step.astype(jnp.float32) / total_global_train_steps,
                    0.0,
                    1.0,
                )
                step_env_params = {
                    "train_step": global_train_step,
                    "train_progress": train_progress,
                    "gain_schedule_split": gain_schedule_split,
                    "gain_schedule_end": gain_schedule_end,
                }
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
                        log_ratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(log_ratio)
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
                        approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
                        clip_fraction = jnp.mean(
                            (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).astype(
                                jnp.float32
                            )
                        )

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_fraction,
                        )

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
            env_transition_step = global_train_step * jnp.array(
                config["NUM_ENVS"], dtype=jnp.int32
            )
            n_updates = global_train_step // jnp.array(
                config["NUM_STEPS"], dtype=jnp.int32
            )
            total_loss_mean = jnp.mean(loss_info[0])
            value_loss_mean = jnp.mean(loss_info[1][0])
            policy_gradient_loss_mean = jnp.mean(loss_info[1][1])
            entropy_mean = jnp.mean(loss_info[1][2])
            approx_kl_mean = jnp.mean(loss_info[1][3])
            clip_fraction_mean = jnp.mean(loss_info[1][4])
            explained_variance = 1.0 - (
                jnp.var(targets - traj_batch.value)
                / (jnp.var(targets) + 1e-8)
            )
            current_lr = (
                linear_schedule(n_updates * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
                if config["ANNEAL_LR"]
                else config["LR"]
            )

            # Reduce host callback payload to scalars computed on device.
            returned_episode = metric["returned_episode"].astype(jnp.float32)
            episodes_finished = jnp.sum(returned_episode)
            episodes_finished_int = episodes_finished.astype(jnp.int32)
            safe_den = jnp.maximum(episodes_finished, 1.0)
            episode_return_mean = (
                jnp.sum(metric["returned_episode_returns"] * returned_episode) / safe_den
            )
            episode_length_mean = (
                jnp.sum(metric["returned_episode_lengths"] * returned_episode) / safe_den
            )
            if config.get("WANDB_LOG", False):
                _wandb_interval_updates = max(
                    int(config.get("WANDB_LOG_INTERVAL_UPDATES", 1)), 1
                )
                _wandb_period = jnp.array(_wandb_interval_updates, dtype=jnp.int32)

                def wandb_callback(
                    env_step,
                    policy_step,
                    n_updates_value,
                    current_lr_value,
                    total_loss_value,
                    value_loss_value,
                    policy_gradient_loss_value,
                    entropy_value,
                    approx_kl_value,
                    clip_fraction_value,
                    explained_variance_value,
                    episodes_finished_value,
                    episode_return_mean_value,
                    episode_length_mean_value,
                ):
                    t_cb0 = time.perf_counter() if profile_perf else None
                    log_data = {
                        "train/approx_kl": float(np.asarray(approx_kl_value)),
                        "train/clip_fraction": float(np.asarray(clip_fraction_value)),
                        "train/clip_range": float(config["CLIP_EPS"]),
                        "train/entropy_loss": -float(np.asarray(entropy_value)),
                        "train/explained_variance": float(
                            np.asarray(explained_variance_value)
                        ),
                        "train/learning_rate": float(np.asarray(current_lr_value)),
                        "train/loss": float(np.asarray(total_loss_value)),
                        "train/policy_gradient_loss": float(
                            np.asarray(policy_gradient_loss_value)
                        ),
                        "train/value_loss": float(np.asarray(value_loss_value)),
                        "train/n_updates": int(np.asarray(n_updates_value)),
                        "time/iterations": int(np.asarray(n_updates_value)),
                        "time/total_timesteps": int(np.asarray(env_step)),
                        "train/policy_step": int(np.asarray(policy_step)),
                        "train/env_transition_step": int(np.asarray(env_step)),
                    }
                    episodes_finished_host = int(np.asarray(episodes_finished_value))
                    if episodes_finished_host > 0:
                        log_data["rollout/ep_rew_mean"] = float(
                            np.asarray(episode_return_mean_value)
                        )
                        log_data["rollout/ep_len_mean"] = float(
                            np.asarray(episode_length_mean_value)
                        )
                        log_data["train/episodes_finished"] = episodes_finished_host

                    wandb.log(log_data, step=int(np.asarray(env_step)))
                    if profile_perf:
                        perf_stats["wandb_calls"] = int(perf_stats.get("wandb_calls", 0)) + 1
                        perf_stats["wandb_host_sec"] = float(
                            perf_stats.get("wandb_host_sec", 0.0)
                        ) + (time.perf_counter() - t_cb0)

                def _do_wandb(_):
                    jax.debug.callback(
                        wandb_callback,
                        env_transition_step,
                        global_train_step,
                        n_updates,
                        current_lr,
                        total_loss_mean,
                        value_loss_mean,
                        policy_gradient_loss_mean,
                        entropy_mean,
                        approx_kl_mean,
                        clip_fraction_mean,
                        explained_variance,
                        episodes_finished_int,
                        episode_return_mean,
                        episode_length_mean,
                    )

                def _skip_wandb(_):
                    pass

                jax.lax.cond(
                    (n_updates > 0) & (n_updates % _wandb_period == 0),
                    _do_wandb,
                    _skip_wandb,
                    operand=None,
                )

            if config.get("DEBUG"):
                _debug_interval_updates = max(
                    int(config.get("DEBUG_PRINT_INTERVAL_UPDATES", 1)), 1
                )
                _debug_period = jnp.array(_debug_interval_updates, dtype=jnp.int32)
                total_timesteps = int(config["TOTAL_TIMESTEPS"])

                def callback(env_step, episodes_finished_value, episode_return_mean_value):
                    t_cb0 = time.perf_counter() if profile_perf else None
                    env_step_int = int(np.asarray(env_step))
                    progress = (
                        min(env_step_int / total_timesteps, 1.0)
                        if total_timesteps > 0
                        else 1.0
                    )
                    bar_width = 36
                    filled = int(bar_width * progress)
                    bar = "#" * filled + "-" * (bar_width - filled)

                    suffix = ""
                    episodes_finished_host = int(np.asarray(episodes_finished_value))
                    if episodes_finished_host > 0:
                        suffix = (
                            f" | ep_rew_mean={float(np.asarray(episode_return_mean_value)):.4f}"
                            f" | episodes={episodes_finished_host}"
                        )

                    print(
                        f"\rprogress [{bar}] {progress * 100:6.2f}% "
                        f"({env_step_int}/{total_timesteps}){suffix}",
                        end="",
                        flush=True,
                    )
                    if env_step_int >= total_timesteps:
                        print()
                    if profile_perf:
                        perf_stats["debug_calls"] = int(perf_stats.get("debug_calls", 0)) + 1
                        perf_stats["debug_host_sec"] = float(
                            perf_stats.get("debug_host_sec", 0.0)
                        ) + (time.perf_counter() - t_cb0)

                def _do_debug(_):
                    jax.debug.callback(
                        callback,
                        env_transition_step,
                        episodes_finished_int,
                        episode_return_mean,
                    )

                def _skip_debug(_):
                    pass

                jax.lax.cond(
                    (n_updates > 0) & (n_updates % _debug_period == 0),
                    _do_debug,
                    _skip_debug,
                    operand=None,
                )

            runner_state = (train_state, env_state, last_obs, rng, global_train_step)

            # Periodic checkpoint via host callback (no recompilation needed).
            # global_train_step counts env steps; each _update_step adds NUM_STEPS.
            _ckpt_interval = config.get("CHECKPOINT_INTERVAL_UPDATES", 0)
            if _ckpt_interval > 0:
                _ckpt_fn = config["CHECKPOINT_FN"]
                _ckpt_period = jnp.array(
                    _ckpt_interval * config["NUM_STEPS"], dtype=jnp.int32
                )

                def _do_ckpt(_):
                    jax.debug.callback(_ckpt_fn, train_state, global_train_step)

                def _skip_ckpt(_):
                    pass

                jax.lax.cond(
                    (global_train_step > 0)
                    & (global_train_step % _ckpt_period == 0),
                    _do_ckpt,
                    _skip_ckpt,
                    operand=None,
                )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            _rng,
            resumed_global_train_step,
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
