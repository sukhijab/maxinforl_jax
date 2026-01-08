import functools

import chex
import jax.random
from jaxrl.networks.common import MLP
import flax.linen as nn
import flax
from typing import Dict, Callable
import jax.numpy as jnp
import optax
import os
from jaxtyping import PyTree
from jax.scipy.stats import norm


@chex.dataclass
class NormalizerState:
    mean: chex.Array
    std: chex.Array
    num_points: int


@chex.dataclass
class EnsembleNormalizerState:
    input_normalizer_state: NormalizerState
    output_normalizer_state: NormalizerState
    info_gain_normalizer_state: NormalizerState


@chex.dataclass
class EnsembleState:
    vmapped_params: PyTree
    opt_state: PyTree
    step: int
    ensemble_normalizer_state: EnsembleNormalizerState


class Normalizer:
    max_points: jnp.array = jnp.array(1e6, dtype=jnp.int32)

    @staticmethod
    def reset(normalizer_state: NormalizerState) -> NormalizerState:
        return NormalizerState(
            mean=jnp.zeros_like(normalizer_state.mean),
            std=jnp.ones(normalizer_state.std),
            num_points=0,
        )

    def update_stats(self, x: chex.Array, normalizer_state: NormalizerState) -> NormalizerState:
        assert len(x.shape) == 2 and x.shape[-1] == normalizer_state.mean.shape[-1]
        num_points = x.shape[0]
        total_points = num_points + normalizer_state.num_points
        mean = (normalizer_state.mean * normalizer_state.num_points
                + jnp.sum(x, axis=0)) / total_points
        new_s_n = jnp.square(normalizer_state.std) * normalizer_state.num_points \
                  + jnp.sum(jnp.square(x - mean), axis=0) + \
                  normalizer_state.num_points * jnp.square(normalizer_state.mean - mean)

        new_var = new_s_n / total_points
        std = jnp.clip(jnp.sqrt(new_var), min=1e-3)
        new_normalizer_state = NormalizerState(
            mean=mean,
            std=std,
            num_points=jnp.minimum(total_points, self.max_points)  # keep at most max number of points to avoid overflow
        )
        return new_normalizer_state

    @staticmethod
    def normalize(x: chex.Array, normalizer_state: NormalizerState):
        return (x - normalizer_state.mean) / normalizer_state.std

    @staticmethod
    def denormalize(norm_x: chex.Array, normalizer_state: NormalizerState):
        return norm_x * normalizer_state.std + normalizer_state.mean

    @staticmethod
    def scale(unscaled_x: chex.Array, normalizer_state: NormalizerState):
        return unscaled_x * normalizer_state.std


class DeterministicEnsemble(object):

    def __init__(self,
                 model_kwargs: Dict,
                 optimizer: optax.GradientTransformation,
                 num_heads: int = 5,
                 agg_info_gain: str = 'mean',
                 normalize_data: bool = True,
                 normalize_info_gain: bool = True,
                 use_entropy_for_int_rew: bool = True,
                 activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu
                 ):
        self.num_heads = num_heads
        self.tx = optimizer
        self.agg_info_gain = agg_info_gain
        self.model = MLP(**model_kwargs, activations=activations)
        self.learn_std = False
        self.normalize_data = normalize_data
        self.normalize_info_gain = normalize_info_gain
        self.use_entropy_for_int_rew = use_entropy_for_int_rew
        self.normalizer = Normalizer()
        out_size = model_kwargs['hidden_dims'][-1]
        self._disg_weights = jnp.ones(out_size)

    def init(self, key: jax.random.PRNGKey, input: jnp.ndarray):
        key = jax.random.split(key, self.num_heads)
        vmapped_params = jax.vmap(self.model.init, in_axes=(0, None))(key, input)
        out, _ = self.apply(input, params=vmapped_params)
        out_size = out.shape[-1]

        input_normalizer_state = NormalizerState(
            mean=jnp.zeros(input.shape[-1]),
            std=jnp.ones_like(input.shape[-1]),
            num_points=0,
        )
        output_normalizer_state = NormalizerState(
            mean=jnp.zeros(out_size),
            std=jnp.ones(out_size),
            num_points=0,
        )

        info_gain_normalizer_state = NormalizerState(
            mean=jnp.zeros(1),
            std=jnp.ones(1),
            num_points=0,
        )

        ensemble_normalizer_state = EnsembleNormalizerState(
            input_normalizer_state=input_normalizer_state,
            output_normalizer_state=output_normalizer_state,
            info_gain_normalizer_state=info_gain_normalizer_state,
        )
        self._disg_weights = jnp.ones(out_size)

        opt_state = self.tx.init(vmapped_params)
        return EnsembleState(
            vmapped_params=vmapped_params,
            opt_state=opt_state,
            step=0,
            ensemble_normalizer_state=ensemble_normalizer_state,
        )

    def set_disg_weights(self, disg_weights: jnp.ndarray):
        chex.assert_shape(disg_weights, self._disg_weights.shape)
        self._disg_weights = disg_weights

    def update_normalization_stats(self, input, output, state: EnsembleState):
        if self.normalize_data:
            new_input_normalizer_state = self.normalizer.update_stats(
                input,
                normalizer_state=state.ensemble_normalizer_state.input_normalizer_state)
            new_output_normalizer_state = self.normalizer.update_stats(
                output,
                normalizer_state=state.ensemble_normalizer_state.output_normalizer_state)
            new_ens_normalizer_state = state.ensemble_normalizer_state.replace(
                input_normalizer_state=new_input_normalizer_state,
                output_normalizer_state=new_output_normalizer_state,
            )
            new_state = state.replace(ensemble_normalizer_state=new_ens_normalizer_state)
            return new_state
        else:
            return state

    def __call__(self, input, state: EnsembleState, denormalize_output: bool = True):
        input = self.normalizer.normalize(x=input,
                                          normalizer_state=state.ensemble_normalizer_state.input_normalizer_state)
        normalizer_out_mean, normalizer_out_std = self.apply(input, params=state.vmapped_params)
        if denormalize_output:
            output_mean = jax.vmap(lambda x: self.normalizer.denormalize(
                norm_x=x,
                normalizer_state=state.ensemble_normalizer_state.output_normalizer_state))(normalizer_out_mean)
            output_std = jax.vmap(lambda x: self.normalizer.scale(
                unscaled_x=x,
                normalizer_state=state.ensemble_normalizer_state.output_normalizer_state))(normalizer_out_std)
            return output_mean, output_std
        else:
            return normalizer_out_mean, normalizer_out_std

    @functools.partial(jax.jit, static_argnums=(0,))
    def apply(self, input, params):
        out = jax.vmap(self.model.apply, in_axes=(0, None))(params, input)
        return out, jnp.ones_like(out) * 1e-3

    def _neg_log_posterior(self,
                           predicted_outputs: chex.Array,
                           predicted_stds: chex.Array,
                           target_outputs: chex.Array) -> chex.Array:
        nll = jax.vmap(jax.vmap(self._nll), in_axes=(0, 0, None))(predicted_outputs, predicted_stds, target_outputs)
        neg_log_post = nll.mean()
        return neg_log_post

    def _nll(self,
             predicted_outputs: chex.Array,
             predicted_stds: chex.Array,
             target_outputs: chex.Array) -> chex.Array:
        # chex.assert_equal_shape([target_outputs, predicted_stds[0, ...], predicted_outputs[0, ...]])
        if self.learn_std:
            log_prob = norm.logpdf(target_outputs, loc=predicted_outputs, scale=predicted_stds)
            return -jnp.mean(log_prob)
        else:
            # replace predicted stds with ones for stable learning
            loss = jnp.square(target_outputs - predicted_outputs).mean()
            return loss
            # log_prob =
            # log_prob = norm.logpdf(target_outputs, loc=predicted_outputs, scale=jnp.ones_like(predicted_stds))

    @functools.partial(jax.jit, static_argnums=(0,))
    def loss(self, params, input, y):
        out, std = self.apply(input, params=params)
        neg_log_prob = self._neg_log_posterior(predicted_outputs=out, predicted_stds=std, target_outputs=y)
        mse = jnp.square(out - y[jnp.newaxis]).mean()
        return neg_log_prob, mse

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(self, input, output, state: EnsembleState):
        state = self.update_normalization_stats(input, output, state)
        input = self.normalizer.normalize(input, state.ensemble_normalizer_state.input_normalizer_state)
        output = self.normalizer.normalize(output, state.ensemble_normalizer_state.output_normalizer_state)
        params, opt_state, step = state.vmapped_params, state.opt_state, state.step
        (loss, mse), grads = jax.value_and_grad(self.loss, has_aux=True)(params, input, output)
        updates, new_opt_state = self.tx.update(grads, opt_state,
                                                params)
        new_params = optax.apply_updates(params, updates)

        new_state = state.replace(
            vmapped_params=new_params,
            opt_state=new_opt_state,
            step=step + 1,
        )

        return new_state, (loss, mse)

    def save(self, save_path: str, state: EnsembleState):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(flax.serialization.to_bytes(state.vmapped_params))

    def load(self, load_path: str, state: EnsembleState) -> EnsembleState:
        with open(load_path, 'rb') as f:
            params = flax.serialization.from_bytes(state.params, f.read())
        return state.replace(params=params)

    @functools.partial(jax.jit, static_argnums=(0, 3))
    def get_info_gain(self, input, state: EnsembleState, update_normalizer: bool = False):
        # we look at the normalized disagreement and std
        mean, std = self(input=input, state=state, denormalize_output=False)
        al_std = jnp.clip(jnp.sqrt(jnp.square(std).mean(0)), min=1e-3)
        ep_std = mean.std(axis=0)
        if self.use_entropy_for_int_rew:
            ratio = jnp.square(ep_std / al_std) * self._disg_weights
            if self.agg_info_gain == 'sum':
                info_gain = jnp.log(1 + ratio).sum(axis=-1).reshape(-1, 1)
            elif self.agg_info_gain == 'mean':
                info_gain = jnp.log(1 + ratio).mean(axis=-1).reshape(-1, 1)
            elif self.agg_info_gain == 'max':
                info_gain = jnp.log(1 + ratio).max(axis=-1).reshape(-1)
            else:
                raise NotImplementedError
        else:
            ep_std = ep_std * self._disg_weights
            if self.agg_info_gain == 'sum':
                info_gain = ep_std.sum(axis=-1).reshape(-1, 1)
            elif self.agg_info_gain == 'mean':
                info_gain = ep_std.mean(axis=-1).reshape(-1, 1)
            elif self.agg_info_gain == 'max':
                info_gain = ep_std.max(axis=-1).reshape(-1)
            else:
                raise NotImplementedError
        if self.normalize_info_gain and update_normalizer:
            # stop gradients wrt the info gain for normalization
            new_info_gain_normalizer_state = \
                self.normalizer.update_stats(x=jax.lax.stop_gradient(info_gain),
                                             normalizer_state=state.ensemble_normalizer_state.info_gain_normalizer_state)
        else:
            new_info_gain_normalizer_state = state.ensemble_normalizer_state.info_gain_normalizer_state

        info_gain = self.normalizer.normalize(x=info_gain, normalizer_state=new_info_gain_normalizer_state)
        new_norm_state = state.ensemble_normalizer_state.replace(
            info_gain_normalizer_state=new_info_gain_normalizer_state)
        new_state = state.replace(ensemble_normalizer_state=new_norm_state)
        return info_gain.reshape(-1), new_state


class ProbabilisticEnsemble(DeterministicEnsemble):
    def __init__(self, model_kwargs: Dict, sig_min: float = 1e-3, sig_max: float = 1e2, *args, **kwargs):
        hidden_dims = list(model_kwargs['hidden_dims'])
        hidden_dims[-1] = 2 * hidden_dims[-1]
        hidden_dims = tuple(hidden_dims)
        model_kwargs['hidden_dims'] = hidden_dims
        self.sig_min = sig_min
        self.sig_max = sig_max
        super().__init__(model_kwargs=model_kwargs, *args, **kwargs)
        self.learn_std = True

    def apply_single(self, input, params):
        out = self.model.apply(params, input)
        mu, sig = jnp.split(out, 2, axis=-1)
        sig = nn.softplus(sig)
        sig = jnp.clip(sig, self.sig_min, self.sig_max)
        return mu, sig

    def apply(self, input, params):
        return jax.vmap(self.apply_single, in_axes=(None, 0))(input, params)


def main():
    key = jax.random.PRNGKey(0)
    output_dim = 2
    train_data_size = 256
    num_heads = 5

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, train_data_size).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys * (1 + noise_level * jax.random.normal(key=jax.random.PRNGKey(0), shape=ys.shape))

    tx = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    model_key, key = jax.random.split(key, 2)

    ensemble = DeterministicEnsemble(
        model_kwargs={'hidden_dims': [64, 64, output_dim]},
        optimizer=tx,
        num_heads=num_heads)
    ensemble_state = ensemble.init(key=key, input=xs)
    num_steps = 20_000

    for i in range(num_steps):
        sample_key, key = jax.random.split(key, 2)
        inds = jax.random.randint(key=sample_key, shape=(32,), minval=0, maxval=train_data_size)
        x_train, y_train = xs[inds], ys[inds]
        ensemble_state, aux = ensemble.update(input=x_train, output=y_train, state=ensemble_state)
        info_gain, ensemble_state = ensemble.get_info_gain(input=x_train, state=ensemble_state, update_normalizer=False)
        print(f'step: {i}, loss: {aux[0]}, mse: {aux[1]}, info gain: {info_gain.mean()}')

    import matplotlib.pyplot as plt

    test_xs = jnp.linspace(-5, 15, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    preds, std = ensemble(test_xs, state=ensemble_state)
    preds_mean = preds.mean(axis=0)
    ep_std = preds.std(axis=0)
    total_std = jnp.sqrt(jnp.square(ep_std) + jnp.square(std).mean(axis=0))

    for j in range(output_dim):
        plt.scatter(xs.reshape(-1), ys[:, j], label='Data', color='red')
        plt.plot(test_xs, preds_mean[:, j], label='Mean', color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (preds_mean[:, j] - 2 * ep_std[:, j]).reshape(-1),
                         (preds_mean[:, j] + 2 * ep_std[:, j]).reshape(
                             -1),
                         label=r'$2\sigma_{eps}$', alpha=0.3, color='blue')
        plt.fill_between(test_xs.reshape(-1),
                         (preds_mean[:, j] - 2 * total_std[:, j]).reshape(-1),
                         (preds_mean[:, j] + 2 * total_std[:, j]).reshape(
                             -1),
                         label=r'$2\sigma_{tot}$', alpha=0.3, color='yellow')
        for i in range(num_heads):
            plt.plot(test_xs, preds[i, :, j], label='Mean', color='black', alpha=0.5)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label='True', color='green')
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        # plt.savefig(f'bnn_{j}.pdf')
        plt.show()


if __name__ == '__main__':
    main()
