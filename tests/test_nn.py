"""Check derivatives, optimizer updates, and small network training."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, softmax

from tsdst.nn import activations as a, initializers as i, optimizers as o
from tsdst.nn.model import NeuralNetwork


@pytest.mark.parametrize("function,derivative", [
    ("relu", "relu_der"), ("elu", "elu_der"), ("sigmoid", "sigmoid_der"),
    ("sigmoid_scale", "sigmoid_scale_der"), ("selu", "selu_der"),
    ("gelu", "gelu_der"), ("gelu_speedy", "gelu_speedy_der"),
    ("linear", "linear_der")])
def test_elementwise_derivatives(function, derivative):
    x = np.array([-.8, .3, 1.2])
    fn, der = getattr(a, function), getattr(a, derivative)
    numerical = (fn(x + 1e-5) - fn(x - 1e-5)) / 2e-5
    assert_allclose(der(x), numerical, rtol=1e-5, atol=1e-6)


def test_softmax_and_gelu():
    z = np.array([[1., 2., 3.], [-1., 0., 2.]])
    assert_allclose(a.softmax(z), softmax(z, axis=1))
    expected = z.copy()
    expected[:, -1] = 0
    assert_allclose(a.softmax_oneout(z), softmax(expected, axis=1))
    assert_allclose(a.gelu_approx(z), a.gelu(z), atol=1e-6)
    jacobian = a.softmax_der(z)
    assert jacobian.shape == (6, 6)
    assert_allclose(jacobian.sum(axis=1), 0, atol=1e-14)
    assert_allclose(a.create_jac_mask((6, 6), 3)[:3, 3:], 0)


def test_losses_and_combined_derivatives():
    target = np.array([[1., 0.], [0., 1.]])
    z = np.array([[.5, -.5], [-.5, .5]])
    prediction = a.softmax(z)
    assert a.cross_entropy(target, prediction) == pytest.approx(-np.log(prediction.diagonal()).sum())
    assert_allclose(a.cross_entropy_der(target, prediction), -target / prediction)
    assert_allclose(a.softmax_cross_entropy_der(target, prediction), prediction - target)
    assert_allclose(a.softmax_cross_entropy_der_fullmath(target, prediction, z), prediction - target)
    binary = np.array([[0.], [1.]])
    binary_z = np.array([[-1.], [1.]])
    probabilities = expit(binary_z)
    assert np.isfinite(a.cross_entropy_binary(binary, probabilities))
    assert_allclose(a.cross_entropy_binary_der(binary, probabilities),
                    -(binary / probabilities - (1 - binary) / (1 - probabilities)))
    assert_allclose(a.sigmoid_cross_entropy_binary_der(binary, probabilities, binary_z), probabilities - binary)
    assert a.mse(binary, probabilities) == pytest.approx(np.sum((binary - probabilities) ** 2))
    assert_allclose(a.mse_der(binary, probabilities), 2 * (probabilities - binary))
    assert_allclose(a.mse_linear_der(binary, probabilities, binary_z), a.mse_der(binary, probabilities))


@pytest.mark.parametrize("name", ["he_uniform", "he_normal", "xavier_uniform", "xavier_normal",
                                    "lecun_uniform", "lecun_normal", "random_normal"])
def test_initializers(name):
    values = getattr(i, name)(8, 5)
    assert values.shape == (8, 5)
    assert np.isfinite(values).all() and np.std(values) > 0


def test_optimizer_updates():
    params = {"w": np.array([1., 2.])}
    gradient = {"w": np.array([.2, -.3])}
    assert_allclose(o.gradient_descent(deepcopy(params), gradient, {"learning_rate": .1})["w"], [.98, 2.03])
    options = dict(learning_rate=.1, beta1=.9, beta2=.999, eps=1e-8, i=0,
                   mt={"w": np.zeros(2)}, vt={"w": np.zeros(2)})
    assert_allclose(o.adam(deepcopy(params), gradient, options)["w"], [.9, 2.1], atol=1e-7)


def network_spec():
    return {"output": dict(depth=1, initializer="xavier_uniform", activation="sigmoid",
                            activation_args={}, derivative="sigmoid_cross_entropy_binary_der",
                            derivative_args={}, use_batch_norm=False,
                            **{"lambda": {"Weight": 0, "activity": 0, "bias": 0},
                               "lp_norm": {"Weight": 2, "activity": 2, "bias": 2}},
                            cost="cross_entropy_binary", cost_args={})}


def test_network_training_and_predictions():
    x = np.linspace(-2, 2, 40)[:, None]
    y = (x > 0).astype(float)
    model = NeuralNetwork(network_spec(), batch_size=40, num_iterations=40,
                          optimizer_args=dict(learning_rate=.05, beta1=.9, beta2=.999, eps=1e-8),
                          print_cost=False, scorer="accuracy")
    assert model.fit(x, y) is model
    prediction = model.predict(x)
    assert prediction.shape == y.shape
    assert model.costs[-1] < model.costs[0]
    assert model.score(x, y) >= .9
    assert model.draw_predictive_samples(x, n_samples=3).shape == (40, 3)


def test_network_gradients_match_finite_differences():
    x = np.array([[-1.], [.2], [1.]])
    y = np.array([[0.], [1.], [1.]])
    model = NeuralNetwork(network_spec(), print_cost=False)
    params = model.initialize_wb(x)
    output, zs, bn, hzs, dropout, _ = model.forward_prop(x, params, {})
    gradients = model.back_prop(x, y, params, zs, bn, hzs, dropout)
    for key in params:
        plus, minus = deepcopy(params), deepcopy(params)
        plus[key][0, 0] += 1e-5
        minus[key][0, 0] -= 1e-5
        loss_plus = a.cross_entropy_binary(y, model.forward_prop(x, plus, {})[0]) / len(x)
        loss_minus = a.cross_entropy_binary(y, model.forward_prop(x, minus, {})[0]) / len(x)
        assert gradients[key][0, 0] == pytest.approx((loss_plus - loss_minus) / 2e-5, rel=1e-4)


def test_network_oversized_batch_and_training_metric():
    x = np.linspace(-2, 2, 20)[:, None]
    y = (x > 0).astype(float)
    spec = network_spec()
    spec["output"]["evaluation_metric"] = "accuracy"
    model = NeuralNetwork(spec, batch_size=100, num_iterations=3, print_cost=False).fit(x, y)
    assert model.costs.shape == (3,)
    assert np.isfinite(model.metric).all()
    assert "depth" in spec["output"] and model.model is not spec
