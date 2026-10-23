"""Check complete network gradients and prediction state against fixed losses."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tsdst.nn import activations, initializers
from tsdst.nn.model import NeuralNetwork


def architecture(batch_norm=False, dropout=1.0, penalty=0.0, outputs=1):
    def layer(depth, activation, derivative):
        return dict(
            depth=depth,
            initializer="xavier_uniform",
            activation=activation,
            derivative=derivative,
            activation_args={},
            derivative_args={},
            dropout_keep_prob=dropout,
            use_batch_norm=batch_norm,
            **{
                "lambda": {"Weight": penalty, "activity": penalty, "bias": penalty},
                "lp_norm": {"Weight": 2, "activity": 2, "bias": 2},
            },
        )

    hidden = layer(3, "sigmoid", "sigmoid_der")
    output = layer(
        outputs,
        "sigmoid" if outputs == 1 else "softmax",
        "sigmoid_cross_entropy_binary_der"
        if outputs == 1
        else "softmax_cross_entropy_der",
    )
    output.update(
        cost="cross_entropy_binary" if outputs == 1 else "cross_entropy", cost_args={}
    )
    return {"hidden": hidden, "output": output}


@pytest.mark.parametrize("batch_norm", [False, True])
@pytest.mark.parametrize("dropout", [1.0, 0.75])
@pytest.mark.parametrize("penalty", [0.0, 0.2])
@pytest.mark.parametrize("outputs", [1, 2])
def test_all_parameter_gradients_match_complete_loss(
    batch_norm, dropout, penalty, outputs
):
    x = np.array([[-1.0, 0.3], [0.2, -0.5], [1.0, 0.7], [0.5, -0.4]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    if outputs == 2:
        y = np.column_stack([1 - y.ravel(), y.ravel()])
    model = NeuralNetwork(
        architecture(batch_norm, dropout, penalty, outputs),
        m_scale=1.7,
        print_cost=False,
    )
    np.random.seed(93)
    parameters = model.initialize_wb(x)
    state = {
        prefix + str(i): value
        for i in range(2)
        for prefix, value in [("mu_r", 0.0), ("var_r", 1.0)]
    }

    def evaluate(values):
        np.random.seed(121)
        result = model.forward_prop(x, values, deepcopy(state), train=True)
        prediction, _, _, _, _, regularizers = result
        loss = model.activations[model.model["output"]["cost"]](y, prediction)
        return (loss + sum(regularizers.values())) / (len(x) * model.m_scale), result

    _, result = evaluate(parameters)
    gradients = model.back_prop(
        x, y, parameters, result[1], result[2], result[3], result[4]
    )
    for name, values in parameters.items():
        assert gradients[name].shape == values.shape
        for index in np.ndindex(values.shape):
            plus, minus = deepcopy(parameters), deepcopy(parameters)
            plus[name][index] += 1e-5
            minus[name][index] -= 1e-5
            reference = (evaluate(plus)[0] - evaluate(minus)[0]) / 2e-5
            assert_allclose(
                gradients[name][index],
                reference,
                atol=2e-7,
                rtol=2e-4,
                err_msg=f"{name}{index}",
            )


def test_dropout_draws_preserve_normalization_state_and_infer_output_width():
    x = np.array([[-1.0, 0.3], [0.2, -0.5], [1.0, 0.7], [0.5, -0.4]])
    y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    model = NeuralNetwork(
        architecture(True, 0.75, outputs=2),
        batch_size=4,
        num_iterations=3,
        print_cost=False,
    ).fit(x, y)
    before = deepcopy(model.batch_norm)
    predicted = model.predict(x)
    draws = model.draw_predictive_samples(x, n_samples=5)
    assert draws.shape == (4, 10)
    assert_allclose(draws.reshape(4, 5, 2).sum(axis=2), 1)
    assert draws.reshape(4, 5, 2).std(axis=1).max() > 0
    for name in before:
        assert_allclose(model.batch_norm[name], before[name])
    assert_allclose(model.predict(x), predicted)
    with pytest.raises(ValueError, match="output"):
        model.draw_predictive_samples(x, n_samples=2, n_outputs=1)


def test_gradient_descent_training_applies_the_recorded_gradient():
    x = np.array([[-1.0, 0.3], [0.2, -0.5], [1.0, 0.7], [0.5, -0.4]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = NeuralNetwork(
        architecture(),
        optimizer="gradient_descent",
        optimizer_args={"learning_rate": 0.07},
        batch_size=4,
        num_iterations=1,
        print_cost=False,
        random_state=42,
    )
    np.random.seed(42)
    initial = model.initialize_wb(x)
    output = model.forward_prop(x, initial, {}, train=True)
    gradient = model.back_prop(
        x, y, initial, output[1], output[2], output[3], output[4]
    )
    model.fit(x, y)
    for key in initial:
        assert_allclose(model.wb[key], initial[key] - 0.07 * gradient[key], atol=1e-12)


def test_adam_keeps_moments_between_updates_and_visits_each_row_once():
    x = np.column_stack([np.arange(8), np.arange(8) / 10])
    y = (np.arange(8) % 2).reshape(-1, 1)
    model = NeuralNetwork(
        architecture(), batch_size=3, num_iterations=2, shuffle=True, print_cost=False
    )
    original_forward = model.forward_prop
    original_update = model.update_parameters
    visited, steps, prior_moments = [], [], []

    def forward(values, *args, **kwargs):
        visited.extend(values[:, 0].tolist())
        return original_forward(values, *args, **kwargs)

    def update(parameters, gradient, options):
        steps.append(options["i"])
        prior_moments.append(
            sum(np.abs(value).sum() for value in options["mt"].values())
        )
        return original_update(parameters, gradient, options)

    model.forward_prop = forward
    model.update_parameters = update
    model.fit(x, y)
    assert steps == list(range(6))
    assert prior_moments[0] == 0 and all(value > 0 for value in prior_moments[1:])
    assert sorted(visited[:8]) == sorted(visited[8:]) == list(range(8))


def test_callable_layer_components_work_and_prediction_preserves_random_state():
    specification = architecture()
    for layer in specification.values():
        layer["initializer"] = initializers.xavier_uniform
        layer["activation"] = activations.sigmoid
    specification["hidden"]["derivative"] = activations.sigmoid_der
    specification["output"]["derivative"] = activations.sigmoid_cross_entropy_binary_der
    specification["output"]["cost"] = activations.cross_entropy_binary
    x = np.array([[-1.0, 0.3], [0.2, -0.5], [1.0, 0.7], [0.5, -0.4]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = NeuralNetwork(specification, num_iterations=2, print_cost=False).fit(x, y)
    before = np.random.get_state()
    prediction = model.predict(x)
    after = np.random.get_state()
    assert np.isfinite(prediction).all()
    assert before[0] == after[0]
    assert_allclose(before[1], after[1])
    assert before[2:] == after[2:]
