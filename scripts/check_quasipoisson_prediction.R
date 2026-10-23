# Compare Poisson and quasi-Poisson response predictions on fixed counts.
# Run with: Rscript --vanilla scripts/check_quasipoisson_prediction.R

control <- glm.control(epsilon = 1e-12, maxit = 100)
counts <- c(0, 0, 0, 0, 10, 10, 10, 10)

check_case <- function(name, data, formula, newdata, expected, expected_phi) {
    poisson_fit <- glm(formula, data = data, family = poisson(), control = control)
    quasi_fit <- glm(formula, data = data, family = quasipoisson(), control = control)
    response <- unname(predict(quasi_fit, newdata, type = "response"))
    eta <- unname(predict(quasi_fit, newdata, type = "link"))
    phi <- summary(quasi_fit)$dispersion
    unit <- predict(quasi_fit, newdata, type = "response", se.fit = TRUE, dispersion = 1)
    four <- predict(quasi_fit, newdata, type = "response", se.fit = TRUE, dispersion = 4)
    stopifnot(
        isTRUE(all.equal(response, expected, tolerance = 1e-8)),
        isTRUE(all.equal(exp(eta), response, tolerance = 1e-8)),
        isTRUE(all.equal(phi, expected_phi, tolerance = 1e-8)),
        isTRUE(all.equal(unname(predict(poisson_fit, newdata, type = "response")),
                         response, tolerance = 1e-8)),
        isTRUE(all.equal(unit$fit, four$fit, tolerance = 1e-8)),
        isTRUE(all.equal(four$se.fit, 2 * unit$se.fit, tolerance = 1e-8))
    )
    # Match the variance-to-mean ratio to generalized Poisson for comparison.
    # This does not fit a generalized Poisson model or estimate an MCMC delta.
    delta <- 1 - 1 / sqrt(phi)
    data.frame(case = name, feature = newdata$x, eta = eta, response = response,
               phi = phi, delta = delta, theta = response * (1 - delta),
               se_dispersion1 = unname(unit$se.fit),
               se_dispersion4 = unname(four$se.fit))
}

intercept <- check_case(
    "intercept", data.frame(y = counts), y ~ 1, data.frame(x = 0),
    expected = 5, expected_phi = 40 / 7
)
groups <- check_case(
    "groups", data.frame(x = rep(c(0, 1), each = 8), y = c(counts, 2 * counts)),
    y ~ x, data.frame(x = c(0, 1)), expected = c(5, 10), expected_phi = 60 / 7
)
options(digits = 16)
write.csv(rbind(intercept, groups), stdout(), row.names = FALSE)
