# Fit R's warpbreaks data with quasi-Poisson and generalized Poisson models.
# The Bayesian target matches tsdst's fixed-C posterior, including its priors.
# Emit small CSV sections so the Python comparison can retain the evidence.

arguments <- commandArgs(trailingOnly = TRUE)
iterations <- if (length(arguments)) as.integer(arguments[1]) else 100000L
burn <- as.integer(iterations / 5)
control <- glm.control(epsilon = 1e-12, maxit = 100)
data <- warpbreaks
design <- model.matrix(~ wool + tension, data)
counts <- data$breaks
quasi <- glm(breaks ~ wool + tension, data, family = quasipoisson(), control = control)
poisson <- glm(breaks ~ wool + tension, data, family = poisson(), control = control)
stopifnot(max(abs(fitted(quasi) - fitted(poisson))) < 1e-8)

log_likelihood <- function(parameters) {
    delta <- parameters[ncol(design) + 1L]
    if (!is.finite(delta) || delta < 0 || delta >= 1) return(-Inf)
    theta <- exp(drop(design %*% parameters[seq_len(ncol(design))])) * (1 - delta)
    shifted <- theta + delta * counts
    if (any(!is.finite(theta)) || any(theta <= 0) || any(shifted <= 0)) return(-Inf)
    sum(log(theta) + (counts - 1) * log(shifted) - theta - delta * counts - lgamma(counts + 1))
}

log_posterior <- function(parameters) {
    value <- log_likelihood(parameters)
    if (!is.finite(value)) return(-Inf)
    beta <- parameters[2:ncol(design)]
    delta <- parameters[ncol(design) + 1L]
    # These priors reproduce the current tsdst implementation exactly.
    value + dnorm(parameters[1], 0, 10, log = TRUE) +
        sum(-log(4000) - abs(beta) / 2000) + dnorm(exp(delta), 0, 0.4, log = TRUE)
}

initial <- c(coef(quasi), 1 - 1 / sqrt(summary(quasi)$dispersion))
objective <- function(parameters, posterior) {
    value <- if (posterior) log_posterior(parameters) else log_likelihood(parameters)
    if (is.finite(value)) -value else 1e100
}
mle <- optim(initial, objective, posterior = FALSE, method = "BFGS",
             control = list(reltol = 1e-12, maxit = 10000))
mode <- optim(initial, objective, posterior = TRUE, method = "BFGS", hessian = TRUE,
              control = list(reltol = 1e-12, maxit = 10000))
stopifnot(mle$convergence == 0, mode$convergence == 0)
proposal <- solve(mode$hessian) * 2.38^2 / length(initial)
proposal_chol <- t(chol(proposal))

chains <- lapply(c(101L, 202L, 303L, 404L), function(seed) {
    set.seed(seed)
    current <- mode$par
    value <- log_posterior(current)
    draws <- matrix(NA_real_, iterations, length(initial))
    accepted <- 0L
    for (i in seq_len(iterations)) {
        candidate <- current + drop(proposal_chol %*% rnorm(length(initial)))
        next_value <- log_posterior(candidate)
        if (is.finite(next_value) && log(runif(1)) < next_value - value) {
            current <- candidate
            value <- next_value
            accepted <- accepted + 1L
        }
        draws[i, ] <- current
    }
    attr(draws, "acceptance") <- accepted / iterations
    draws
})

retained <- lapply(chains, function(draws) draws[(burn + 1L):iterations, , drop = FALSE])
summarize <- function(draws, names) {
    pooled <- do.call(rbind, draws)
    n <- nrow(draws[[1]])
    batch_size <- as.integer(sqrt(n))
    batches <- lapply(draws, function(chain) {
        sapply(seq_len(ncol(chain)), function(j) {
            colMeans(matrix(chain[seq_len((n %/% batch_size) * batch_size), j], nrow = batch_size))
        })
    })
    batch_variance <- Reduce("+", lapply(batches, function(batch) apply(batch, 2, var))) / length(draws)
    mcse <- sqrt(batch_size * batch_variance / nrow(pooled))
    halves <- unlist(lapply(draws, function(chain) {
        half <- nrow(chain) %/% 2L
        list(chain[seq_len(half), , drop = FALSE], chain[(half + 1L):(2L * half), , drop = FALSE])
    }), recursive = FALSE)
    within <- Reduce("+", lapply(halves, function(chain) apply(chain, 2, var))) / length(halves)
    between <- apply(do.call(rbind, lapply(halves, colMeans)), 2, var)
    half_n <- nrow(halves[[1]])
    rhat <- sqrt(((half_n - 1) / half_n * within + between) / within)
    data.frame(name = names, mean = colMeans(pooled), sd = apply(pooled, 2, sd),
               mcse = mcse, rhat = rhat, ess = apply(pooled, 2, var) / mcse^2,
               lower = apply(pooled, 2, quantile, 0.025), upper = apply(pooled, 2, quantile, 0.975))
}

parameter_names <- c(colnames(design), "delta")
parameters <- summarize(retained, parameter_names)
parameters$mle <- mle$par
parameters$mode <- mode$par
parameters$quasi <- c(coef(quasi), NA_real_)
parameters$all_draw_mean <- colMeans(do.call(rbind, chains))
unique_rows <- !duplicated(design)
prediction_design <- design[unique_rows, , drop = FALSE]
prediction_draws <- lapply(retained, function(chain) exp(chain[, seq_len(ncol(design)), drop = FALSE] %*% t(prediction_design)))
predictions <- summarize(prediction_draws, paste(data$wool[unique_rows], data$tension[unique_rows], sep = ":"))
predictions$quasi <- fitted(quasi)[unique_rows]
predictions$mle <- exp(drop(prediction_design %*% mle$par[seq_len(ncol(design))]))
predictions$mode <- exp(drop(prediction_design %*% mode$par[seq_len(ncol(design))]))
prediction_coefficients <- parameters$mean[seq_len(ncol(design))]
predictions$plugin <- exp(drop(prediction_design %*% prediction_coefficients))
prediction_delta <- parameters$mean[length(initial)]
predictions$old_plugin <- predictions$plugin * (1 - prediction_delta)

evaluations <- rbind(initial, mode$par, mode$par + c(0.03, -0.02, 0.01, 0.02, 0.03))
evaluation_table <- data.frame(evaluations, log_posterior = apply(evaluations, 1, log_posterior))
covariance <- data.frame(proposal)
data_table <- data.frame(data, design[, -1, drop = FALSE], quasi_prediction = fitted(quasi))
metadata <- data.frame(r_version = R.version.string, iterations = iterations, burn = burn,
                       C = 1000, quasi_phi = summary(quasi)$dispersion,
                       bayesian_phi = 1 / (1 - prediction_delta)^2,
                       acceptance_min = min(sapply(chains, attr, "acceptance")),
                       acceptance_max = max(sapply(chains, attr, "acceptance")),
                       negative_delta_fraction = mean(do.call(rbind, retained)[, length(initial)] < 0))
options(digits = 16)
emit <- function(name, value) {
    cat(paste0("@@", name, "@@\n"))
    write.csv(value, stdout(), row.names = FALSE)
}
emit("DATA", data_table)
emit("PARAMETERS", parameters)
emit("PREDICTIONS", predictions)
emit("EVALUATIONS", evaluation_table)
emit("PROPOSAL", covariance)
emit("METADATA", metadata)
