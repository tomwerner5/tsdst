# Generate numerical audit references using the authors' R implementations.
# Arguments: output folder; optional folder containing coda and sAIC sources.
# Otherwise install coda and sAIC in the R library before running this script.

arguments <- commandArgs(trailingOnly = TRUE)
folder <- arguments[1]
if (length(arguments) > 1) {
    for (file in list.files(file.path(arguments[2], "coda", "R"), full.names = TRUE)) source(file)
    source(file.path(arguments[2], "sAIC", "R", "sAIC.R"))
} else {
    library(coda)
    library(sAIC)
}
dir.create(folder, recursive = TRUE, showWarnings = FALSE)
control <- glm.control(epsilon = 1e-14, maxit = 100)
x <- cbind(seq(-1.5, 1.5, length.out = 60), sin(seq_len(60)/4), cos(seq_len(60)/7))
data <- data.frame(x1 = x[, 1], x2 = x[, 2], x3 = x[, 3],
                   binary = as.integer((seq_len(60) %% 4) == 0 | (x[, 1] > .8 & seq_len(60) %% 2 == 0)),
                   counts = rep(c(0, 1, 2, 5, 3), 12) + as.integer(x[, 1] > .5))
write.csv(data, file.path(folder, "aic-data.csv"), row.names = FALSE)
answers <- list()
coefs <- list()
for (family in c("binomial", "poisson")) {
    y <- if (family == "binomial") data$binary else data$counts
    fit <- glm(y ~ x, family = family, control = control)
    coefs[[family]] <- data.frame(family = family, parameter = seq_len(4), value = coef(fit))
    for (intercept in c(0, .4)) {
        beta <- c(intercept, .3, 0, -.2)
        value <- sAIC(x, y, beta, family)$AIC
        answers[[length(answers) + 1L]] <- data.frame(family = family, intercept = intercept, AIC = value)
    }
}
write.csv(do.call(rbind, answers), file.path(folder, "aic-reference.csv"), row.names = FALSE)
write.csv(do.call(rbind, coefs), file.path(folder, "aic-unregularized-coefficients.csv"), row.names = FALSE)

chain <- as.matrix(read.csv(file.path(folder, "raftery-chain.csv")))
raftery_answers <- list()
for (q in c(.025, .5, .975)) for (spacing in c(1, 2)) {
    result <- raftery.diag(mcmc(chain, thin = spacing), q = q, r = .02, s = .95)$resmatrix
    raftery_answers[[length(raftery_answers) + 1L]] <- data.frame(
        q = q, spacing = spacing, parameter = seq_len(ncol(chain)),
        M = result[, "M"], N = result[, "N"], Nmin = result[, "Nmin"], I = result[, "I"])
}
write.csv(do.call(rbind, raftery_answers), file.path(folder, "raftery-reference.csv"), row.names = FALSE)

regression <- data.frame(x1 = x[, 1], x2 = x[, 2],
                          y = 2 + .7*x[, 1] - .4*x[, 2] + sin(seq_len(60)*1.7)/3)
linear <- lm(y ~ x1 + x2, regression)
prediction_data <- data.frame(x1 = c(-1.1, .2, 1.3), x2 = c(.4, -.3, .5))
confidence <- predict(linear, prediction_data, interval = "confidence")
prediction <- predict(linear, prediction_data, interval = "prediction")
write.csv(regression, file.path(folder, "ols-data.csv"), row.names = FALSE)
write.csv(data.frame(prediction_data, confidence, prediction_lower = prediction[, 2],
                     prediction_upper = prediction[, 3]), file.path(folder, "ols-intervals.csv"), row.names = FALSE)
write.csv(data.frame(AIC = AIC(linear), logLik = as.numeric(logLik(linear)), parameters = attr(logLik(linear), "df")),
          file.path(folder, "ols-aic.csv"), row.names = FALSE)
write.csv(vcov(linear), file.path(folder, "ols-covariance.csv"), row.names = FALSE)

logistic <- glm(binary ~ x1 + x2 + x3, data, family = binomial(), control = control)
logistic_prediction <- predict(logistic, data[1:3, ], type = "link", se.fit = TRUE)
z <- qnorm(.975)
write.csv(data.frame(data[1:3, c("x1", "x2", "x3")], eta = logistic_prediction$fit,
                     lower = logistic_prediction$fit-z*logistic_prediction$se.fit,
                     upper = logistic_prediction$fit+z*logistic_prediction$se.fit),
          file.path(folder, "logistic-intervals.csv"), row.names = FALSE)

times <- c(.4, 1.1, 2.3, 3.5, 5.)
shape <- exp(log(2) + .2*c(-2, -.5, 0, 1, 2))
scale <- 3
events <- c(1, 0, 1, 0, 1)
hazard <- -pweibull(times, shape, scale, lower.tail = FALSE, log.p = TRUE)
martingale <- events-hazard
deviance <- sign(martingale)*sqrt(2*(hazard-events-events*log(hazard)))
write.csv(data.frame(time = times, shape = shape, mean = scale*gamma(1+1/shape),
                     variance = scale^2*(gamma(1+2/shape)-gamma(1+1/shape)^2),
                     logpdf = dweibull(times, shape, scale, log = TRUE),
                     logsf = pweibull(times, shape, scale, lower.tail = FALSE, log.p = TRUE),
                     events = events, deviance = deviance),
          file.path(folder, "weibull-reference.csv"), row.names = FALSE)
cat(R.version.string, "\nReference files saved in", folder, "\n")
