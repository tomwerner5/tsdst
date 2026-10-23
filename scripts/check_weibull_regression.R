# Generate an independent scale-regression reference with survival::survreg.
# Pass the output directory as the first argument.
library(survival)
folder <- commandArgs(trailingOnly = TRUE)[1]
dir.create(folder, recursive = TRUE, showWarnings = FALSE)
source_data <- survival::lung
keep <- complete.cases(source_data[, c("time", "status", "age", "sex", "ph.ecog")])
data <- with(source_data[keep, ], data.frame(
    time = time, event = as.integer(status == 2), age10 = (age - 60)/10,
    female = sex - 1, ecog = ph.ecog))
fit <- survreg(Surv(time, event) ~ age10 + female + ecog,
               data = data, dist = "weibull",
               control = survreg.control(maxiter = 100, rel.tolerance = 1e-12))
beta <- coef(fit)
shape <- 1/fit$scale
scale <- exp(model.matrix(fit) %*% beta)
contribution <- ifelse(data$event == 1,
                       dweibull(data$time, shape, scale, log = TRUE),
                       pweibull(data$time, shape, scale, lower.tail = FALSE, log.p = TRUE))
write.csv(data, file.path(folder, "lung-data.csv"), row.names = FALSE)
write.csv(data.frame(parameter = c(names(beta), "log_shape"),
                     value = c(beta, log(shape))),
          file.path(folder, "survreg-parameters.csv"), row.names = FALSE)
write.csv(data.frame(shape = shape, scale = scale,
                     mean = scale*gamma(1+1/shape),
                     variance = scale^2*(gamma(1+2/shape)-gamma(1+1/shape)^2),
                     q25 = qweibull(.25, shape, scale),
                     median = predict(fit, type = "quantile", p = .5),
                     q75 = qweibull(.75, shape, scale),
                     log_contribution = contribution),
          file.path(folder, "survreg-predictions.csv"), row.names = FALSE)
write.csv(data.frame(log_likelihood = as.numeric(logLik(fit)),
                     observations = nrow(data), events = sum(data$event),
                     survreg_scale = fit$scale),
          file.path(folder, "survreg-summary.csv"), row.names = FALSE)
cat(R.version.string, "\n")
cat("survival", as.character(packageVersion("survival")), "\n")
print(coef(fit))
cat("Weibull shape:", shape, " Log likelihood:", as.numeric(logLik(fit)), "\n")
