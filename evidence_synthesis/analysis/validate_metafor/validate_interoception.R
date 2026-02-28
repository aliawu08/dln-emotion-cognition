#!/usr/bin/env Rscript
# validate_interoception.R — Cross-validate Python REML against metafor
# Dataset: Interoception measure extraction (k=8)
#
# IMPORTANT: Python uses Fisher-z transform and vi_z = 1/(N_total - 3*k),
# matching the conservative variance estimate for pooled measures.
#
# Outputs: validate_metafor/output_interoception.csv

library(metafor)

root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", "..", ".."))
data_path <- file.path(root, "evidence_synthesis", "extraction",
                       "interoception_measure_extraction.csv")
out_path  <- file.path(root, "evidence_synthesis", "analysis",
                       "validate_metafor", "output_interoception.csv")

d <- read.csv(data_path, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows from %s\n", nrow(d), data_path))

# Match Python exactly
d$yi   <- atanh(pmin(pmax(d$r_pooled, -0.999), 0.999))
d$vi_z <- 1.0 / (d$N_total - 3.0 * d$k)

results <- data.frame(parameter = character(), metafor_value = numeric(),
                      stringsAsFactors = FALSE)
add <- function(param, val) {
  results[nrow(results) + 1, ] <<- list(param, as.numeric(val))
}

# --- Baseline (intercept-only) ---
res0 <- rma(yi = yi, vi = vi_z, data = d, method = "REML")
add("base_tau2",   res0$tau2)
add("base_I2",     res0$I2 / 100)
add("base_Q",      res0$QE)
add("base_mu",     coef(res0))
add("base_se_mu",  res0$se)
add("base_ci_lo",  res0$ci.lb)
add("base_ci_hi",  res0$ci.ub)

# Profile-likelihood CI for tau2
ci0 <- confint(res0)
add("base_tau2_pl_lo", ci0$random["tau^2", "ci.lb"])
add("base_tau2_pl_hi", ci0$random["tau^2", "ci.ub"])

# --- DLN moderator (Knapp-Hartung) ---
d$stage <- factor(d$dln_stage_code, levels = c("dot", "linear", "network"))
res1 <- rma(yi = yi, vi = vi_z, mods = ~ stage, data = d,
            method = "REML", test = "knha")

add("mod_tau2",        res1$tau2)
add("mod_I2",          res1$I2 / 100)
add("mod_Q",           res1$QE)
add("mod_beta_intcp",  coef(res1)[1])
add("mod_se_intcp",    res1$se[1])
add("mod_ci_lo_intcp", res1$ci.lb[1])
add("mod_ci_hi_intcp", res1$ci.ub[1])
add("mod_beta_linear", coef(res1)[2])
add("mod_se_linear",   res1$se[2])
add("mod_ci_lo_linear", res1$ci.lb[2])
add("mod_ci_hi_linear", res1$ci.ub[2])
add("mod_beta_network", coef(res1)[3])
add("mod_se_network",   res1$se[3])
add("mod_ci_lo_network", res1$ci.lb[3])
add("mod_ci_hi_network", res1$ci.ub[3])

# QM
add("mod_QM",   res1$QM)
add("mod_QM_p", res1$QMp)

# Profile-likelihood CI for tau2 (moderator)
ci1 <- confint(res1)
add("mod_tau2_pl_lo", ci1$random["tau^2", "ci.lb"])
add("mod_tau2_pl_hi", ci1$random["tau^2", "ci.ub"])

# --- Egger's test ---
reg <- regtest(res0, model = "lm")
add("egger_z", reg$zval)
add("egger_p", reg$pval)

# --- Write output ---
write.csv(results, out_path, row.names = FALSE)
cat(sprintf("Wrote %d parameters to %s\n", nrow(results), out_path))
print(results)
