#!/usr/bin/env Rscript
# validate_desmedt.R — Cross-validate Python REML against metafor
# Dataset: Desmedt et al. (2022) HCT criterion extraction (k=7)
#
# IMPORTANT: Python uses Fisher-z transformed values and vi_z = 1/(N_approx - 3).
# Two analyses: (1) absolute |r| with dot as 1 vs linear as 0 (linear=reference),
# (2) signed r with same coding.
#
# Outputs: validate_metafor/output_desmedt.csv

library(metafor)

root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", "..", ".."))
data_path <- file.path(root, "evidence_synthesis", "extraction",
                       "desmedt2022_criterion_extraction.csv")
out_path  <- file.path(root, "evidence_synthesis", "analysis",
                       "validate_metafor", "output_desmedt.csv")

d <- read.csv(data_path, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows from %s\n", nrow(d), data_path))

# Match Python exactly
d$abs_r    <- abs(d$r_pooled)
d$z_abs    <- atanh(pmin(pmax(d$abs_r, -0.999), 0.999))
d$z_signed <- atanh(pmin(pmax(d$r_pooled, -0.999), 0.999))
d$vi_z     <- 1.0 / (d$N_approx - 3.0)

# Python uses is_dot dummy (linear=reference, dot=1):
# X_mod_abs = [ones, is_dot]  where names = ["Intercept", "stage[dot]"]
d$is_dot <- as.numeric(d$dln_stage_code == "dot")

results <- data.frame(parameter = character(), metafor_value = numeric(),
                      stringsAsFactors = FALSE)
add <- function(param, val) {
  results[nrow(results) + 1, ] <<- list(param, as.numeric(val))
}

# ============================================================
# Analysis 1: Absolute |r| (cross-level mismatch)
# ============================================================

# --- Baseline ---
res0_abs <- rma(yi = z_abs, vi = vi_z, data = d, method = "REML")
add("abs_base_tau2",   res0_abs$tau2)
add("abs_base_I2",     res0_abs$I2 / 100)
add("abs_base_Q",      res0_abs$QE)
add("abs_base_mu",     coef(res0_abs))
add("abs_base_se_mu",  res0_abs$se)
add("abs_base_ci_lo",  res0_abs$ci.lb)
add("abs_base_ci_hi",  res0_abs$ci.ub)

# --- Moderator: is_dot (linear as reference) ---
res1_abs <- rma(yi = z_abs, vi = vi_z, mods = ~ is_dot, data = d,
                method = "REML", test = "knha")
add("abs_mod_tau2",       res1_abs$tau2)
add("abs_mod_I2",         res1_abs$I2 / 100)
add("abs_mod_Q",          res1_abs$QE)
add("abs_mod_beta_intcp", coef(res1_abs)[1])
add("abs_mod_se_intcp",   res1_abs$se[1])
add("abs_mod_ci_lo_intcp", res1_abs$ci.lb[1])
add("abs_mod_ci_hi_intcp", res1_abs$ci.ub[1])
add("abs_mod_beta_dot",   coef(res1_abs)[2])
add("abs_mod_se_dot",     res1_abs$se[2])
add("abs_mod_ci_lo_dot",  res1_abs$ci.lb[2])
add("abs_mod_ci_hi_dot",  res1_abs$ci.ub[2])
add("abs_mod_QM",   res1_abs$QM)
add("abs_mod_QM_p", res1_abs$QMp)

# ============================================================
# Analysis 2: Signed r
# ============================================================

# --- Baseline ---
res0_sign <- rma(yi = z_signed, vi = vi_z, data = d, method = "REML")
add("signed_base_tau2",   res0_sign$tau2)
add("signed_base_I2",     res0_sign$I2 / 100)
add("signed_base_Q",      res0_sign$QE)
add("signed_base_mu",     coef(res0_sign))

# --- Moderator: is_dot (linear as reference) ---
res1_sign <- rma(yi = z_signed, vi = vi_z, mods = ~ is_dot, data = d,
                 method = "REML", test = "knha")
add("signed_mod_tau2",       res1_sign$tau2)
add("signed_mod_beta_intcp", coef(res1_sign)[1])
add("signed_mod_beta_dot",   coef(res1_sign)[2])
add("signed_mod_se_intcp",   res1_sign$se[1])
add("signed_mod_se_dot",     res1_sign$se[2])

# --- Egger's test ---
reg <- regtest(res0_abs, model = "lm")
add("abs_egger_z", reg$zval)
add("abs_egger_p", reg$pval)

# Profile-likelihood CIs
ci0 <- confint(res0_abs)
add("abs_base_tau2_pl_lo", ci0$random["tau^2", "ci.lb"])
add("abs_base_tau2_pl_hi", ci0$random["tau^2", "ci.ub"])

ci1 <- confint(res1_abs)
add("abs_mod_tau2_pl_lo", ci1$random["tau^2", "ci.lb"])
add("abs_mod_tau2_pl_hi", ci1$random["tau^2", "ci.ub"])

# --- Write output ---
write.csv(results, out_path, row.names = FALSE)
cat(sprintf("Wrote %d parameters to %s\n", nrow(results), out_path))
print(results)
