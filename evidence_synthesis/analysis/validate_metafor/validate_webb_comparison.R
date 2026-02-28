#!/usr/bin/env Rscript
# validate_webb_comparison.R — Cross-validate three-level REML + CR2
# Dataset: Webb et al. (2012) comparison-level extraction (k=306, 190 studies)
#
# This is the most important validation: verifies the custom three-level REML
# and CR2 cluster-robust variance estimator against metafor::rma.mv() and
# clubSandwich::coef_test().
#
# Outputs: validate_metafor/output_webb_comparison.csv

library(metafor)

# clubSandwich is optional — if not installed, CR2 tests are skipped
has_club <- requireNamespace("clubSandwich", quietly = TRUE)
if (has_club) library(clubSandwich)

root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", "..", ".."))
data_path <- file.path(root, "evidence_synthesis", "extraction",
                       "webb2012_comparison_extraction.csv")
out_path  <- file.path(root, "evidence_synthesis", "analysis",
                       "validate_metafor", "output_webb_comparison.csv")

d <- read.csv(data_path, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows from %s\n", nrow(d), data_path))
cat(sprintf("Unique studies: %d\n", length(unique(d$study))))

results <- data.frame(parameter = character(), metafor_value = numeric(),
                      stringsAsFactors = FALSE)
add <- function(param, val) {
  results[nrow(results) + 1, ] <<- list(param, as.numeric(val))
}

# --- Two-level baseline (ignoring clustering, for comparison) ---
res0_2level <- rma(yi = d_composite, vi = vi, data = d, method = "REML")
add("twolevel_base_tau2", res0_2level$tau2)
add("twolevel_base_mu",   coef(res0_2level))

# --- Three-level baseline (intercept-only) ---
res0 <- rma.mv(yi = d_composite, V = vi,
               random = ~ 1 | study / comparison_id,
               data = d, method = "REML")
add("threelevel_base_sigma2_between", res0$sigma2[1])  # study level (outer)
add("threelevel_base_sigma2_within",  res0$sigma2[2])  # comparison level (inner)
add("threelevel_base_mu",     coef(res0))
add("threelevel_base_se_mu",  res0$se)
add("threelevel_base_ci_lo",  res0$ci.lb)
add("threelevel_base_ci_hi",  res0$ci.ub)
add("threelevel_base_Q",      res0$QE)

# I2 decomposition
# metafor computes I2 for rma.mv differently than our manual formula,
# so we report both the raw variance components and our own I2 formula
# to ensure the Python code matches the same calculation
v_bar <- mean(d$vi)
total_var <- res0$sigma2[1] + res0$sigma2[2] + v_bar
add("threelevel_base_I2_total",   (res0$sigma2[1] + res0$sigma2[2]) / total_var)
add("threelevel_base_I2_within",  res0$sigma2[2] / total_var)
add("threelevel_base_I2_between", res0$sigma2[1] / total_var)

# --- Three-level DLN moderator ---
d$stage <- factor(d$dln_stage, levels = c("dot", "linear", "network"))
res1 <- rma.mv(yi = d_composite, V = vi, mods = ~ stage,
               random = ~ 1 | study / comparison_id,
               data = d, method = "REML")

add("threelevel_mod_sigma2_between", res1$sigma2[1])  # study level (outer)
add("threelevel_mod_sigma2_within",  res1$sigma2[2])  # comparison level (inner)
add("threelevel_mod_beta_intcp",     coef(res1)[1])
add("threelevel_mod_se_intcp",       res1$se[1])
add("threelevel_mod_beta_linear",    coef(res1)[2])
add("threelevel_mod_se_linear",      res1$se[2])
add("threelevel_mod_beta_network",   coef(res1)[3])
add("threelevel_mod_se_network",     res1$se[3])
add("threelevel_mod_QM",  res1$QM)
add("threelevel_mod_QMp", res1$QMp)

# --- CR2 cluster-robust standard errors (if clubSandwich available) ---
if (has_club) {
  cr2 <- coef_test(res1, vcov = "CR2", cluster = d$study)
  add("cr2_se_intcp",   cr2$SE[1])
  add("cr2_se_linear",  cr2$SE[2])
  add("cr2_se_network", cr2$SE[3])
  add("cr2_df_intcp",   cr2$df[1])
  add("cr2_df_linear",  cr2$df[2])
  add("cr2_df_network", cr2$df[3])
  add("cr2_t_intcp",    cr2$tstat[1])
  add("cr2_t_linear",   cr2$tstat[2])
  add("cr2_t_network",  cr2$tstat[3])
  add("cr2_p_intcp",    cr2$p_Satt[1])
  add("cr2_p_linear",   cr2$p_Satt[2])
  add("cr2_p_network",  cr2$p_Satt[3])
  cat("clubSandwich CR2 results included\n")
} else {
  cat("clubSandwich not installed — CR2 tests skipped\n")
}

# --- Two-level DLN moderator (for comparison with standard pipeline) ---
res1_2level <- rma(yi = d_composite, vi = vi, mods = ~ stage, data = d,
                   method = "REML", test = "knha")
add("twolevel_mod_tau2",        res1_2level$tau2)
add("twolevel_mod_beta_intcp",  coef(res1_2level)[1])
add("twolevel_mod_se_intcp",    res1_2level$se[1])
add("twolevel_mod_beta_linear", coef(res1_2level)[2])
add("twolevel_mod_se_linear",   res1_2level$se[2])
add("twolevel_mod_beta_network", coef(res1_2level)[3])
add("twolevel_mod_se_network",   res1_2level$se[3])

# --- Write output ---
write.csv(results, out_path, row.names = FALSE)
cat(sprintf("Wrote %d parameters to %s\n", nrow(results), out_path))
print(results)
