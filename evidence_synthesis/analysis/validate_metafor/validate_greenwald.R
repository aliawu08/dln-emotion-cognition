#!/usr/bin/env Rscript
# validate_greenwald.R — Cross-validate Python REML against metafor
# Dataset: Greenwald et al. (2009) study-level extraction (k=184)
#
# The DLN coding is applied at runtime from topic domain (matching Python).
# Primary outcome: ICC (Fisher-z transformed), with 3-level coding including
# mixed_unclear as a fourth level.
#
# Outputs: validate_metafor/output_greenwald.csv

library(metafor)

root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", "..", ".."))
data_path <- file.path(root, "evidence_synthesis", "extraction",
                       "greenwald2009_study_extraction.csv")
out_path  <- file.path(root, "evidence_synthesis", "analysis",
                       "validate_metafor", "output_greenwald.csv")

d <- read.csv(data_path, stringsAsFactors = FALSE)
cat(sprintf("Loaded %d rows from %s\n", nrow(d), data_path))

# Apply DLN coding (must match Python exactly)
topic_to_dln <- c(
  "Consumer"        = "dot",
  "Race (Bl/Wh)"   = "linear",
  "Politics"        = "linear",
  "Gender/sex"      = "linear",
  "Other intergroup"= "linear",
  "Relationships"   = "network",
  "Personality"     = "mixed_unclear",
  "Drugs/tobacco"   = "mixed_unclear",
  "Clinical"        = "mixed_unclear"
)
d$dln_stage <- topic_to_dln[d$topic]
stopifnot(!any(is.na(d$dln_stage)))

# Fisher-z transform ICC
d$yi_icc <- atanh(pmin(pmax(d$icc, -0.999), 0.999))
d$vi_icc <- 1 / (d$n - 3)

results <- data.frame(parameter = character(), metafor_value = numeric(),
                      stringsAsFactors = FALSE)
add <- function(param, val) {
  results[nrow(results) + 1, ] <<- list(param, as.numeric(val))
}

# --- 1. Baseline ICC (all k=184) ---
res0 <- rma(yi = yi_icc, vi = vi_icc, data = d, method = "REML")
add("icc_base_tau2",   res0$tau2)
add("icc_base_I2",     res0$I2 / 100)
add("icc_base_Q",      res0$QE)
add("icc_base_mu",     coef(res0))
add("icc_base_se_mu",  res0$se)
add("icc_base_ci_lo",  res0$ci.lb)
add("icc_base_ci_hi",  res0$ci.ub)

# --- 2. DLN moderator on ICC (all k=184, 4 levels incl mixed) ---
# Python uses design_matrix_stage with reference="dot" which creates:
# Intercept + stage[linear] + stage[network]
# But mixed_unclear is a 4th level, so design_matrix_stage would
# not include it. Looking at the Python code, it calls:
#   X_mod, names_mod = design_matrix_stage(sub["dln_stage"], reference="dot")
# design_matrix_stage only has levels ["dot","linear","network"]
# so mixed_unclear rows get NaN/dropped? No — looking at the code:
# pd.Categorical(stage, categories=levels) will map anything not in levels
# to NaN, so the dummy columns will be 0 for mixed_unclear rows,
# meaning mixed_unclear is ABSORBED into the intercept along with dot.
#
# We need to match this exactly in R:
d$stage_3level <- factor(
  ifelse(d$dln_stage %in% c("dot", "linear", "network"), d$dln_stage, "dot"),
  levels = c("dot", "linear", "network")
)
# Actually let's be precise: in Python, Categorical with categories=[dot,linear,network]
# will code mixed_unclear as neither linear nor network, so its dummies are (0,0),
# same as dot. This means mixed_unclear is pooled with dot in the intercept.

res1 <- rma(yi = yi_icc, vi = vi_icc, mods = ~ stage_3level, data = d,
            method = "REML", test = "knha")

add("icc_mod_tau2",        res1$tau2)
add("icc_mod_I2",          res1$I2 / 100)
add("icc_mod_Q",           res1$QE)
add("icc_mod_beta_intcp",  coef(res1)[1])
add("icc_mod_se_intcp",    res1$se[1])
add("icc_mod_ci_lo_intcp", res1$ci.lb[1])
add("icc_mod_ci_hi_intcp", res1$ci.ub[1])
add("icc_mod_beta_linear", coef(res1)[2])
add("icc_mod_se_linear",   res1$se[2])
add("icc_mod_ci_lo_linear", res1$ci.lb[2])
add("icc_mod_ci_hi_linear", res1$ci.ub[2])
add("icc_mod_beta_network", coef(res1)[3])
add("icc_mod_se_network",   res1$se[3])
add("icc_mod_ci_lo_network", res1$ci.lb[3])
add("icc_mod_ci_hi_network", res1$ci.ub[3])
add("icc_mod_QM",   res1$QM)
add("icc_mod_QM_p", res1$QMp)

# --- 3. Drop mixed, k=125 baseline ---
d_clean <- d[d$dln_stage %in% c("dot", "linear", "network"), ]
d_clean$stage <- factor(d_clean$dln_stage, levels = c("dot", "linear", "network"))
res2 <- rma(yi = yi_icc, vi = vi_icc, data = d_clean, method = "REML")
add("icc_dropmixed_base_tau2", res2$tau2)
add("icc_dropmixed_base_mu",   coef(res2))

res3 <- rma(yi = yi_icc, vi = vi_icc, mods = ~ stage, data = d_clean,
            method = "REML", test = "knha")
add("icc_dropmixed_mod_tau2",        res3$tau2)
add("icc_dropmixed_mod_beta_intcp",  coef(res3)[1])
add("icc_dropmixed_mod_beta_linear", coef(res3)[2])
add("icc_dropmixed_mod_beta_network", coef(res3)[3])

# --- 4. Egger's test ---
reg <- regtest(res0, model = "lm")
add("icc_egger_z", reg$zval)
add("icc_egger_p", reg$pval)

# --- Write output ---
write.csv(results, out_path, row.names = FALSE)
cat(sprintf("Wrote %d parameters to %s\n", nrow(results), out_path))
print(results)
