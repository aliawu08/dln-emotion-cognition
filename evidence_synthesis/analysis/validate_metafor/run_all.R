#!/usr/bin/env Rscript
# run_all.R — Run all metafor validation scripts and report status.
#
# Usage (from repo root):
#   Rscript evidence_synthesis/analysis/validate_metafor/run_all.R
#
# Or from the validate_metafor directory:
#   Rscript run_all.R
#
# Prerequisites:
#   install.packages(c("metafor", "clubSandwich"))

cat("=== metafor Cross-Validation Suite ===\n\n")

# Find script directory
script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)

scripts <- c(
  "validate_webb_strategy.R",
  "validate_greenwald.R",
  "validate_hoyt.R",
  "validate_desmedt.R",
  "validate_interoception.R",
  "validate_webb_comparison.R"
)

status <- character(length(scripts))

for (i in seq_along(scripts)) {
  script_path <- file.path(script_dir, scripts[i])
  cat(sprintf("\n--- [%d/%d] %s ---\n", i, length(scripts), scripts[i]))

  tryCatch({
    source(script_path, local = TRUE)
    status[i] <- "OK"
    cat(sprintf("  -> PASS\n"))
  }, error = function(e) {
    status[i] <<- paste("FAIL:", conditionMessage(e))
    cat(sprintf("  -> FAIL: %s\n", conditionMessage(e)))
  })
}

cat("\n\n=== SUMMARY ===\n")
for (i in seq_along(scripts)) {
  cat(sprintf("  %-35s %s\n", scripts[i], status[i]))
}

n_pass <- sum(status == "OK")
cat(sprintf("\n%d/%d scripts completed successfully.\n", n_pass, length(scripts)))

if (n_pass == length(scripts)) {
  cat("All output CSVs written. Run validate_reml.py to compare against Python.\n")
} else {
  cat("Some scripts failed. Check error messages above.\n")
}
