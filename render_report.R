## Small helper to render the Quarto document to HTML
## If Quarto CLI is installed but not on PATH / not visible to R, set QUARTO_PATH
## for this session to the default user install location (common on Windows).
default_quarto_bin <- "C:/Users/EmmanuelAdewuyi/AppData/Local/Programs/Quarto/bin"
if (Sys.getenv("QUARTO_PATH") == "" && dir.exists(default_quarto_bin)) {
  Sys.setenv(QUARTO_PATH = default_quarto_bin)
  message("Set QUARTO_PATH for this R session to: ", default_quarto_bin)
}

if (!requireNamespace("quarto", quietly = TRUE)) {
  message("`quarto` R package is not installed. You can render using the `quarto` CLI instead: `quarto render documentation.qmd`.")
} else {
  # Safe check whether quarto CLI is discoverable
  q <- tryCatch(quarto::find_quarto(), error = function(e) NULL)
  if (is.null(q)) {
    message("Quarto CLI not found by the R package. Ensure Quarto is on PATH or set QUARTO_PATH, then restart R.")
    message("Detected common install location: ", default_quarto_bin)
  } else {
    message("Quarto found at: ", q)
    quarto::quarto_render("documentation.qmd")
  }
}



