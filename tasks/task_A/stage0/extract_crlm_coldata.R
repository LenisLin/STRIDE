#!/usr/bin/env Rscript

required_stage0_columns <- function() {
  c("ID", "PID", "Tissue", "Position", "SubType", "Area")
}

internal_source_index_column <- function() {
  ".stage0_source_index"
}

position_pattern <- function() {
  "^\\([[:space:]]*-?[0-9.]+([eE][-+]?[0-9]+)?[[:space:]]*,[[:space:]]*-?[0-9.]+([eE][-+]?[0-9]+)?[[:space:]]*\\)$"
}

parse_position <- function(position_vec) {
  pos_chr <- as.character(position_vec)
  invalid <- is.na(pos_chr) | !grepl(position_pattern(), pos_chr)
  if (any(invalid)) {
    bad_examples <- unique(utils::head(pos_chr[invalid], 5))
    stop(
      sprintf(
        "Position contains malformed coordinates: %s",
        paste(bad_examples, collapse = " | ")
      ),
      call. = FALSE
    )
  }

  stripped <- sub("^\\(", "", pos_chr)
  stripped <- sub("\\)$", "", stripped)
  parts <- strsplit(stripped, ",", fixed = TRUE)

  data.frame(
    x = as.numeric(trimws(vapply(parts, `[`, character(1), 1))),
    y = as.numeric(trimws(vapply(parts, `[`, character(1), 2))),
    stringsAsFactors = FALSE
  )
}

extract_coldata_frame <- function(obj) {
  if (isS4(obj)) {
    coldata_attr <- attr(obj, "colData")
    if (!is.null(coldata_attr) && isS4(coldata_attr)) {
      list_data <- attr(coldata_attr, "listData")
      if (!is.null(list_data)) {
        return(as.data.frame(lapply(list_data, function(col) {
          if (is.factor(col)) {
            return(as.character(col))
          }
          col
        }), stringsAsFactors = FALSE))
      }
    }
    if (!is.null(coldata_attr) && inherits(coldata_attr, "data.frame")) {
      return(as.data.frame(lapply(coldata_attr, function(col) {
        if (is.factor(col)) {
          return(as.character(col))
        }
        col
      }), stringsAsFactors = FALSE))
    }
  }
  if (inherits(obj, "data.frame")) {
    return(as.data.frame(lapply(obj, function(col) {
      if (is.factor(col)) {
        return(as.character(col))
      }
      col
    }), stringsAsFactors = FALSE))
  }
  stop("Unsupported RDS object: expected data.frame-like colData or SingleCellExperiment", call. = FALSE)
}

filter_stage0_cells <- function(coldata_df) {
  missing <- setdiff(required_stage0_columns(), colnames(coldata_df))
  if (length(missing) > 0) {
    stop(
      sprintf("Missing required colData columns: %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }

  df <- as.data.frame(coldata_df, stringsAsFactors = FALSE)
  df[[internal_source_index_column()]] <- seq_len(nrow(df))
  df$ID <- as.character(df$ID)
  df$PID <- as.character(df$PID)
  df$Tissue <- as.character(df$Tissue)
  df$SubType <- as.character(df$SubType)
  df$Area <- as.numeric(df$Area)

  coords <- parse_position(df$Position)
  df$x <- coords$x
  df$y <- coords$y

  filtered <- df[df$Tissue %in% c("TC", "IM", "PT"), , drop = FALSE]

  roi_df <- unique(filtered[c("ID", "PID", "Tissue")])
  compartment_counts <- xtabs(~ PID + Tissue, data = roi_df)
  needed_compartments <- c("TC", "IM", "PT")

  if (!all(needed_compartments %in% colnames(compartment_counts))) {
    missing_compartments <- setdiff(needed_compartments, colnames(compartment_counts))
    stop(
      sprintf(
        "Filtered data is missing one or more canonical compartments: %s",
        paste(missing_compartments, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  eligible_patients <- rownames(compartment_counts)[
    apply(compartment_counts[, needed_compartments, drop = FALSE] >= 2, 1, all)
  ]

  filtered[filtered$PID %in% eligible_patients, , drop = FALSE]
}

assert_roi_clinical_stability <- function(filtered_df) {
  excluded <- c(
    internal_source_index_column(),
    "Position",
    "Area",
    "SubType",
    "CellID",
    "MajorType",
    "MajorType2",
    "Major_flowsom100pheno15",
    "x",
    "y"
  )
  candidate_cols <- setdiff(colnames(filtered_df), excluded)
  by_roi <- split(filtered_df, filtered_df$ID)

  unstable <- c()
  for (column in candidate_cols) {
    n_unique_per_roi <- vapply(
      by_roi,
      function(chunk) length(unique(as.character(chunk[[column]]))),
      integer(1)
    )
    if (any(n_unique_per_roi > 1L)) {
      unstable <- c(unstable, column)
    }
  }

  if (length(unstable) > 0) {
    stop(
      sprintf(
        "ROI-level clinical columns are not stable within ID: %s",
        paste(unique(unstable), collapse = ", ")
      ),
      call. = FALSE
    )
  }
}

build_roi_clinical_table <- function(filtered_df) {
  assert_roi_clinical_stability(filtered_df)

  excluded <- c(
    internal_source_index_column(),
    "Position",
    "Area",
    "SubType",
    "CellID",
    "MajorType",
    "MajorType2",
    "Major_flowsom100pheno15",
    "x",
    "y"
  )
  keep_cols <- setdiff(colnames(filtered_df), excluded)
  unique(filtered_df[keep_cols])
}

build_cell_table <- function(filtered_df) {
  keep_cols <- c("ID", "PID", "Tissue", "SubType", "Area", "x", "y")
  if ("CellID" %in% colnames(filtered_df)) {
    keep_cols <- c(keep_cols, "CellID")
  }
  filtered_df[keep_cols]
}

extract_expression_assay <- function(obj) {
  assays_attr <- attr(obj, "assays")
  assays_data <- attr(assays_attr, "data")
  assay_list <- attr(assays_data, "listData")
  if (is.null(assay_list) || length(assay_list) != 1L) {
    stop("Expected exactly one assay in the CRLM cohort object", call. = FALSE)
  }

  assay_matrix <- assay_list[[1]]
  if (!is.matrix(assay_matrix)) {
    stop("Expected the CRLM cohort assay payload to be a matrix", call. = FALSE)
  }

  assay_dimnames <- attr(assay_matrix, "dimnames")
  marker_names <- assay_dimnames[[1]]
  if (is.null(marker_names) || length(marker_names) != nrow(assay_matrix)) {
    stop("Assay matrix is missing marker names on the feature axis", call. = FALSE)
  }

  list(
    matrix = assay_matrix,
    marker_names = as.character(marker_names)
  )
}

write_expression_artifacts <- function(
  filtered_df,
  assay_matrix,
  marker_names,
  expr_out,
  markers_out,
  chunk_size = 50000L
) {
  retained_idx <- filtered_df[[internal_source_index_column()]]
  if (length(marker_names) != nrow(assay_matrix)) {
    stop("Marker count does not match the assay feature axis", call. = FALSE)
  }
  if (length(retained_idx) != nrow(filtered_df)) {
    stop("Filtered assay columns do not align with the filtered cell table", call. = FALSE)
  }

  con <- file(expr_out, open = "wb")
  on.exit(close(con), add = TRUE)
  writeLines(marker_names, con = markers_out, sep = "\n", useBytes = TRUE)
  for (start_idx in seq.int(1L, length(retained_idx), by = chunk_size)) {
    end_idx <- min(start_idx + chunk_size - 1L, length(retained_idx))
    chunk <- assay_matrix[, retained_idx[start_idx:end_idx], drop = FALSE]
    writeBin(as.numeric(chunk), con = con, size = 4L, endian = .Platform$endian)
  }
}

parse_args <- function(args) {
  if (length(args) %% 2 != 0) {
    stop("Arguments must be passed as --key value pairs", call. = FALSE)
  }

  parsed <- list()
  idx <- 1L
  while (idx <= length(args)) {
    key <- args[[idx]]
    value <- args[[idx + 1L]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Invalid argument name: %s", key), call. = FALSE)
    }
    parsed[[substring(key, 3L)]] <- value
    idx <- idx + 2L
  }
  parsed
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  required_args <- c("rds", "cells_out", "roi_clinical_out", "expr_out", "markers_out")
  missing_args <- setdiff(required_args, names(args))
  if (length(missing_args) > 0) {
    stop(
      sprintf("Missing required arguments: %s", paste(missing_args, collapse = ", ")),
      call. = FALSE
    )
  }

  obj <- readRDS(args$rds)
  coldata_df <- extract_coldata_frame(obj)
  filtered_df <- filter_stage0_cells(coldata_df)
  cell_table <- build_cell_table(filtered_df)
  roi_clinical <- build_roi_clinical_table(filtered_df)
  assay_info <- extract_expression_assay(obj)

  utils::write.csv(cell_table, file = args$cells_out, row.names = FALSE, quote = TRUE)
  utils::write.csv(roi_clinical, file = args$roi_clinical_out, row.names = FALSE, quote = TRUE)
  write_expression_artifacts(
    filtered_df = filtered_df,
    assay_matrix = assay_info$matrix,
    marker_names = assay_info$marker_names,
    expr_out = args$expr_out,
    markers_out = args$markers_out
  )
}

if (sys.nframe() == 0L) {
  main()
}
