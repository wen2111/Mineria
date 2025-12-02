library(FactoMineR)
library(factoextra)

# --- Dimensiones fijas por cada dataset ---
ncp_map <- list(Imputado=19, reducido=8, reducido_plus=10, transformada=28)

# ---------- Función principal ----------
prep_famd_fixed_ncp <- function(
    data,
    dataset_key,                        # "Imputado" | "reducido" | "reducido_plus" | "transformada"
    ncp_map,
    target = "Exited",                  # variable objetivo
    id_cols = c("group","ID","ClienteID"),
    train_prop = 0.7,
    seed = 123,
    filter_official_train = TRUE,       # si existe group=="train", filtrar antes
    save_dir = "."                      # carpeta donde guardar .rds
){
  stopifnot(is.data.frame(data))
  if (!dataset_key %in% names(ncp_map)) {
    stop("dataset_key no está en ncp_map: ", dataset_key)
  }
  ncp_fixed <- ncp_map[[dataset_key]]
  set.seed(seed)
  
  # 1) Filtra oficial train si procede
  if (filter_official_train && "group" %in% names(data) && any(data$group == "train", na.rm = TRUE)) {
    data <- subset(data, group == "train")
  }
  
  # 2) Quita columnas ID/aux
  cols_to_drop <- intersect(id_cols, names(data))
  if (length(cols_to_drop) > 0) data <- subset(data, select = setdiff(names(data), cols_to_drop))
  
  # 3) Check target y limpieza de NA en target
  if (!target %in% names(data)) stop("No se encuentra la columna target: ", target)
  data <- data[!is.na(data[[target]]), , drop = FALSE]
  
  # 4) Pasar character -> factor (coherencia con FAMD)
  char_cols <- names(data)[vapply(data, is.character, logical(1))]
  for (nm in char_cols) data[[nm]] <- factor(data[[nm]])
  
  # 5) Split interno
  n <- nrow(data)
  idx_train <- sample.int(n, size = floor(train_prop * n))
  df_train  <- data[idx_train, , drop = FALSE]
  df_test   <- data[-idx_train, , drop = FALSE]
  
  # 6) X/y
  ytr <- df_train[[target]]; if (!is.factor(ytr)) ytr <- factor(ytr)
  yte <- df_test[[target]]; if (!is.factor(yte)) yte <- factor(yte)
  x_train <- df_train[, setdiff(names(df_train), target), drop = FALSE]
  x_test  <- df_test[,  setdiff(names(df_test),  target), drop = FALSE]
  
  # 7) Alinear niveles factor en test
  for (nm in names(x_train)) {
    if (is.factor(x_train[[nm]])) {
      x_test[[nm]] <- factor(x_test[[nm]], levels = levels(x_train[[nm]]))
    }
  }
  
  # 8) Ajustar FAMD SOLO con train (ncp = Inf para obtener todas y recortar luego)
  famd_fit <- FAMD(x_train, ncp = Inf, graph = FALSE)
  
  # 9) Coords de train y proyección de test
  coord_train_full <- famd_fit$ind$coord
  coord_test_full  <- predict(famd_fit, newdata = x_test)$coord
  
  # 10) Recorte a ncp fijo según mapa (con tope de columnas disponibles)
  max_dim_train <- ncol(coord_train_full)
  max_dim_test  <- ncol(coord_test_full)
  ncp_use <- min(ncp_fixed, max_dim_train, max_dim_test)
  if (ncp_fixed > max_dim_train) {
    warning(sprintf("ncp_map[%s]=%d > dim disponibles (%d). Se usará %d.",
                    dataset_key, ncp_fixed, max_dim_train, ncp_use))
  }
  coord_train <- coord_train_full[, seq_len(ncp_use), drop = FALSE]
  coord_test  <- coord_test_full[,  seq_len(ncp_use),  drop = FALSE]
  
  # 11) Guardar a disco
  train_file <- file.path(save_dir, paste0("famd_coords_", dataset_key, "_train.rds"))
  test_file  <- file.path(save_dir, paste0("famd_coords_", dataset_key, "_test.rds"))
  saveRDS(coord_train, train_file)
  saveRDS(coord_test,  test_file)
  
  # 12) Salida
  list(
    Xtr = coord_train,
    Xte = coord_test,
    ytr = ytr,
    yte = yte,
    ncp_used = ncp_use,
    famd_model = famd_fit,
    saved_files = list(train=train_file, test=test_file)
  )
}



# Imputado
res_imp <- prep_famd_fixed_ncp(
  data = data_imputado,
  dataset_key = "Imputado",
  ncp_map = ncp_map,
  target = "Exited",
  id_cols = c("group","ID","ClienteID"),
  train_prop = 0.7,
  seed = 123,
  filter_official_train = TRUE,  # primero group=='train'
  save_dir = "."                 # guarda .rds en el directorio actual
)

# Reducido
res_red <- prep_famd_fixed_ncp(
  data = data_reducida, dataset_key = "reducido",
  ncp_map = ncp_map, target = "Exited",
  id_cols = c("group","ID","ClienteID"),
  filter_official_train = FALSE  # si este objeto no trae 'group', pon FALSE
)

# Reducido plus
res_rp <- prep_famd_fixed_ncp(
  data = data_reducida_plus, dataset_key = "reducido_plus",
  ncp_map = ncp_map, target = "Exited",
  id_cols = c("group","ID","ClienteID"),
  filter_official_train = TRUE
)

# Transformada
res_t <- prep_famd_fixed_ncp(
  data = data_transformada, dataset_key = "transformada",
  ncp_map = ncp_map, target = "Exited",
  id_cols = c("group","ID","ClienteID"),
  filter_official_train = TRUE
)


save.image("famd_auto.RData")
