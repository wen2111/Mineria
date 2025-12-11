library(recipes)
library(e1071)
library(dplyr)


load("~/Documents/GitHub/Mineria/DATA/data_imputado.RData")
ID <- data_imputado$ID

load("~/Documents/GitHub/Mineria/DATA/dataaaaaaaaaaaaaa.RData")
data_reducida_plus$ID <- ID

# =========================
# 2. Separar train y test
# =========================
datatrainfull <- data_reducida_plus[1:7000, ]  # contiene Exited
datatest_reducido <- data_reducida_plus[7001:10000, ]  # no tiene Exited

datatrainfull$group <- NULL
datatest_reducido$group <- NULL

# =========================
# 3. Dummificar variables categóricas
# =========================
rec <- recipe(Exited ~ ., data = datatrainfull) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

rec_prep <- prep(rec)

train_processed <- bake(rec_prep, datatrainfull)

# Para test, no debe tener Exited
test_processed <- bake(rec_prep, new_data = datatest_reducido)

# Guardar IDs y eliminar de test
IDs <- test_processed$ID
test_processed$ID <- NULL

# =========================
# 4. Asegurar que test tenga las mismas columnas que train
# =========================
missing_cols <- setdiff(names(train_processed), names(test_processed))
# excluir la columna de respuesta
missing_cols <- setdiff(missing_cols, "Exited")
for(col in missing_cols){
  test_processed[[col]] <- 0
}

# Reordenar columnas igual que train (excepto Exited)
train_features <- train_processed %>% select(-Exited)
test_processed <- test_processed[, names(train_features)]

# =========================
# 5. Entrenar SVM radial
# =========================
svm_best <- svm(
  Exited ~ .,
  data = train_processed,
  kernel = "radial",
  cost = 5,
  gamma = 0.05
)

# =========================
# 6. Predecir clases en test
# =========================
pred_class <- predict(svm_best, test_processed)

# =========================
# 7. Convertir a Yes/No
# =========================
pred_class <- ifelse(pred_class == "1", "Yes", "No")

# =========================
# 8. Crear CSV final
# =========================
resultado_final <- data.frame(
  ID = IDs,
  Exited = pred_class
)

write.csv(
  resultado_final,
  "~/Documents/GitHub/Mineria/Entrega_2/SVM_radial_submit_final.csv",
  row.names = FALSE
)
