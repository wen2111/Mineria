########################################################
# Random Forest mejorado (sin FAMD) - script listo
########################################################

########################################################
# RANDOM FOREST – GRID SEARCH COMPLETO (sin FAMD)
########################################################

library(caret)
library(randomForest)
library(pROC)
library(e1071)

########################################################
# 1. CARGA DE DATOS
########################################################

load("~/GitHub/Mineria/DATA/dataaaaaaaaaaaaaa.RData")
mydata <- data_reducida

# IDs de submission
load("~/GitHub/Mineria/DATA/data.RData")
test_submission_id <- data$ID[7001:10000]

# Eliminar columna group si existe
mydata$group <- NULL

########################################################
# 2. DUMMYFICACIÓN CORRECTA
########################################################

dmy <- dummyVars("~ .",
                 data = mydata[, !(names(mydata) %in% c("Exited"))],
                 fullRank = TRUE)

mydata_dummy <- data.frame(predict(dmy, newdata = mydata))
mydata_dummy$Exited <- mydata$Exited

########################################################
# 3. SEPARACIÓN TRAIN / TEST HOLDOUT
########################################################

train_global <- mydata_dummy[1:7000, ]
test_holdout <- mydata_dummy[7001:10000, ]

vars_remove <- c("Surname", "ID")
train_global <- train_global[, !names(train_global) %in% vars_remove]
test_holdout <- test_holdout[, !names(test_holdout) %in% vars_remove]

train_global$Exited <- factor(train_global$Exited,
                              levels = c("1","0"),
                              labels = c("Yes","No"))

########################################################
# 4. TRAIN / VALIDATION SPLIT PARA AFINAR HIPERPARÁMETROS
########################################################

set.seed(123)
idx <- createDataPartition(train_global$Exited, p = 0.7, list = FALSE)
train2 <- train_global[idx, ]
valid2 <- train_global[-idx, ]

########################################################
# 5. CONTROL (10-FOLD CV + SMOTE)
########################################################

ctrl <- trainControl(
  method = "cv",
  number = 10,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = TRUE
)

########################################################
# 6. DEFINICIÓN DEL GRID COMPLETO
########################################################

# mtry depende del número de predictores
mtry_base <- sqrt(ncol(train2) - 1)
mtry_grid <- unique(floor(c(mtry_base/2, mtry_base, mtry_base*2)))
mtry_grid[mtry_grid < 1] <- 1

tuneGrid <- expand.grid(mtry = mtry_grid)

# hiperparámetros adicionales
ntree_values    <- c(300, 500)
nodesize_values <- c(5, 10, 20)
maxnodes_values <- c(NA, 30, 50)

########################################################
# 7. GRID SEARCH MANUAL (ntree, nodesize, maxnodes)
########################################################

results_list <- list()
counter <- 1
set.seed(123)

for(nt in ntree_values) {
  for(ns in nodesize_values) {
    for(mn in maxnodes_values) {
      
      cat(">>> PROBANDO:",
          "ntree=", nt,
          "| nodesize=", ns,
          "| maxnodes=", mn, "\n")
      
      rf_tmp <- train(
        Exited ~ .,
        data = train2,
        method = "rf",
        metric = "ROC",
        trControl = ctrl,
        tuneGrid = tuneGrid,
        ntree = nt,
        nodesize = ns,
        maxnodes = if(!is.na(mn)) mn else NULL,
        importance = TRUE
      )
      
      bestROC <- max(rf_tmp$results$ROC)
      
      results_list[[counter]] <- list(
        model = rf_tmp,
        ntree = nt,
        nodesize = ns,
        maxnodes = mn,
        ROC = bestROC
      )
      
      cat("    ROC CV:", bestROC, "\n\n")
      counter <- counter + 1
    }
  }
}

# Mejor modelo
roc_all <- sapply(results_list, function(x) x$ROC)
best_idx <- which.max(roc_all)
best_cfg <- results_list[[best_idx]]
rf_best  <- best_cfg$model

cat("\nMEJOR CONFIGURACIÓN ENCONTRADA:\n")
print(best_cfg)

########################################################
# 8. CORREGIR: OBTENER UMBRAL ÓPTIMO PARA F1 (NO ROC)
########################################################

# PRIMERO: Asegurarse de que las predicciones son correctas
cat("\n=== VERIFICANDO ESTRUCTURA DE DATOS ===\n")

# Verificar estructura de train2 y valid2
cat("Clase de train2$Exited:", class(train2$Exited), "\n")
cat("Niveles de train2$Exited:", levels(train2$Exited), "\n")

# Corregir si es necesario
train2$Exited <- factor(train2$Exited, levels = c("Yes", "No"))
valid2$Exited <- factor(valid2$Exited, levels = c("Yes", "No"))

# Obtener el modelo final entrenado
rf_model <- rf_best$finalModel

# Hacer predicciones CORRECTAMENTE
probs_valid <- predict(rf_model, newdata = valid2, type = "prob")
probs_train <- predict(rf_model, newdata = train2, type = "prob")

# Verificar estructura de las predicciones
cat("\nEstructura de probs_valid:\n")
str(probs_valid)

# Si es una matriz, convertir a dataframe y renombrar
if (is.matrix(probs_valid)) {
  probs_valid <- as.data.frame(probs_valid)
  probs_train <- as.data.frame(probs_train)
  
  # Asegurar nombres correctos
  if (all(c("1", "0") %in% colnames(probs_valid))) {
    colnames(probs_valid) <- c("No", "Yes")  # ¡Cuidado! 1=No, 0=Yes o viceversa
    colnames(probs_train) <- c("No", "Yes")
  } else if (all(c("Yes", "No") %in% colnames(probs_valid))) {
    # Ya está bien
  } else {
    # Renombrar basado en el orden
    colnames(probs_valid) <- c("No", "Yes")
    colnames(probs_train) <- c("No", "Yes")
  }
}

cat("\nNombres de columnas en probs_valid:", colnames(probs_valid), "\n")

# SEGUNDO: Buscar umbral que maximice F1 (NO ROC)
cat("\n=== BUSCANDO UMBRAL QUE MAXIMICE F1-SCORE ===\n")

# Función para calcular F1
calcular_f1 <- function(umbral, probs, reales) {
  # Asegurarse de tener probabilidad de clase positiva
  if ("Yes" %in% colnames(probs)) {
    prob_pos <- probs$Yes
  } else if ("1" %in% colnames(probs)) {
    prob_pos <- probs$"1"
  } else {
    prob_pos <- probs[, 2]  # Asumir segunda columna es positiva
  }
  
  preds <- factor(ifelse(prob_pos > umbral, "Yes", "No"), 
                  levels = c("Yes", "No"))
  
  cm <- confusionMatrix(preds, reales, positive = "Yes")
  return(cm$byClass["F1"])
}

# Probar diferentes umbrales
umbrales <- seq(0.1, 0.5, by = 0.01)
f1_scores <- numeric(length(umbrales))

for(i in 1:length(umbrales)) {
  f1_scores[i] <- calcular_f1(umbrales[i], probs_valid, valid2$Exited)
  cat(sprintf("Umbral: %.2f | F1: %.4f\n", umbrales[i], f1_scores[i]))
}

# Encontrar mejor umbral
best_idx <- which.max(f1_scores)
umbral_opt_f1 <- umbrales[best_idx]
best_f1 <- f1_scores[best_idx]

cat(sprintf("\nMEJOR UMBRAL: %.3f (F1 = %.4f)\n", umbral_opt_f1, best_f1))


########################################################
# 9. MATRICES DE CONFUSIÓN CON UMBRAL ÓPTIMO
########################################################

cat("\n=== MATRICES DE CONFUSIÓN CON UMBRAL ÓPTIMO (F1) ===\n")

# Función auxiliar para extraer probabilidad positiva
extraer_prob_positiva <- function(probs) {
  if ("Yes" %in% colnames(probs)) return(probs$Yes)
  if ("1" %in% colnames(probs)) return(probs$"1")
  return(probs[, 2])
}

# Extraer probabilidades
prob_train_yes <- extraer_prob_positiva(probs_train)
prob_valid_yes <- extraer_prob_positiva(probs_valid)

# Crear predicciones
umbral_opt_f1<-0.1
pred_train <- factor(ifelse(prob_train_yes > umbral_opt_f1, "Yes", "No"),
                     levels = c("Yes", "No"))
pred_valid <- factor(ifelse(prob_valid_yes > umbral_opt_f1, "Yes", "No"),
                     levels = c("Yes", "No"))

# Verificar longitudes
cat("Longitud pred_train:", length(pred_train), "| Longitud train2:", nrow(train2), "\n")
cat("Longitud pred_valid:", length(pred_valid), "| Longitud valid2:", nrow(valid2), "\n")

# Matrices de confusión
conf_train <- confusionMatrix(pred_train, train2$Exited, positive = "Yes", mode = "prec_recall")
conf_valid <- confusionMatrix(pred_valid, valid2$Exited, positive = "Yes", mode = "prec_recall")

# Resultados
resultados_rf <- data.frame(
  Dataset = c("Train", "Validation"),
  Accuracy = c(conf_train$overall["Accuracy"], conf_valid$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Precision"], conf_valid$byClass["Precision"]),
  Recall = c(conf_train$byClass["Sensitivity"], conf_valid$byClass["Sensitivity"]),
  F1 = c(conf_train$byClass["F1"], conf_valid$byClass["F1"]),
  Specificity = c(conf_train$byClass["Specificity"], conf_valid$byClass["Specificity"]),
  Threshold = umbral_opt_f1
)

print(resultados_rf)

# Mostrar matrices
cat("\n=== MATRIZ DE CONFUSIÓN (VALIDACIÓN) ===\n")
print(conf_valid$table)

########################################################
# 10. IMPORTANCIA DE VARIABLES
########################################################

vi <- varImp(rf_best, scale = TRUE)
plot(vi, top = 20, main = "20 variables más importantes")

########################################################
# 11. ENTRENAR MODELO FINAL CON TODOS LOS DATOS
########################################################

cat("\n=== ENTRENANDO MODELO FINAL CON TODOS LOS DATOS ===\n")

# Obtener parámetros del mejor modelo
best_params <- rf_best$bestTune
best_ntree <- best_cfg$ntree
best_nodesize <- best_cfg$nodesize
best_maxnodes <- best_cfg$maxnodes

# Entrenar modelo final con todos los datos
rf_final <- randomForest(
  Exited ~ .,
  data = train_global,
  mtry = best_params$mtry,
  ntree = best_ntree,
  nodesize = best_nodesize,
  maxnodes = if(!is.na(best_maxnodes)) best_maxnodes else NULL,
  importance = TRUE
)

cat("Modelo final entrenado con", best_ntree, "árboles\n")

########################################################
# 12. PREDICCIÓN FINAL PARA KAGGLE
########################################################

cat("\n=== GENERANDO PREDICCIONES PARA KAGGLE ===\n")

# Predecir en test_holdout
probs_kaggle <- predict(rf_final, newdata = test_holdout, type = "prob")

# Extraer probabilidad positiva
if (is.matrix(probs_kaggle)) {
  probs_kaggle <- as.data.frame(probs_kaggle)
  if (all(c("1", "0") %in% colnames(probs_kaggle))) {
    colnames(probs_kaggle) <- c("No", "Yes")
  }
}

prob_kaggle_yes <- extraer_prob_positiva(probs_kaggle)

# Aplicar umbral óptimo para F1
pred_kaggle <- ifelse(prob_kaggle_yes >= 0.1, "Yes", "No")

# Verificar distribución
cat("Distribución de predicciones:\n")
kaggle_counts <- table(pred_kaggle)
print(kaggle_counts)


# Crear submission
submission <- data.frame(ID = test_submission_id, Exited = pred_kaggle)
write.csv(submission, "~/GitHub/Mineria/submissions queue/para cuando no lleguemos a 3 en un dia/rfthresh01.csv", row.names = FALSE)
