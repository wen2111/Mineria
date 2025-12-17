#####################################################
######## XGBOOTING REDUCIDA CON HASBALANCE ##########
#############################################Melissa#

library(caret)
library(xgboost)
library(Matrix)
library(pROC)
library(ggplot2)
library(dplyr)
library(scales)

load("~/GitHub/Mineria/Entrega_2/Boosting/xgboot_melissa/data_reducida_con_ID.RData")
mydata <- data_reducida
set.seed(689)
#####################################################
# PREPARACIÓN
#####################################################
# Separar Train y Test (para Kaggle, Exited vacía)
train <- subset(mydata, group == "train")
test  <- subset(mydata, group == "test")

# Guardar ID para submission y eliminar variables innecesarias para el modelo
test_submission_id <- test$ID
variables_eliminar <- c("group", "Surname", "ID")

train <- train[, !names(train) %in% variables_eliminar]
test  <- test[, !names(test) %in% c("group", "Surname", "ID")]

#####################################################
### FEATURE ENGINEERING
#####################################################
# Creamos HasBalance porque los que tienen saldo se van mas
# XGBoost prefiere números: 1 si tiene saldo, 0 si no.
train$HasBalance <- ifelse(train$Balance > 0, 1, 0)
test$HasBalance  <- ifelse(test$Balance > 0, 1, 0)

# Convertir la variable objetivo a numérica 0/1 para XGBoost
# "Yes" o "1" es la clase positiva.
train$Exited <- ifelse(train$Exited == "Yes" | train$Exited == "1", 1, 0)

#####################################################
### 3. PARTICIÓN INTERNA (TRAIN2 / TEST2)
#####################################################

index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ]
test2  <- train[-index, ]

#####################################################
### 4. TRANSFORMACIÓN NUMÉRICA (ONE-HOT ENCODING)
#####################################################
# XGBoost no acepta factores. Convertimos todo a matriz numérica.
# Importante: Creamos el "molde" solo con los datos de entrenamiento (train2).

# Separamos predictores de la variable objetivo
predictors_train2 <- train2[, !names(train2) %in% "Exited"]
predictors_test2  <- test2[, !names(test2) %in% "Exited"]
predictors_kaggle <- test[, !names(test) %in% "Exited"] # El test de Kaggle no tiene Exited

# Creamos el esquema de transformación
dummy_obj <- dummyVars(~ ., data = predictors_train2, fullRank = FALSE)

# Aplicamos el esquema a los 3 conjuntos de datos
mat_train2  <- predict(dummy_obj, newdata = predictors_train2)
mat_test2   <- predict(dummy_obj, newdata = predictors_test2)
mat_kaggle  <- predict(dummy_obj, newdata = predictors_kaggle)

# Convertimos a formato nativo de XGBoost (DMatrix)
# test2 y train2 llevan etiqueta (label), kaggle no.
dtrain2 <- xgb.DMatrix(data = mat_train2, label = train2$Exited)
dtest2  <- xgb.DMatrix(data = mat_test2, label = test2$Exited)
dkaggle <- xgb.DMatrix(data = mat_kaggle)

#####################################################
### 5. ENTRENAMIENTO XGBOOST CON CV
#####################################################
# Configuración para subir el F1 (Atención a scale_pos_weight)
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,              # Aprendizaje lento para evitar overfitting
  max_depth = 4,           # Árboles no muy profundos
  subsample = 0.7,
  colsample_bytree = 0.7,
  scale_pos_weight = 4     # CLAVE: Compensar desbalanceo (aprox 80/20 ratio)
)


# Cross Validation para encontrar el número óptimo de rondas
cv_res <- xgb.cv(
  params = params,
  data = dtrain2,
  nrounds = 1000,
  nfold = 5,
  stratified = TRUE,       # CLAVE: Evita el error de "dataset empty"
  early_stopping_rounds = 50,
  print_every_n = 50,
  maximize = TRUE
)

# Entrenar modelo final con la mejor iteración encontrada
modelo_xgb <- xgb.train(
  params = params,
  data = dtrain2,
  nrounds = cv_res$best_iteration
)

#####################################################
### 6. EVALUACIÓN Y UMBRAL ÓPTIMO (TEST2)
#####################################################
# Predecimos probabilidades sobre la validación interna
probs_test2 <- predict(modelo_xgb, dtest2)

# Curva ROC para buscar el mejor punto de corte
roc_obj <- roc(test2$Exited, probs_test2)

# Buscamos el umbral que maximiza la suma de Sensibilidad y Especificidad
coords_optimas <- coords(roc_obj, "best", 
                         ret = c("threshold", "sensitivity", "specificity"), 
                         best.method = "closest.topleft")
umbral_optimo <- 0.47

cat("El umbral optimizado es:", umbral_optimo, "\n")

################## GRAFICO ROC ########################
plot(roc_obj, print.auc = TRUE, print.thres = "best", col="blue", main="ROC Curve (Validation)")

#####################################################
### 7. CÁLCULO DE KPIs (TRAIN2 VS TEST2)
#####################################################

# Predicciones en TRAIN2
probs_train2 <- predict(modelo_xgb, dtrain2)
pred_class_train2 <- ifelse(probs_train2 > umbral_optimo, 1, 0)

# Predicciones en TEST2 (ya tienes probs_test2)
pred_class_test2 <- ifelse(probs_test2 > umbral_optimo, 1, 0)

# Factores (niveles bien definidos)
f_pred_train2 <- factor(pred_class_train2, levels = c(0, 1))
f_pred_test2  <- factor(pred_class_test2,  levels = c(0, 1))
f_real_train2 <- factor(train2$Exited, levels = c(0, 1))
f_real_test2  <- factor(test2$Exited,  levels = c(0, 1))

# Matrices de confusión
cm_train <- confusionMatrix(
  data = f_pred_train2,
  reference = f_real_train2,
  positive = "1",
  mode = "prec_recall"
)

cm_test <- confusionMatrix(
  data = f_pred_test2,
  reference = f_real_test2,
  positive = "1",
  mode = "prec_recall"
)

# Tabla resumen con TODAS las métricas
resultados <- data.frame(
  Dataset = c("Train2", "Test2"),
  
  Error_rate = c(
    1 - cm_train$overall["Accuracy"],
    1 - cm_test$overall["Accuracy"]
  ),
  
  Accuracy = c(
    cm_train$overall["Accuracy"],
    cm_test$overall["Accuracy"]
  ),
  
  Precision = c(
    cm_train$byClass["Precision"],
    cm_test$byClass["Precision"]
  ),
  
  Recall_Sensitivity = c(
    cm_train$byClass["Sensitivity"],
    cm_test$byClass["Sensitivity"]
  ),
  
  Specificity = c(
    cm_train$byClass["Specificity"],
    cm_test$byClass["Specificity"]
  ),
  
  F1_Score = c(
    cm_train$byClass["F1"],
    cm_test$byClass["F1"]
  )
)

# Mostrar resultados
resultados

#####################################################
### 8. GENERACIÓN SUBMIT KAGGLE (CORREGIDO A YES/NO)
#####################################################

# 1. Calculamos las probabilidades con el modelo XGBoost
probs_kaggle <- predict(modelo_xgb, dkaggle)

# 2. Aplicamos el UMBRAL ÓPTIMO para obtener 0 y 1
pred_class_num <- ifelse(probs_kaggle > umbral_optimo, 1, 0)

# 3. CONVERSIÓN A TEXTO (Yes/No)
# Si es 1 -> "Yes", si es 0 -> "No"
pred_kaggle_text <- ifelse(pred_class_num == 1, "Yes", "No")

# 4. Crear el Dataframe final
submit <- data.frame(
  ID = test_submission_id,
  Exited = pred_kaggle_text
)

# 5. Guardar el archivo
write.csv(submit, "submission_xgb_yesno4.csv", row.names = FALSE)

# 6. Verificación visual
print("Primeras filas del archivo a enviar:")
head(submit)