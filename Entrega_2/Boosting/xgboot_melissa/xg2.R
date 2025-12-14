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

load("data_reducida_con_ID.RData")
mydata <- data_reducida

mydata <- data_transformada
vars <- c(
  "Age",
  "EstimatedSalary",
  "AvgTransactionAmount",
  "CreditScore",
  "DigitalEngagementScore",
  "Balance",
  "NumOfProducts_grupo",
  "TransactionFrequency",
  "Tenure",
  "NetPromoterScore",
  "Geography",
  "Gender",
  "IsActiveMember",
  "Exited"
)
mydata<-mydata[,vars]
mydata$group<-data_reducida$group

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
set.seed(777)
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
##############################################################
bounds <- list(
  eta = c(0.01, 0.1),         
  max_depth = c(3L, 6L),        
  min_child_weight = c(1L, 5L), 
  subsample = c(0.6, 0.9),      
  colsample_bytree = c(0.6, 0.9)
)

#install.packages("ParBayesianOptimization")
library(ParBayesianOptimization)

cv_xgb <- function(eta, max_depth, min_child_weight, subsample, colsample_bytree) {
  
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = eta,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    scale_pos_weight = 4
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain2,
    nrounds = 400,
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 50,
    maximize = TRUE,
    verbose = 0
  )
  
  list(Score = cv$evaluation_log[cv$best_iteration]$test_auc_mean)
}

opt <- bayesOpt(
  FUN = cv_xgb,
  bounds = bounds,
  initPoints = 10,
  iters.n = 30
)

# Mejores parámetros encontrados
best_params <- getBestPars(opt)
print(best_params)


final_cv <- xgb.cv(
  params = c(best_params,
             list(objective = "binary:logistic",
                  eval_metric = "auc",
                  scale_pos_weight = 4)),
  data = dtrain2,
  nrounds = 1000,
  nfold = 5,
  stratified = TRUE,
  early_stopping_rounds = 50,
  maximize = TRUE,
  verbose = 0
)

best_nrounds <- final_cv$best_iteration

final_model <- xgb.train(
  params = c(best_params,
             list(objective = "binary:logistic",
                  eval_metric = "auc",
                  scale_pos_weight = 4)),
  data = dtrain2,
  nrounds = best_nrounds
)

##############################################################################

#####################################################
### 6. EVALUACIÓN Y UMBRAL ÓPTIMO (TEST2)
#####################################################
# Predecimos probabilidades sobre la validación interna
probs_test2 <- predict(final_model, dtest2)

# Curva ROC para buscar el mejor punto de corte
roc_obj <- roc(test2$Exited, probs_test2)

# Buscamos el umbral que maximiza la suma de Sensibilidad y Especificidad
coords_optimas <- coords(roc_obj, "best", 
                         ret = c("threshold", "sensitivity", "specificity"), 
                         best.method = "closest.topleft")
umbral_optimo <- coords_optimas$threshold

cat("El umbral optimizado es:", umbral_optimo, "\n")

################## GRAFICO ROC ########################
plot(roc_obj, print.auc = TRUE, print.thres = "best", col="blue", main="ROC Curve (Validation)")

#####################################################
### 7. CÁLCULO DE KPIS (TRAIN2 VS TEST2)
#####################################################
# Obtenemos predicciones también para train2 para ver si hay overfitting
probs_train2 <- predict(final_model, dtrain2)
# Convertimos probabilidad a clase usando el umbral optimizado
pred_class_train2 <- ifelse(probs_train2 > umbral_optimo, 1, 0)
pred_class_test2  <- ifelse(probs_test2 > umbral_optimo, 1, 0)

# Convertimos a Factor para caret (asegurando niveles 0 y 1)
f_pred_train2 <- factor(pred_class_train2, levels = c(0, 1))
f_pred_test2  <- factor(pred_class_test2, levels = c(0, 1))
f_real_train2 <- factor(train2$Exited, levels = c(0, 1))
f_real_test2  <- factor(test2$Exited, levels = c(0, 1))

# Matrices de Confusión
cm_train <- confusionMatrix(f_pred_train2, f_real_train2, positive = "1", mode = "prec_recall")
cm_test  <- confusionMatrix(f_pred_test2, f_real_test2, positive = "1", mode = "prec_recall")

# Tabla Resumen
resultados <- data.frame(
  Dataset = c("Train2", "Test2"),
  Accuracy = c(cm_train$overall["Accuracy"], cm_test$overall["Accuracy"]),
  Precision = c(cm_train$byClass["Precision"], cm_test$byClass["Precision"]),
  Recall = c(cm_train$byClass["Sensitivity"], cm_test$byClass["Sensitivity"]),
  F1_Score = c(cm_train$byClass["F1"], cm_test$byClass["F1"])
)

print(resultados)
