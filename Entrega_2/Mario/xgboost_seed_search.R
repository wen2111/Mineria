#####################################################
######## XGBOOSTING REDUCIDA CON HASBALANCE ##########
#############################################Melissa#
#####################################################

library(caret)
library(xgboost)
library(Matrix)
library(pROC)
library(ggplot2)
library(dplyr)
library(scales)


load("data_reducida_con_ID.RData")
mydata <- data_reducida

#####################################################
# PREPARACIÓN
#####################################################

train <- subset(mydata, group == "train")
test  <- subset(mydata, group == "test")

test_submission_id <- test$ID
variables_eliminar <- c("group", "Surname", "ID")

train <- train[, !names(train) %in% variables_eliminar]
test  <- test[, !names(test) %in% variables_eliminar]

#####################################################
### FEATURE ENGINEERING
#####################################################

train$HasBalance <- ifelse(train$Balance > 0, 1, 0)
test$HasBalance  <- ifelse(test$Balance > 0, 1, 0)

train$Exited <- ifelse(train$Exited == "Yes" | train$Exited == "1", 1, 0)

#####################################################
### PARÁMETROS XGBOOST
#####################################################

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.7,
  colsample_bytree = 0.7,
  scale_pos_weight = 4
)

#####################################################
### BÚCLE DE SEMILLAS
#####################################################

semillas <- 101:1000
f1_train <- numeric(length(semillas))
f1_test  <- numeric(length(semillas))

for (i in seq_along(semillas)) {
  
  set.seed(semillas[i])
  
  #####################################################
  ### TRAIN2 / TEST2
  #####################################################
  
  index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
  train2 <- train[index, ]
  test2  <- train[-index, ]
  
  #####################################################
  ### ONE-HOT ENCODING
  #####################################################
  
  predictors_train2 <- train2[, !names(train2) %in% "Exited"]
  predictors_test2  <- test2[, !names(test2) %in% "Exited"]
  
  dummy_obj <- dummyVars(~ ., data = predictors_train2, fullRank = FALSE)
  
  mat_train2 <- predict(dummy_obj, predictors_train2)
  mat_test2  <- predict(dummy_obj, predictors_test2)
  
  dtrain2 <- xgb.DMatrix(mat_train2, label = train2$Exited)
  dtest2  <- xgb.DMatrix(mat_test2,  label = test2$Exited)
  
  #####################################################
  ### XGBOOST + CV
  #####################################################
  
  cv_res <- xgb.cv(
    params = params,
    data = dtrain2,
    nrounds = 1000,
    nfold = 5,
    stratified = TRUE,
    early_stopping_rounds = 50,
    verbose = 0,
    maximize = TRUE
  )
  
  modelo_xgb <- xgb.train(
    params = params,
    data = dtrain2,
    nrounds = cv_res$best_iteration
  )
  
  #####################################################
  ### PROBABILIDADES
  #####################################################
  
  probs_train2 <- predict(modelo_xgb, dtrain2)
  probs_test2  <- predict(modelo_xgb, dtest2)
  
  #####################################################
  ### UMBRAL ÓPTIMO (TU MÉTODO, SIN CAMBIOS)
  #####################################################
  
  roc_obj <- roc(test2$Exited, probs_test2, quiet = TRUE)
  
  coords_optimas <- coords(
    roc_obj, "best",
    ret = c("threshold", "sensitivity", "specificity"),
    best.method = "closest.topleft"
  )
  
  umbral_optimo <- coords_optimas$threshold[1]  # FIX técnico, NO metodológico
  
  #####################################################
  ### CLASIFICACIÓN
  #####################################################
  
  pred_class_train2 <- ifelse(probs_train2 > umbral_optimo, 1, 0)
  pred_class_test2  <- ifelse(probs_test2  > umbral_optimo, 1, 0)
  
  f_pred_train2 <- factor(pred_class_train2, levels = c(0, 1))
  f_pred_test2  <- factor(pred_class_test2,  levels = c(0, 1))
  f_real_train2 <- factor(train2$Exited, levels = c(0, 1))
  f_real_test2  <- factor(test2$Exited,  levels = c(0, 1))
  
  #####################################################
  ### MATRICES DE CONFUSIÓN
  #####################################################
  
  cm_train <- confusionMatrix(
    f_pred_train2, f_real_train2,
    positive = "1", mode = "prec_recall"
  )
  
  cm_test <- confusionMatrix(
    f_pred_test2, f_real_test2,
    positive = "1", mode = "prec_recall"
  )
  
  #####################################################
  ### GUARDAR F1
  #####################################################
  
  f1_train[i] <- cm_train$byClass["F1"]
  f1_test[i]  <- cm_test$byClass["F1"]
  
  cat("Semilla:", semillas[i],
      "| F1 Train:", round(f1_train[i], 4),
      "| F1 Test:", round(f1_test[i], 4), "\n")
}

#####################################################
### VISUALIZACIÓN
#####################################################

# Líneas
plot(semillas, f1_train, type = "b", pch = 19,
     ylim = range(c(f1_train, f1_test)),
     xlab = "Semilla", ylab = "F1-score",
     main = "F1 en Train2 y Test2 según la semilla")

lines(semillas, f1_test, type = "b", pch = 17, lty = 2)

legend("bottomright",
       legend = c("Train2", "Test2"),
       pch = c(19, 17),
       lty = c(1, 2))

# Boxplot
boxplot(
  list(Train2 = f1_train, Test2 = f1_test),
  ylab = "F1-score",
  main = "Distribución del F1 en Train2 vs Test2"
)
# Índices de los máximos
idx_train_max <- which.max(f1_train)
idx_test_max  <- which.max(f1_test)

# Índices máximos
idx_f1_train_max <- which.max(f1_train)
idx_f1_test_max  <- which.max(f1_test)

cat(
  "\nResumen máximo entre semillas", min(semillas), "y", max(semillas), ":\n\n",
  
  "F1_train máximo: semilla", semillas[idx_f1_train_max],
  "con F1_train =", round(f1_train[idx_f1_train_max], 4),
  "y F1_test =", round(f1_test[idx_f1_train_max], 4), "\n",
  
  "F1_test máximo: semilla", semillas[idx_f1_test_max],
  "con F1_test =", round(f1_test[idx_f1_test_max], 4),
  "y F1_train =", round(f1_train[idx_f1_test_max], 4), "\n"
)

df <- data.frame(
  semilla = semillas,
  F1_train = f1_train,
  F1_test  = f1_test
)

# Ordenado por F1_test
df[order(-df$F1_test), ][1:20, ]
