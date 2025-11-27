###############################################
##### GLM BOOTSTRAP REDUCIDA SIN BALANCEAR ####
###############################################

library(caret)

load("data_reducida_con_ID.RData")

# SEPARAR TRAIN Y TEST
train <- subset(mydata, group == "train") # 7000 obs
test <- subset(mydata, group == "test")   # 3000 obs

# ELIMINAR VARIABLES NO NECESARIAS
variables_eliminar <- c("group", "Surname", "ID", "Age", "Balance")
train <- train[, !names(train) %in% variables_eliminar]
test <- test[, !names(test) %in% c("group", "Surname", "Age", "Balance")] # Mantener ID en test si es necesario para el submit

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("1","0"),
                       labels = c("Yes","No"))

# PARTICION TRAIN2/TEST2

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

# BOOTSTRAP

ctrl_boot_auc <- trainControl(method = "boot", 
                              number = 200,         # 200 samples
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

fit_boot_auc <- train(Exited ~ ., data=train2, 
                      method = "glm", family = "binomial",
                      trControl = ctrl_boot_auc, metric = "ROC")

auc_boot <- fit_boot_auc$results$ROC
cat('Area under curve (Bootstrap):', round(as.numeric(auc_boot),3), '\n')

# Predicciones probabilísticas
train_pred_prob <- predict(fit_boot_auc, newdata = train2, type = "prob")
test_pred_prob  <- predict(fit_boot_auc, newdata = test2,  type = "prob")

# Threshold óptimo
library(PRROC)

probs <- test_pred_prob$Yes
labels <- ifelse(test2$Exited == "Yes", 1, 0)

thresholds <- seq(0,1,0.01)
f1_values <- sapply(thresholds, function(th){
  pred <- ifelse(probs > th, 1, 0)
  TP <- sum(pred==1 & labels==1)
  FP <- sum(pred==1 & labels==0)
  FN <- sum(pred==0 & labels==1)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  2*((precision*recall)/(precision+recall))
})

best_th <- thresholds[which.max(f1_values)]
best_th


# Convertimos a clases con threshold=0.23
# si prob > threshold → Yes
train_pred_cut <- ifelse(train_pred_prob$Yes > best_th, "Yes", "No")
test_pred_cut  <- ifelse(test_pred_prob$Yes > best_th, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("Yes","No"))
test_pred_cut  <- factor(test_pred_cut,  levels = c("Yes","No"))

# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train2$Exited)
conf_test  <- confusionMatrix(test_pred_cut, test2$Exited)

# F1-score function
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)


kpis <- data.frame(
  Dataset = c("Train2", "Test2"),
  Error_rate = c(1 - conf_train$overall["Accuracy"],
                 1 - conf_test$overall["Accuracy"]),
  Accuracy = c(conf_train$overall["Accuracy"],
               conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Pos Pred Value"],
                conf_test$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"],
                         conf_test$byClass["Sensitivity"]),
  Specificity = c(conf_train$byClass["Specificity"],
                  conf_test$byClass["Specificity"]),
  F1_Score = c(f1_train, f1_test)
)

kpis

# KAGGLE
set.seed(123)
fit_final_glm <- train(
  Exited ~ ., data = train,
  method = "glm", family = "binomial",
  trControl = ctrl_boot_auc,
  metric = "ROC"
)

pred_kaggle_prob <- predict(fit_final_glm, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > best_th, "Yes", "No")


submission <- data.frame(ID = test$ID, Exited = pred_kaggle_class)
write.csv(submission, "submission_glm_bootstrap_threshold.csv", row.names = FALSE)


########################## PROBAARRRRRRRR ##############


# Probit link
fit_probit <- train(
  Exited ~ .,
  data = train2,
  method = "glm",
  family = binomial(link = "probit"),
  trControl = ctrl_boot_auc,
  metric = "ROC"
)

# Cauchit link (para colas más pesadas)
fit_cauchit <- train(
  Exited ~ .,
  data = train2,
  method = "glm", 
  family = binomial(link = "cauchit"),
  trControl = ctrl_boot_auc,
  metric = "ROC"
)

###

# Usar ROC como métrica que maneja bien clases desbalanceadas
ctrl_balanced <- trainControl(
  method = "boot",
  number = 200,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "up" # Upsampling de la clase minoritaria
)

fit_balanced <- train(
  Exited ~ .,
  data = train2,
  method = "glm",
  family = "binomial",
  trControl = ctrl_balanced,
  metric = "ROC"
)

###

# Random Forest
library(randomForest)
fit_rf <- train(
  Exited ~ .,
  data = train2,
  method = "rf",
  trControl = ctrl_boot_auc,
  metric = "ROC"
)

# XGBoost
library(xgboost)
fit_xgb <- train(
  Exited ~ .,
  data = train2,
  method = "xgbTree",
  trControl = ctrl_boot_auc,
  metric = "ROC"
)


###

evaluate_model <- function(model, test_data, model_name) {
  probs <- predict(model, newdata = test_data, type = "prob")$Yes
  preds <- ifelse(probs > best_th, "Yes", "No")
  preds <- factor(preds, levels = c("Yes", "No"))
  
  cm <- confusionMatrix(preds, test_data$Exited)
  
  f1 <- 2 * (cm$byClass["Precision"] * cm$byClass["Sensitivity"]) / 
    (cm$byClass["Precision"] + cm$byClass["Sensitivity"])
  
  return(data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    F1_Score = as.numeric(f1),
    ROC = ifelse("ROC" %in% names(model$results), 
                 max(model$results$ROC), NA)
  ))
}

# Comparar modelos
results <- rbind(
  evaluate_model(fit_boot_auc, test2, "GLM_logit"),
  evaluate_model(fit_probit, test2, "GLM_probit"),
  evaluate_model(fit_balanced, test2, "GLM_balanced")
)


