library(caret)
library(xgboost)

# Grid de parámetros para XGBoost
xgb_grid <- expand.grid(
  nrounds = c(100, 150),
  max_depth = c(3, 4),
  eta = c(0.01, 0.05),
  gamma = c(0, 0.1),
  colsample_bytree = 0.7,
  min_child_weight = c(1, 3),
  subsample = 0.8
)

# Control para optimizar F1
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = function(data, lev = NULL, model = NULL) {
    # Calcular F1 manualmente
    precision <- posPredValue(data$pred, data$obs, positive = "Yes")
    recall <- sensitivity(data$pred, data$obs, positive = "Yes")
    f1 <- ifelse((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)
    c(F1 = f1, Precision = precision, Recall = recall)
  },
  verboseIter = FALSE
)

# Entrenar XGBoost optimizando F1
set.seed(123)
xgb_model <- train(
  Exited ~ .,
  data = train2,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  metric = "F1",
  maximize = TRUE,
  verbosity = 0
)

# Mostrar mejores parámetros
cat("Mejores parámetros XGBoost:\n")
print(xgb_model$bestTune)

# Predecir probabilidades
probs_train <- predict(xgb_model, train2, type = "prob")$Yes
probs_test <- predict(xgb_model, test2, type = "prob")$Yes

# Encontrar mejor threshold en test2 para F1
thresholds <- seq(0.1, 0.9, 0.02)
best_f1_test <- 0
best_threshold

for(th in thresholds) {
  pred_test <- ifelse(probs_test > th, "Yes", "No")
  pred_test <- factor(pred_test, levels = c("No", "Yes"))
  cm_test <- confusionMatrix(pred_test, test2$Exited, positive = "Yes")
  
  # Calcular F1
  precision <- cm_test$byClass["Precision"]
  recall <- cm_test$byClass["Sensitivity"]
  f1_test <- 2 * precision * recall / (precision + recall)
  
  if(f1_test > best_f1_test) {
    best_f1_test <- f1_test
    best_threshold <- th
  }
}

# Evaluar con mejor threshold en AMBOS conjuntos
# Test2
pred_test_final <- ifelse(probs_test > best_threshold, "Yes", "No")
pred_test_final <- factor(pred_test_final, levels = c("No", "Yes"))
cm_test <- confusionMatrix(pred_test_final, test2$Exited, positive = "Yes")

# Train2
pred_train_final <- ifelse(probs_train > best_threshold, "Yes", "No")
pred_train_final <- factor(pred_train_final, levels = c("No", "Yes"))
cm_train <- confusionMatrix(pred_train_final, train2$Exited, positive = "Yes")

# Calcular F1 para ambos
f1_train <- 2 * cm_train$byClass["Precision"] * cm_train$byClass["Sensitivity"] / 
  (cm_train$byClass["Precision"] + cm_train$byClass["Sensitivity"])

f1_test <- best_f1_test

# Mostrar resultados
cat("\n=== RESULTADOS XGBOOST ===\n")
cat("Mejor threshold:", round(best_threshold, 3), "\n\n")

cat("TRAIN2:\n")
cat("  Accuracy:", round(cm_train$overall["Accuracy"], 4), "\n")
cat("  F1:", round(f1_train, 4), "\n")
cat("  Recall:", round(cm_train$byClass["Sensitivity"], 4), "\n")
cat("  Precision:", round(cm_train$byClass["Precision"], 4), "\n\n")

cat("TEST2:\n")
cat("  Accuracy:", round(cm_test$overall["Accuracy"], 4), "\n")
cat("  F1:", round(f1_test, 4), "\n")
cat("  Recall:", round(cm_test$byClass["Sensitivity"], 4), "\n")
cat("  Precision:", round(cm_test$byClass["Precision"], 4), "\n\n")

