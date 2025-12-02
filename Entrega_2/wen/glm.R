library(caret)

mejores_enlaces <- c("logit", "cloglog", "probit")
resultados <- data.frame()

for(enlace in mejores_enlaces) {
  cat("\n=== ENLACE:", enlace, "===\n")
  
  # 1. Entrenar modelo
  model <- glm(Exited ~ ., data = train2, family = binomial(link = enlace))
  
  # 2. Predecir en train2 y test2
  probs_train <- predict(model, train2, type = "response")
  probs_test <- predict(model, test2, type = "response")
  
  # 3. Encontrar mejor threshold en test2
  thresholds <- seq(0.1, 0.9, 0.05)
  best_f1_test <- 0
  best_threshold <- 0.5
  
  for(th in thresholds) {
    pred_test <- ifelse(probs_test > th, "Yes", "No")
    pred_test <- factor(pred_test, levels = c("No", "Yes"))
    cm_test <- confusionMatrix(pred_test, test2$Exited, positive = "Yes")
    f1_test <- 2 * cm_test$byClass["Precision"] * cm_test$byClass["Sensitivity"] / 
      (cm_test$byClass["Precision"] + cm_test$byClass["Sensitivity"])
    
    if(f1_test > best_f1_test) {
      best_f1_test <- f1_test
      best_threshold <- th
    }
  }
  
  # 4. Evaluar con el mejor threshold en AMBOS conjuntos
  # Test2
  pred_test_final <- ifelse(probs_test > best_threshold, "Yes", "No")
  pred_test_final <- factor(pred_test_final, levels = c("No", "Yes"))
  cm_test_final <- confusionMatrix(pred_test_final, test2$Exited, positive = "Yes")
  
  # Train2
  pred_train_final <- ifelse(probs_train > best_threshold, "Yes", "No")
  pred_train_final <- factor(pred_train_final, levels = c("No", "Yes"))
  cm_train_final <- confusionMatrix(pred_train_final, train2$Exited, positive = "Yes")
  
  # 5. Calcular métricas
  f1_train <- 2 * cm_train_final$byClass["Precision"] * cm_train_final$byClass["Sensitivity"] / 
    (cm_train_final$byClass["Precision"] + cm_train_final$byClass["Sensitivity"])
  
  f1_test <- best_f1_test
  
  # 6. Calcular diferencia (overfitting)
  diff_f1 <- f1_train - f1_test
  diff_accuracy <- cm_train_final$overall["Accuracy"] - cm_test_final$overall["Accuracy"]
  
  # 7. Mostrar resultados
  cat("Threshold óptimo:", round(best_threshold, 3), "\n")
  cat("Train2 - F1:", round(f1_train, 4), "| Accuracy:", round(cm_train_final$overall["Accuracy"], 4), "\n")
  cat("Test2  - F1:", round(f1_test, 4), "| Accuracy:", round(cm_test_final$overall["Accuracy"], 4), "\n")
  cat("Diferencia (Train-Test):\n")
  cat("  F1:", round(diff_f1, 4), "| Accuracy:", round(diff_accuracy, 4), "\n")
  
  # 8. Evaluar overfitting
  if(diff_f1 > 0.1) {
    cat("⚠️  FUERTE OVERFITTING (F1 diferencia > 0.1)\n")
  } else if(diff_f1 > 0.05) {
    cat("⚠️  MODERADO OVERFITTING (F1 diferencia > 0.05)\n")
  } else {
    cat("✅  BUENA GENERALIZACIÓN\n")
  }
  
  # 9. Guardar resultados
  resultados <- rbind(resultados, data.frame(
    Enlace = enlace,
    Threshold = best_threshold,
    F1_Train = round(f1_train, 4),
    F1_Test = round(f1_test, 4),
    Diff_F1 = round(diff_f1, 4),
    Accuracy_Train = round(cm_train_final$overall["Accuracy"], 4),
    Accuracy_Test = round(cm_test_final$overall["Accuracy"], 4),
    Overfitting = ifelse(diff_f1 > 0.1, "Fuerte", 
                         ifelse(diff_f1 > 0.05, "Moderado", "Bajo"))
  ))
}

# Mostrar tabla comparativa
cat("\n=== RESUMEN COMPARATIVO ===\n")
print(resultados)