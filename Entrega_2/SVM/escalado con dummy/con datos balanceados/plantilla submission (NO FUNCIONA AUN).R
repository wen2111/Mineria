best_cost <- 5
best_degree <- 5
modelo_svm_tuned <- svm(
  Exited ~ ., 
  data = trainbase, 
  cost = best_cost, 
  kernel = "polynomial", 
  degree = best_degree,
  probability = TRUE 
)
best_threshold<-0.18
svm.pred_prob_matrix <- predict(modelo_svm_tuned, testbase[-9], probability = TRUE)
svm.pred_prob <- attr(svm.pred_prob_matrix, "probabilities")[, "1"]
svm.pred_class <- ifelse(svm.pred_prob >= best_threshold, "1", "0")
pclass<-as.factor(svm.pred_class)
predicciones_factor <- ifelse(pclass == "1", "Yes", "No")




id<-data$ID[7001:10000]
resultado_final <- data.frame(
  ID = id,
  Exited = predicciones_factor
)

# Guardar en Excel
write.csv(resultado_final, "~/GitHub/Mineria/Entrega_2/SVM/escalado con dummy/con datos balanceados/submit_poly_balanced.csv", row.names = FALSE)
