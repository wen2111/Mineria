# --- Librerías ---
library(caret)
library(e1071)
library(pROC)
library(dplyr)
library(recipes)
library(MLmetrics)

# --- Carga de datos ---
load("~/GitHub/Mineria/DATA/dataaaaaaaaaaaaaa.RData")
bd <- data_reducida_plus

# --- Preparación de datos ---
trainbase <- bd %>% filter(group == "train") %>% select(-group)
trainbase$Exited <- factor(trainbase$Exited, levels=c("0","1"))

set.seed(123)
train_idx <- createDataPartition(trainbase$Exited, p=0.7, list=FALSE)
train <- trainbase[train_idx, ]
test  <- trainbase[-train_idx, ]

# --- Separar features y target ---
x_train <- train %>% select(-Exited)
y_train <- train$Exited
x_test  <- test %>% select(-Exited)
y_test  <- test$Exited

# --- Convertir features a numéricos (necesario para PCA) ---
x_train <- x_train %>% mutate(across(everything(), as.numeric))
x_test  <- x_test %>% mutate(across(everything(), as.numeric))

# --- Escalar y aplicar PCA ---
preproc <- preProcess(x_train, method = c("center", "scale", "pca"), pcaComp = min(ncol(x_train), 10))
x_train_pca <- predict(preproc, x_train)
x_test_pca  <- predict(preproc, x_test)

# --- Convertir y_train y y_test a numérico para AUC ---
y_train_num <- as.numeric(as.character(y_train))
y_test_num  <- as.numeric(as.character(y_test))

# --- Definir kernels a evaluar ---
kernels <- c("linear", "polynomial", "radial", "sigmoid")

# --- Crear tabla para resultados ---
results <- data.frame(Dataset=character(),
                      Kernel=character(),
                      Accuracy=double(),
                      Precision=double(),
                      Recall=double(),
                      F1=double(),
                      AUC=double(),
                      stringsAsFactors = FALSE)

# --- Entrenar SVM y calcular métricas ---
for(k in kernels){
  set.seed(123)
  svm_model <- svm(x=x_train_pca, y=y_train, kernel=k, probability=TRUE)
  
  # --- Métricas en train ---
  prob_train <- attr(predict(svm_model, x_train_pca, probability=TRUE), "probabilities")[, "1"]
  pred_train <- factor(ifelse(prob_train >= 0.2, "1", "0"), levels=c("0","1"))
  cm_train <- confusionMatrix(pred_train, y_train, positive="1")
  auc_train <- as.numeric(roc(y_train_num, prob_train)$auc)
  
  results <- rbind(results, data.frame(Dataset="Train",
                                       Kernel=k,
                                       Accuracy=cm_train$overall["Accuracy"],
                                       Precision=cm_train$byClass["Precision"],
                                       Recall=cm_train$byClass["Recall"],
                                       F1=cm_train$byClass["F1"],
                                       AUC=auc_train))
  
  # --- Métricas en test ---
  prob_test <- attr(predict(svm_model, x_test_pca, probability=TRUE), "probabilities")[, "1"]
  pred_test <- factor(ifelse(prob_test >= 0.2, "1", "0"), levels=c("0","1"))
  cm_test <- confusionMatrix(pred_test, y_test, positive="1")
  auc_test <- as.numeric(roc(y_test_num, prob_test)$auc)
  
  results <- rbind(results, data.frame(Dataset="Test",
                                       Kernel=k,
                                       Accuracy=cm_test$overall["Accuracy"],
                                       Precision=cm_test$byClass["Precision"],
                                       Recall=cm_test$byClass["Recall"],
                                       F1=cm_test$byClass["F1"],
                                       AUC=auc_test))
}

# --- Mostrar tabla ---
print(results)
