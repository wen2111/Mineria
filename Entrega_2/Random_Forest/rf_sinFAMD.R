#####################################################
############## RANDOM FOREST SIN FAMD (Melissa) #####
#####################################################

library(randomForest)
library(caret)
library(pROC)

load("data_reducida_con_ID.RData")

mydata <- data_reducida

################## PREPARACION ######################
train <- subset(mydata, group == "train") 
test  <- subset(mydata, group == "test") 

variables_eliminar <- c("group", "Surname", "ID")
train <- train[, !names(train) %in% variables_eliminar]
test_submission_id <- test$ID 
test <- test[, !names(test) %in% c("group", "Surname", "ID")]

# Definimos factor: 1=Yes, 0=No
train$Exited <- factor(train$Exited, levels = c("1","0"), labels = c("Yes","No"))

################## PARTICION ######################
set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # Usamos train2 ORIGINAL
test2  <- train[-index, ] # Usamos test2 ORIGINAL

################## CONFIGURACIÓN DEL MODELO ####################
# 1. Control con Cross-Validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# 2. Grid de mtry (Calculado sobre variables originales)
# Restamos 1 porque una columna es 'Exited'
mtry.class <- sqrt(ncol(train2) - 1) 
tuneGrid <- data.frame(mtry = floor(c(mtry.class, mtry.class*2, mtry.class/2)))
# Asegurar que mtry no sea menor a 1 ni mayor que n_variables
tuneGrid$mtry[tuneGrid$mtry < 1] <- 1
tuneGrid <- unique(tuneGrid)

################## RANDOM FOREST (VARIABLES ORIGINALES) #######
set.seed(123)
rf_original <- train(
  Exited ~ ., 
  data = train2,
  method = "rf",
  metric = "ROC",        
  trControl = ctrl,
  tuneGrid = tuneGrid,
  # --- PARÁMETROS ANTI-OVERFITTING ---
  ntree = 400,           
  nodesize = 40          # Mantenemos esto para evitar memorización
)

print(rf_original)

################## CURVA ROC Y UMBRAL ##########################
probs_test  <- predict(rf_original, newdata = test2, type = "prob")
probs_train <- predict(rf_original, newdata = train2, type = "prob")

roc_obj <- roc(test2$Exited, probs_test$Yes, percent = TRUE)

# Encontrar mejor umbral
coords_optimas <- coords(roc_obj, "best", 
                         ret = c("threshold", "sensitivity", "specificity"), 
                         best.method = "closest.topleft")
nuevo_umbral <- coords_optimas$threshold

# Plotear
plot.roc(roc_obj, print.auc = TRUE, print.thres = "best", main = "ROC Curve (RF Sin FAMD)")

################## PREDICCIONES Y KPIS FINALES #################
# Aplicamos el umbral optimizado
pred_train_opt <- factor(ifelse(probs_train$Yes > nuevo_umbral, "Yes", "No"), levels = c("Yes", "No"))
pred_test_opt  <- factor(ifelse(probs_test$Yes > nuevo_umbral, "Yes", "No"), levels = c("Yes", "No"))

# Matrices
conf_train <- caret::confusionMatrix(pred_train_opt, train2$Exited, positive = "Yes", mode = "prec_recall")
conf_test  <- caret::confusionMatrix(pred_test_opt, test2$Exited, positive = "Yes", mode = "prec_recall")

# Dataframe Resumen
resultados_rf <- data.frame(
  Dataset = c("Train", "Test"),
  Accuracy = c(conf_train$overall["Accuracy"], conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Precision"], conf_test$byClass["Precision"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"], conf_test$byClass["Sensitivity"]),
  F1_Score = c(conf_train$byClass["F1"], conf_test$byClass["F1"])
)

print(resultados_rf)

################## PREDICCIÓN FINAL SUBMISSION (KAGGLE) ##########
# 1. Predecir directamente sobre el test set limpio (variables originales)
probs_kaggle <- predict(rf_original, newdata = test, type = "prob")

# 2. Aplicar el umbral optimizado
pred_submission_class <- ifelse(probs_kaggle$Yes > nuevo_umbral, "Yes", "No")

################## GENERACIÓN ARCHIVO CSV #########################
submit <- data.frame(
  ID = test_submission_id,
  Exited = pred_submission_class
)

# Guardar
write.csv(submit, "submission_rf_original_optimo.csv", row.names = FALSE)

head(submit)
