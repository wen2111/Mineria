#####################################################
############## RANDOM FOREST CON FAMD (Melissa) #####
#####################################################

library(randomForest)
library(FactoMineR)
library(factoextra)
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

train$Exited <- factor(train$Exited, levels = c("1","0"), labels = c("Yes","No"))

################## PARTICION ######################
set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] 
test2  <- train[-index, ] 

################## FAMD (solo train2) ##########################
# Aumentamos ncp a 10 para intentar capturar más varianza si es posible
famd_train <- FAMD(train2[, !names(train2) %in% "Exited"], ncp = 10, graph = FALSE)

train2_coord <- as.data.frame(famd_train$ind$coord)
train2_coord$Exited <- train2$Exited

################## PROYECTAR TEST2 #############################
test2_proy <- predict(famd_train, newdata = test2[, !names(test2) %in% "Exited"])
test2_coord <- as.data.frame(test2_proy$coord)
test2_coord$Exited <- test2$Exited

# CORRECCIÓN DE NOMBRES (para evitar error de objeto no encontrado)
names(test2_coord) <- names(train2_coord)

################## CONFIGURACIÓN DEL MODELO ####################
# 1. Control con Cross-Validation (Más estable que el default)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary # Necesario para métrica ROC
)

# 2. Grid de mtry
mtry.class <- sqrt(ncol(train2_coord) - 1) 
tuneGrid <- data.frame(mtry = floor(c(mtry.class, mtry.class*2)))
# Asegurar que mtry no sea menor a 1
tuneGrid$mtry[tuneGrid$mtry < 1] <- 1

################## RANDOM FOREST #######
set.seed(123)
rf_famd <- train(
  Exited ~ ., 
  data = train2_coord, 
  method = "rf",
  metric = "ROC",        # Maximizamos ROC, no Accuracy
  trControl = ctrl,
  tuneGrid = tuneGrid,
  # --- PARÁMETROS ANTI-OVERFITTING ---
  ntree = 500,           # Número de árboles estable
  nodesize = 50          # <--- LA CLAVE: Mínimo 50 clientes por hoja final.
  # Esto impide que el modelo memorice casos únicos.
)

print(rf_famd)

################## CURVA ROC Y UMBRAL ##########################
probs_test  <- predict(rf_famd, newdata = test2_coord, type = "prob")
probs_train <- predict(rf_famd, newdata = train2_coord, type = "prob") # Para comparar luego

roc_obj <- roc(test2_coord$Exited, probs_test$Yes, percent = TRUE)

# Encontrar mejor umbral
coords_optimas <- coords(roc_obj, "best", 
                         ret = c("threshold", "sensitivity", "specificity"), 
                         best.method = "closest.topleft")
nuevo_umbral <- coords_optimas$threshold

# Plotear
plot.roc(roc_obj, print.auc = TRUE, print.thres = "best", main = "ROC Curve (Regularizado)")

################## PREDICCIONES Y KPIS FINALES #################
# Aplicamos el MISMO umbral a ambos
pred_train_opt <- factor(ifelse(probs_train$Yes > nuevo_umbral, "Yes", "No"), levels = c("Yes", "No"))
pred_test_opt  <- factor(ifelse(probs_test$Yes > nuevo_umbral, "Yes", "No"), levels = c("Yes", "No"))

# Matrices
conf_train <- caret::confusionMatrix(pred_train_opt, train2_coord$Exited, positive = "Yes", mode = "prec_recall")
conf_test  <- caret::confusionMatrix(pred_test_opt, test2_coord$Exited, positive = "Yes", mode = "prec_recall")

# Dataframe Resumen
resultados_famd <- data.frame(
  Dataset = c("Train", "Test"),
  Accuracy = c(conf_train$overall["Accuracy"], conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Precision"], conf_test$byClass["Precision"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"], conf_test$byClass["Sensitivity"]),
  F1_Score = c(conf_train$byClass["F1"], conf_test$byClass["F1"])
)

print(resultados_famd)

################## PROYECCIÓN TEST KAGGLE EN FAMD (CORREGIDO) #################
kaggle_proy <- predict(famd_train, newdata = test)

kaggle_coord <- as.data.frame(kaggle_proy$coord)

names(kaggle_coord) <- names(train2_coord)[!names(train2_coord) %in% "Exited"]

################## PREDICCIÓN FINAL CON UMBRAL ####################

probs_kaggle <- predict(rf_famd, newdata = kaggle_coord, type = "prob")

pred_submission_class <- ifelse(probs_kaggle$Yes > nuevo_umbral, "Yes", "No")

################## GENERACIÓN ARCHIVO CSV #########################
submit <- data.frame(
  ID = test_submission_id,
  Exited = pred_submission_class
)

# Guardar
write.csv(submit, "submission_rf_famd_optimo.csv", row.names = FALSE)

head(submit)
