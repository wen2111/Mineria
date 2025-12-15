library(randomForest)
library(FactoMineR)
library(factoextra)
library(caret)
library(dplyr)
library(pROC)

load("~/GitHub/Mineria/Entrega_2/glm/glm_melissa/data_reducida_con_ID.RData")

#mydata <- data_reducida

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
  ,"group"
)
mydata<-mydata[,vars]

################## PREPARACION ######################
train <- subset(mydata, group == "train") 
test  <- subset(mydata, group == "test") 

variables_eliminar <- c("group", "Surname", "ID")
train <- train[, !names(train) %in% variables_eliminar]
test_submission_id <- test$ID 
test <- test[, !names(test) %in% c("group", "Surname", "ID")]

train$Exited <- factor(train$Exited, levels = c("1","0"), labels = c("Yes","No"))

################## CONFIGURACIÓN DEL MODELO ####################
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

mtry.class <- sqrt(ncol(train) - 1) 
tuneGrid <- data.frame(mtry = floor(c(mtry.class, mtry.class*2)))
tuneGrid$mtry[tuneGrid$mtry < 1] <- 1

################## BUCLE SOBRE SEMILLAS ####################
results_rf <- data.frame(seed = integer(), f1_train = numeric(), f1_test = numeric())

for (s in 1:100) {
  set.seed(s)
  
  # Partición interna 70/30
  index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
  train2 <- train[index, ] 
  test2  <- train[-index, ]
  
  # FAMD sobre train2
  famd_train <- FAMD(train2[, !names(train2) %in% "Exited"], ncp = 10, graph = FALSE)
  train2_coord <- as.data.frame(famd_train$ind$coord)
  train2_coord$Exited <- train2$Exited
  
  # Proyección de test2
  test2_proy <- predict(famd_train, newdata = test2[, !names(test2) %in% "Exited"])
  test2_coord <- as.data.frame(test2_proy$coord)
  test2_coord$Exited <- test2$Exited
  names(test2_coord) <- names(train2_coord)
  
  # Entrenamiento Random Forest (tryCatch para evitar errores)
  rf_model <- tryCatch({
    train(
      Exited ~ ., 
      data = train2_coord, 
      method = "rf",
      metric = "ROC",
      trControl = ctrl,
      tuneGrid = tuneGrid,
      ntree = 500,
      nodesize = 50
    )
  }, error = function(e) NULL)
  
  if (is.null(rf_model)) next
  
  # Predicciones probabilísticas
  probs_train <- predict(rf_model, newdata = train2_coord, type = "prob")
  probs_test  <- predict(rf_model, newdata = test2_coord,  type = "prob")
  
  # ROC y umbral óptimo
  roc_obj <- roc(test2_coord$Exited, probs_test$Yes, percent = TRUE)
  coords_opt <- coords(roc_obj, "best", ret = c("threshold"), best.method = "closest.topleft")
  umbral <- coords_opt$threshold
  
  # Predicciones con threshold
  pred_train_opt <- factor(ifelse(probs_train$Yes > umbral, "Yes", "No"), levels = c("Yes", "No"))
  pred_test_opt  <- factor(ifelse(probs_test$Yes > umbral, "Yes", "No"), levels = c("Yes", "No"))
  
  # Matrices de confusión
  conf_train <- caret::confusionMatrix(pred_train_opt, train2_coord$Exited, positive = "Yes", mode = "prec_recall")
  conf_test  <- caret::confusionMatrix(pred_test_opt,  test2_coord$Exited,  positive = "Yes", mode = "prec_recall")
  
  # Guardar resultados
  results_rf <- rbind(results_rf, data.frame(
    seed = s,
    f1_train = conf_train$byClass["F1"],
    f1_test  = conf_test$byClass["F1"]
  ))
  
  if (s %% 50 == 0) cat("Semilla:", s, "procesada\n")
}

# Top 15 semillas por F1 test

top15_rf <- results_rf %>% arrange(desc(f1_test)) %>% slice(1:15)
top15_rf
