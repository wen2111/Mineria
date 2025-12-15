#####################################################
######## XGBOOTING REDUCIDA CON HASBALANCE ##########

library(caret)
library(xgboost)
library(Matrix)
library(pROC)
library(ggplot2)
library(dplyr)
library(scales)

mydata <- data_reducida

#dummifico data reducido
x<-mydata[,-3] #quito la respuesta
x<-x[,1:4] # cojo solo las cat
x <- fastDummies::dummy_cols(x, 
                             remove_first_dummy = TRUE,  
                             remove_selected_columns = TRUE)
x<-cbind(x,mydata[,6:7]) # adjunto las numericas
x$Exited<-mydata$Exited # añado la respuesta
mydata<-x
mydata$hasB<-ifelse(mydata$Balance==0,0,1)
mydata$Balance<-NULL
# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("0","1"),
                       labels = c("No","Yes"))

# PARTICION TRAIN2/TEST2
semillas <- sample(1:10000, 1000)
f1_train <- numeric(length(semillas))
f1_test  <- numeric(length(semillas))

for (i in seq_along(semillas)) {
  
set.seed(semillas[i])
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

library(MLmetrics)

f1_recall_summary <- function(data, lev = NULL, model = NULL) {
  precision <- Precision(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  recall <- Recall(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  f1 <- F1_Score(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  c(F1 = f1, Recall = recall, Precision = precision)
}

ctrl_boot_auc <- trainControl(method = "cv", 
                              number = 5 ,         
                              classProbs = TRUE,
                              summaryFunction = f1_recall_summary
)

xgb_grid <- expand.grid(
  nrounds = 150 ,
  max_depth = 2 ,
  eta = 0.4,
  gamma = 3,             
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.7
)

fit_tuning <- train(
  Exited ~ ., 
  data = train2,
  method = "xgbTree",
  metric = "F1",
  trControl = ctrl_boot_auc,
  tuneGrid = xgb_grid,  
  preProcess = "scale",
  verbosity = 0
)

# Predicciones probabilísticas
train_pred_prob <- predict(fit_tuning, newdata = train2, type = "prob")
test_pred_prob  <- predict(fit_tuning, newdata = test2,  type = "prob")
#best p
f1_score <- function(y_true, y_pred) {
  tp <- sum(y_true == "Yes" & y_pred == "Yes")
  fp <- sum(y_true == "No"  & y_pred == "Yes")
  fn <- sum(y_true == "Yes" & y_pred == "No")
  
  precision <- tp / (tp + fp)
  recall    <- tp / (tp + fn)
  
  2 * precision * recall / (precision + recall)
}

thresholds <- seq(0.01, 0.99, by = 0.01)

f1_values <- sapply(thresholds, function(t) {
  pred <- ifelse(train_pred_prob$Yes > t, "Yes", "No")
  f1_score(train2$Exited, pred)
})

best_threshold <- thresholds[which.max(f1_values)]

train_pred_cut <- ifelse(train_pred_prob$Yes > best_threshold, "Yes", "No")
test_pred_cut  <- ifelse(test_pred_prob$Yes > best_threshold, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))
test_pred_cut  <- factor(test_pred_cut,  levels = c("No","Yes"))

# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train2$Exited,positive = "Yes")
conf_test  <- confusionMatrix(test_pred_cut, test2$Exited,positive = "Yes")

# F1-score function
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train[i] <- f1_score(conf_train)
f1_test[i]  <- f1_score(conf_test)

cat("Semilla:", semillas[i],
    "| F1 Train:", round(f1_train[i], 4),
    "| F1 Test:", round(f1_test[i], 4), "\n")

}  
