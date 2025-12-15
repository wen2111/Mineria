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
set.seed(123)
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
                              #,search = "grid"
)

xgb_grid <- expand.grid(
  nrounds = 150,
  max_depth = c(2,3,5),
  eta = c(0.1,0.4),
  gamma = c(3,5,6) ,             
  colsample_bytree = c(0.8,0.7),
  min_child_weight = 1,
  subsample = c(0.7,0.8)
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

fit_tuning$bestTune


# Predicciones probabilísticas
train_pred_prob <- predict(fit_tuning, newdata = train2, type = "prob")
test_pred_prob  <- predict(fit_tuning, newdata = test2,  type = "prob")

train_pred_cut <- ifelse(train_pred_prob$Yes > 0.2071429, "Yes", "No")
test_pred_cut  <- ifelse(test_pred_prob$Yes > 0.2071429, "Yes", "No")

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

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)

kpis <- data.frame( Dataset = c("Train2", "Test2"),
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

#final train
fit_tuning$bestTune
xgb_grid_no_tuning <- expand.grid(
  nrounds = 150,
  max_depth = 3,
  eta = 0.1,
  gamma = 3,              
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.5
)
set.seed(459)
fit_final_glm <- train(Exited ~ ., data=train, 
                       method = "xgbTree",
                       trControl = ctrl_boot_auc, metric = "F1",
                       preProcess = c("scale"),verbosity = 0,
                       tuneGrid=xgb_grid_no_tuning
                       
)
train_pred_prob <- predict(fit_final_glm, newdata = train, type = "prob")
train_pred_cut <- ifelse(train_pred_prob$Yes > 0.2071429, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))

# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train$Exited,positive = "Yes")
conf_train
f1_score(conf_train)

# prediccion
pred_kaggle_prob <- predict(fit_final_glm, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > 0.2071429, "Yes", "No")
test$ID<-data$ID[7001:10000]
submission <- data.frame(ID = test$ID, Exited = pred_kaggle_class)
write.csv(submission, "xgboost_tuniing2.csv", row.names = FALSE)
                    
                    