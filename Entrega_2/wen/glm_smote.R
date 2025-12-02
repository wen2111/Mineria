
###############################################
##### GLM BOOTSTRAP REDUCIDA SMOTE ####
###############################################

library(caret)

mydata <- data_reducida
#dummifico
x<-mydata[,-3] #quito la respuesta
x<-x[,1:4] # cojo solo las cat
x <- fastDummies::dummy_cols(x, 
                             remove_first_dummy = TRUE,  
                             remove_selected_columns = TRUE)
x<-cbind(x,mydata[,6:7]) # adjunto las numericas
x$Exited<-mydata$Exited # añado la respuesta
mydata<-x

# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("1","0"),
                       labels = c("Yes","No"))

# PARTICION TRAIN2/TEST2

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

#smote

train2<-SMOTE(train2[,-9],train2$Exited,K=5,dup_size = 1)
train2<-train2$data
names(train2)[9]<-"Exited"
train2$Exited <- factor(train2$Exited, 
                        levels = c("Yes", "No"))
# BOOTSTRAP

ctrl_boot_auc <- trainControl(method = "boot", 
                              number = 200,         # 200 samples
                              classProbs = TRUE,
                              sampling = "up",
                              summaryFunction = twoClassSummary)
fit_boot_auc <- train(Exited ~ ., data=train2, 
                      method = "glm", family = "binomial",
                      trControl = ctrl_boot_auc, metric = "ROC",
                      preProcess = c("center", "scale"))
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

#final train

set.seed(123)
fit_final_glm <- train(
  Exited ~ ., data = train,
  method = "glm", family = "binomial",
  trControl = ctrl_boot_auc,
  metric = "ROC"
)

train_pred_prob <- predict(fit_final_glm, newdata = train, type = "prob")
# si prob > threshold → Yes
train_pred_cut <- ifelse(train_pred_prob$Yes > best_th, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("Yes","No"))
# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train$Exited)
conf_train

pred_kaggle_prob <- predict(fit_final_glm, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > best_th, "Yes", "No")
test$ID<-data$ID[7001:10000]
submission <- data.frame(ID = test$ID, Exited = pred_kaggle_class)
write.csv(submission, "glm_smote.csv", row.names = FALSE)