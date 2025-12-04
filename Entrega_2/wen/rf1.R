library(caret)
library(DAAG)
library(mlbench)
library(pROC)
library(printr)
library(randomForest)
library(ranger)

mydata <- data_reducida
#######3
mydata$CustomerSegment<-NULL
mydata$LoanStatus<-NULL
mydata$IsActiveMember<-NULL
mydata$Gender<-NULL
mydata$ComplaintsCount<-NULL
mydata$HasCrCard<-NULL
mydata$SavingsAccountFlag<-NULL
####
mydata$group<-NULL
mydata$Surname<-NULL
mydata$ID<-NULL
#########
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

############## 0

rf <- randomForest(Exited ~ ., data = train2,)
rf
varImpPlot(rf)

################### 1

mtry.class <- sqrt(ncol(train2) - 1)
tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))
tuneGrid
set.seed(123)
rf.caret <- train(Exited ~ ., data = train2,method = "rf",
                  tuneGrid = tuneGrid,nodesize=15,maxnodes=25)
plot(rf.caret)
rf.caret

p4 <- predict(rf.caret, test2, type = 'prob')
head(p4)
p4 <- p4[,2]
r <- multiclass.roc(test2$Exited, p4, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')

ptest <- predict(rf.caret, test2, type = 'prob')
ptrain <- predict(rf.caret, train2, type = 'prob')

ptrain <- ifelse(ptrain[,2] > 0.08696, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.08696, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

conf_train<-confusionMatrix(ptrain, train2$Exited, positive="Yes")
conf_test<-confusionMatrix(ptest, test2$Exited, positive="Yes")
conf_test
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

######### ok
final_rf1 <- train(Exited ~ ., data = train,method = "rf",tuneGrid = tuneGrid,
                   nodesize=15,maxnodes=25)

ptrain <- predict(final_rf1, train, type = 'prob')
ptrain <- ifelse(ptrain[,2] > 0.06079, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))
cm<-confusionMatrix(ptrain, train$Exited, positive="Yes")
cm
precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1

pred_kaggle_prob <- predict(final_rf1, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > 0.06079, "Yes", "No")


test$ID<-data$ID[7001:10000]
submission <- data.frame(ID = test$ID, Exited = pred_kaggle_class)
write.csv(submission, "rf_imput.csv", row.names = FALSE)

##################### 2

ctrl_rf <- trainControl(
  method = "cv",         # 5-fold cross-validation
  number = 8,
  classProbs = TRUE,     # necesario para AUC
  summaryFunction = twoClassSummary,
  verboseIter = FALSE
)

mtry.class <- sqrt(ncol(train2) - 1)

tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))
set.seed(123)

fit_rf <- train(
  Exited ~ .,
  data = train2,
  method = "rf",
  metric = "ROC",          # optimizar AUC
  trControl = ctrl_rf,
  tuneGrid = tuneGrid,
  ntree = 500,        # número de árboles
  nodesize=15, maxnodes=25,
  preProcess = c("center", "scale") # no modifica nada las metricas
  )

plot(fit_rf)
fit_rf
pred <- predict(fit_rf, newdata = test2,type = 'prob')
pred <- pred[,2]
r <- multiclass.roc(test2$Exited, pred, percent = TRUE)
roc <- r[['rocs']]
r1 <- roc[[1]]
plot.roc(r1,print.auc=TRUE,
         auc.polygon=TRUE,
         grid=c(0.1, 0.2),
         grid.col=c("green", "red"),
         max.auc.polygon=TRUE,
         auc.polygon.col="lightblue",
         print.thres=TRUE,
         main= 'ROC Curve')

obs <- test2$Exited
caret::postResample(pred4, obs)

ptest <- predict(fit_rf, test2, type = 'prob')
ptrain <- predict(fit_rf, train2, type = 'prob')

ptrain <- ifelse(ptrain[,2] > 0.03499, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.03499, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

conf_train<-confusionMatrix(ptrain, train2$Exited, positive="Yes")
(conf_test<-confusionMatrix(ptest, test2$Exited, positive="Yes"))

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
# peores resultados con los predictores del grafico del varImp.
# la prediccion de los YES similar, pero predice pero los casos NO.

# con el balanceo
## mejor sensibilidad con el dataset reducido
## el f1 baja levemente