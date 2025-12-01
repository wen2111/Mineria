library(caret)
library(DAAG)
library(mlbench)
library(pROC)
library(printr)
library(randomForest)
library(ranger)

mydata <- data_imputado
mydata$CustomerSegment<-NULL
mydata$LoanStatus<-NULL
mydata$IsActiveMember<-NULL
mydata$Gender<-NULL
mydata$ComplaintsCount<-NULL
mydata$HasCrCard<-NULL
mydata$SavingsAccountFlag<-NULL
mydata$group<-NULL
mydata$Surname<-NULL
mydata$ID<-NULL
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
                  tuneGrid = tuneGrid,nodesize=20,maxnodes=50)
plot(rf.caret)
rf.caret
pred4 <- predict(rf.caret, newdata = test2)
ptrain <- predict(rf.caret, train2, type = 'raw')
confusionMatrix(ptrain, train2$Exited, positive="Yes")
confusionMatrix(pred4, test2$Exited, positive="Yes")

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
obs <- test2$Exited
caret::postResample(pred4, obs)

ptest <- predict(rf.caret, test2, type = 'prob')
ptrain <- predict(rf.caret, train2, type = 'prob')

ptrain <- ifelse(ptrain[,2] > 0.05, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.05, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

confusionMatrix(ptrain, train2$Exited, positive="Yes")
cm<-confusionMatrix(ptest, test2$Exited, positive="Yes")
cm
precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1

######### ok
final_rf1 <- train(Exited ~ ., data = train,method = "rf",tuneGrid = tuneGrid)

ptrain <- predict(final_rf1, train, type = 'prob')
ptrain <- ifelse(ptrain[,2] > 0.1, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))
cm<-confusionMatrix(ptrain, train$Exited, positive="Yes")
cm
precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1

pred_kaggle_prob <- predict(final_rf1, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > 0.1, "Yes", "No")


test$ID<-data$ID[7001:10000]
submission <- data.frame(ID = test$ID, Exited = pred_kaggle_class)
write.csv(submission, "rf2.csv", row.names = FALSE)

##################### 2

ctrl_rf <- trainControl(
  method = "cv",         # 5-fold cross-validation
  number = 8,
  classProbs = TRUE,     # necesario para AUC
  summaryFunction = twoClassSummary,
  sampling = "smote",    # balanceo automático
  verboseIter = FALSE,
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
  nodesize=30, maxnodes=50
  
)

plot(fit_rf)
fit_rf
pred <- predict(fit_rf, newdata = test2,type = 'raw')
ptrain <- predict(fit_rf, newdata=train2, type = 'raw')
confusionMatrix(ptrain, train2$Exited, positive="Yes")
confusionMatrix(pred, test2$Exited, positive="Yes")
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

ptrain <- ifelse(ptrain[,2] > 0.15, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.15, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

confusionMatrix(ptrain, train2$Exited, positive="Yes")
(cm<-confusionMatrix(ptest, test2$Exited, positive="Yes"))

precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1
