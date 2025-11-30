library(caret)
mydata <- data_reducida_plus
mydata$group<-NULL
# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("0","1"),  
                       labels = c("No","Yes")) 
test$Exited <- factor(test$Exited,
                       levels = c("0","1"),  
                       labels = c("No","Yes")) 

# PARTICION TRAIN2/TEST2

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.8, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

################### 1

mtry.class <- sqrt(ncol(train2) - 1)
tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))
set.seed(123)
rf.caret <- train(Exited ~ ., data = train2,method = "rf",tuneGrid = tuneGrid)
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

ptrain <- ifelse(ptrain[,2] > 0.1, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.1, "Yes", "No")
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
  number = 10,
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
  ntree = 500,             # número de árboles
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

ptrain <- ifelse(ptrain[,2] > 0.2, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.2, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

confusionMatrix(ptrain, train2$Exited, positive="Yes")
(cm<-confusionMatrix(ptest, test2$Exited, positive="Yes"))

precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1

# ---------------------------
# Instalar si no lo tienes
#install.packages("MLmetrics")

# Cargar el paquete
library(MLmetrics)

find_best_threshold <- function(probs, actuals) {
  thresholds <- seq(0.1, 0.9, by = 0.01)
  f1_scores <- sapply(thresholds, function(t) {
    preds <- factor(ifelse(probs > t, "Yes", "No"), levels = c("No","Yes"))
    F1_Score(y_true = actuals, y_pred = preds, positive = "Yes")
  })
  best <- thresholds[which.max(f1_scores)]
  return(best)
}

best_thresh <- find_best_threshold(ptest[,2], test2$Exited)
cat("Mejor umbral según F1 en test:", best_thresh, "\n")

ptest <- ifelse(ptest[,2] > 0.46, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

(cm<-confusionMatrix(ptest, test2$Exited, positive="Yes"))

precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1

##################333 3


rf <- randomForest(Exited ~ ., data = train2,max_features="sqrt",min_samples_split=10, min_samples_leaf=5,max_depth=10)
rf
plot(rf,main="")
legend("right", colnames(rf$err.rate), lty = 1:5, col = 1:6)


ptest <- predict(rf, test2, type = 'prob')
ptrain <- predict(rf, train2, type = 'prob')

ptrain <- ifelse(ptrain[,2] > 0.24, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > 0.24, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

confusionMatrix(ptrain, train2$Exited, positive="Yes")
(cm<-confusionMatrix(ptest, test2$Exited, positive="Yes"))

precision <- cm$byClass["Precision"]     # TP / (TP + FP)
recall <- cm$byClass["Sensitivity"]      # TP / (TP + FN)
f1 <- 2 * (precision * recall) / (precision + recall)
f1


pred3 <- predict(rf, newdata = test)
caret::confusionMatrix(pred3, test$yesno,positive="y") 