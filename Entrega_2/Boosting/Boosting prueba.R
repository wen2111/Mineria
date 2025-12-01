library(caret)
library(pROC)
# install.packages("MLmetrics")
library(MLmetrics)

##
mydata <- data_reducida_plus
mydata$group<-NULL
# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]

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
train2 <- train[index, ]
test2  <- train[-index, ]
head(train2)
head(test2)

################### 1

## CV + AUC + prop + SMOTE
ctrl_xgb <- trainControl(
  method = "cv",
  number = 5, 
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "smote",
  verboseIter = FALSE
)

tuneGrid_xgb <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 5),
  eta = c(0.05, 0.1),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

set.seed(123)
fit_xgb <- train(
  Exited ~ .,
  data = train2,
  method = "xgbTree",
  metric = "ROC", 
  trControl = ctrl_xgb,
  tuneGrid = tuneGrid_xgb
)

plot(fit_xgb)
fit_xgb


## umbral 0.5
pred_train_raw <- predict(fit_xgb, newdata = train2, type = "raw")
pred_test_raw  <- predict(fit_xgb, newdata = test2,  type = "raw")

confusionMatrix(pred_train_raw, train2$Exited, positive = "Yes")
confusionMatrix(pred_test_raw,  test2$Exited,  positive = "Yes")



prob_test <- predict(fit_xgb, newdata = test2, type = "prob")[, "Yes"]

roc_xgb <- roc(
  response  = test2$Exited, 
  predictor = prob_test,
  levels    = c("No", "Yes"), 
  direction = "<",
  percent   = TRUE
)

plot.roc(roc_xgb,
         print.auc       = TRUE,
         auc.polygon     = TRUE,
         grid            = c(0.1, 0.2),
         grid.col        = c("green", "red"),
         max.auc.polygon = TRUE,
         auc.polygon.col = "lightblue",
         print.thres     = TRUE,
         main            = "ROC Curve - XGBoost (test2)")


## definir la función para encontrar el mejor umbral según F1
find_best_threshold <- function(probs, actuals) {
  thresholds <- seq(0.1, 0.9, by = 0.01)
  f1_scores <- sapply(thresholds, function(t) {
    preds <- factor(ifelse(probs > t, "Yes", "No"),
                    levels = c("No", "Yes"))
    F1_Score(y_true = actuals, y_pred = preds, positive = "Yes")
  })
  best <- thresholds[which.max(f1_scores)]
  return(best)
}

## umbral en test 2
best_thresh <- find_best_threshold(prob_test, test2$Exited)
cat("Mejor umbral según F1 en test2:", best_thresh, "\n")

## con el mejor umbral, calcular F1 en test2
pred_test_best <- ifelse(prob_test > best_thresh, "Yes", "No")
pred_test_best <- factor(pred_test_best, levels = c("No", "Yes"))

cm <- confusionMatrix(pred_test_best, test2$Exited, positive = "Yes")
cm

precision <- cm$byClass["Precision"]
recall    <- cm$byClass["Sensitivity"]
f1        <- 2 * (precision * recall) / (precision + recall)
f1


## el mejor grid
best_grid <- fit_xgb$bestTune

set.seed(123)
final_xgb <- train(
  Exited ~ .,
  data = train,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl_xgb,
  tuneGrid = best_grid
)

final_xgb


prob_train_full <- predict(final_xgb, newdata = train, type = "prob")[, "Yes"]
pred_train_full <- ifelse(prob_train_full > best_thresh, "Yes", "No")
pred_train_full <- factor(pred_train_full, levels = c("No", "Yes"))

cm_train <- confusionMatrix(pred_train_full, train$Exited, positive = "Yes")
cm_train

precision <- cm_train$byClass["Precision"]
recall    <- cm_train$byClass["Sensitivity"]
f1_train  <- 2 * (precision * recall) / (precision + recall)
f1_train



# test$Exited <- factor(test$Exited, levels = c("0","1"), labels = c("No","Yes"))

if ("Exited" %in% names(test)) {
  prob_test_full <- predict(final_xgb, newdata = test, type = "prob")[, "Yes"]
  pred_test_full <- ifelse(prob_test_full > best_thresh, "Yes", "No")
  pred_test_full <- factor(pred_test_full, levels = c("No", "Yes"))
  
  cm_test <- confusionMatrix(pred_test_full, test$Exited, positive = "Yes")
  cm_test
  
  precision <- cm_test$byClass["Precision"]
  recall    <- cm_test$byClass["Sensitivity"]
  f1_test   <- 2 * (precision * recall) / (precision + recall)
  f1_test
}


## pred y class
pred_test_xgb <- predict(final_xgb, newdata = test, type = "prob")[, "Yes"]

## Predicción binaria final

class_test_xgb <- ifelse(pred_test_xgb > 0.5, "Yes", "No")
class_test_xgb <- factor(class_test_xgb, levels = c("No", "Yes"))


submission <- data.frame(
  ID     = data$ID[7001:10000],
  Exited = class_test_xgb
)

write.csv(submission, "submission_xgb.csv", row.names = FALSE)
