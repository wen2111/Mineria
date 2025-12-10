library(caret)
library(pROC)
library(MLmetrics)

##
mydata <- data_reducida_plus
mydata$group <- NULL

# SEPARAR TRAIN Y TEST
train <- mydata[1:7000, ]
test  <- mydata[7001:10000, ]

# LABELS PARA EXITED
train$Exited <- factor(train$Exited, levels = c("0","1"), labels = c("No","Yes")) 
test$Exited  <- factor(test$Exited,  levels = c("0","1"), labels = c("No","Yes")) 

# PARTICION TRAIN2/TEST2
set.seed(123)
index <- createDataPartition(train$Exited, p = 0.8, list = FALSE)
train2 <- train[index, ]
test2  <- train[-index, ]

## DUMMIES

# 1. Definir transformador de dummies
dmy <- dummyVars(Exited ~ ., data = train2, fullRank = TRUE)

# 2. Aplicar a train2
train2_x <- as.data.frame(predict(dmy, newdata = train2))
train2_x$Exited <- train2$Exited  # añadimos la y al final

# 3. Aplicar al test2 (misma estructura de columnas)
test2_x <- as.data.frame(predict(dmy, newdata = test2))
test2_x$Exited <- test2$Exited

## usar train2_x en lugar de train2:

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
  data = train2_x,
  method = "xgbTree",
  metric = "ROC", 
  trControl = ctrl_xgb,
  tuneGrid = tuneGrid_xgb
)

fit_xgb
plot(fit_xgb)

prob_test <- predict(fit_xgb, newdata = test2_x, type = "prob")[, "Yes"]

## 4. Buscar mejor umbral por F1 en test2_x
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

best_thresh <- find_best_threshold(prob_test, test2_x$Exited)
cat("Mejor umbral según F1 en test2_x:", best_thresh, "\n")

check_threshold <- function(th) {
  preds <- factor(ifelse(prob_test > th, "Yes", "No"),
                  levels = c("No", "Yes"))
  cm <- confusionMatrix(preds, test2_x$Exited, positive = "Yes")
  data.frame(
    threshold  = th,
    Sensitivity = cm$byClass["Sensitivity"],
    Precision   = cm$byClass["Precision"],
    BalancedAcc = mean(c(cm$byClass["Sensitivity"], cm$byClass["Specificity"]))
  )
}

ths <- seq(0.2, 0.6, by = 0.05)
res <- do.call(rbind, lapply(ths, check_threshold))
print(res)
best_thresh <- 0.25  # según la tabla manual

#################

# Confusion matrix en test2_x con ese umbral
pred_test_best <- ifelse(prob_test > best_thresh, "Yes", "No")
pred_test_best <- factor(pred_test_best, levels = c("No", "Yes"))

cm_test2 <- confusionMatrix(pred_test_best, test2_x$Exited, positive = "Yes")
cm_test2

best_grid <- fit_xgb$bestTune
train_full_x <- as.data.frame(predict(dmy, newdata = train))
train_full_x$Exited <- train$Exited

set.seed(123)
final_xgb <- train(
  Exited ~ .,
  data = train_full_x,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl_xgb,
  tuneGrid = best_grid
)

final_xgb

## Evaluación en train completo con best_thresh
prob_train_full <- predict(final_xgb, newdata = train_full_x, type = "prob")[, "Yes"]
pred_train_full <- ifelse(prob_train_full > best_thresh, "Yes", "No")
pred_train_full <- factor(pred_train_full, levels = c("No", "Yes"))

cm_train <- confusionMatrix(pred_train_full, train_full_x$Exited, positive = "Yes")
cm_train

## 6. Aplicar al test (las 3000 obs para submission)
test_full_x <- as.data.frame(predict(dmy, newdata = test))
# test_full_x$Exited <- test$Exited

prob_test_full <- predict(final_xgb, newdata = test_full_x, type = "prob")[, "Yes"]

# Pred binaria final usando best_thresh (o 0.5)
class_test_xgb <- ifelse(prob_test_full > best_thresh, "Yes", "No")
class_test_xgb <- factor(class_test_xgb, levels = c("No", "Yes"))


## 7. Crear submission
submission <- data.frame(
  ID     = data$ID[7001:10000],
  Exited = class_test_xgb
)

write.csv(submission, "submission_xgb_dummies.csv", row.names = FALSE)
