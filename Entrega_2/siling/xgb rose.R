library(caret)
library(fastDummies)
library(ROSE)
library(xgboost)
library(pROC)
library(MLmetrics)

set.seed(123)

mydata <- data_reducida

# dummificación
x <- mydata[ , -3] # quitar la respuesta
x <- x[ , 1:4] # categóricas
x <- fastDummies::dummy_cols(
  x, 
  remove_first_dummy     = TRUE,  
  remove_selected_columns = TRUE
)

x <- cbind(x, mydata[ , 6:7]) # adjuntar las numericas
x$Exited <- mydata$Exited     # añadir la respuesta
mydata <- x
names(mydata) <- make.names(names(mydata))


# SEPARAR TRAIN Y TEST
train <- mydata[1:7000, ]
test  <- mydata[7001:10000, ]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("1","0"),
                       labels = c("Yes","No"))
# test labels
test$Exited  <- factor(test$Exited,
                       levels = c("1","0"),
                       labels = c("Yes","No"))

# PARTICION TRAIN2/TEST2
set.seed(123)
index  <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ]   # train interno
test2  <- train[-index, ]  # test interno


set.seed(123)
rose_res <- ROSE(Exited ~ ., data = train2, seed = 123)
train2   <- rose_res$data

train2$Exited <- factor(train2$Exited,
                        levels = c("Yes","No"))
# para ver el balance
table(train2$Exited)
table(test2$Exited)

# PREPARAR MATRICES PARA XGBOOST
ctrl_xgb <- trainControl(
  method          = "cv",
  number          = 5, 
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter     = FALSE,
  savePredictions = "final"
)

tuneGrid_xgb <- expand.grid(
  nrounds          = c(100, 200),
  max_depth        = c(3, 5),
  eta              = c(0.05, 0.1),
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 0.8
)

set.seed(123)
fit_xgb <- train(
  Exited ~ .,
  data      = train2,
  method    = "xgbTree",
  metric    = "ROC", 
  trControl = ctrl_xgb,
  tuneGrid  = tuneGrid_xgb
)


fit_xgb
plot(fit_xgb)


## 4. en test2 
# P(Exited == "Yes")
prob_test2 <- predict(fit_xgb, newdata = test2, type = "prob")[, "Yes"]

# buscar el mejor threshold según F1
metrics_by_threshold <- function(probs, actuals) {
  ths <- seq(0.05, 0.6, by = 0.01)
  res <- lapply(ths, function(t) {
    preds <- factor(ifelse(probs > t, "Yes", "No"),
                    levels = levels(actuals))
    cm <- confusionMatrix(preds, actuals, positive = "Yes")
    data.frame(
      threshold   = t,
      Sensitivity = cm$byClass["Sensitivity"],
      Specificity = cm$byClass["Specificity"],
      Precision   = cm$byClass["Precision"],
      BalancedAcc = mean(c(cm$byClass["Sensitivity"],
                           cm$byClass["Specificity"]))
    )
  })
  do.call(rbind, res)
}

res_all <- metrics_by_threshold(prob_test2, test2$Exited)

# View(res_all)
# head(res_all[order(-res_all$Sensitivity), ])
# min Precision acceptable, ejemplo >= 0.35
target_min_precision <- 0.35

cand <- subset(res_all, Precision >= target_min_precision)

# Precision, maximizar Sensitivity
best_idx <- which.max(cand$Sensitivity)
best_row <- cand[best_idx, ]
best_row

best_thresh_recall <- best_row$threshold
best_thresh_recall

best_grid <- fit_xgb$bestTune

set.seed(123)
train_full_rose <- ROSE(Exited ~ ., data = train, seed = 123)$data
train_full_rose$Exited <- factor(train_full_rose$Exited,
                                 levels = c("Yes","No"))
set.seed(123)
final_xgb <- train(
  Exited ~ .,
  data      = train_full_rose,
  method    = "xgbTree",
  metric    = "ROC",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid  = best_grid
)

final_xgb
## train(7000)
prob_train_full <- predict(final_xgb, newdata = train, type = "prob")[, "Yes"]

pred_train_full <- factor(
  ifelse(prob_train_full > best_thresh_recall, "Yes", "No"),
  levels = levels(train$Exited)
)

cm_train <- confusionMatrix(pred_train_full, train$Exited, positive = "Yes")
cm_train

## test(3000)
prob_test_full <- predict(final_xgb, newdata = test, type = "prob")[, "Yes"]

pred_test_full <- factor(
  ifelse(prob_test_full > best_thresh_recall, "Yes", "No"),
  levels = levels(test$Exited)
)

cm_test <- confusionMatrix(pred_test_full, test$Exited, positive = "Yes")
cm_test

## submission
class_test_xgb <- pred_test_full 

submission <- data.frame(
  ID     = data_reducida$ID[7001:10000],
  Exited = class_test_xgb
)
write.csv(submission, "submission_xgb_rose.csv", row.names = FALSE)
