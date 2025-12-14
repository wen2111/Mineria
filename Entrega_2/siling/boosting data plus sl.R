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

# ---- DUMMIES (train2-based) ----
dmy <- dummyVars(Exited ~ ., data = train2, fullRank = TRUE)

train2_x <- as.data.frame(predict(dmy, newdata = train2))
train2_x$Exited <- train2$Exited

test2_x <- as.data.frame(predict(dmy, newdata = test2))
test2_x$Exited <- test2$Exited

# ---- XGBoost CV + SMOTE + savePredictions ----
ctrl_xgb <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  sampling = "smote",
  savePredictions = "final",
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

fit_xgb <- train(
  Exited ~ .,
  data = train2_x,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl_xgb,
  tuneGrid = tuneGrid_xgb
)

print(fit_xgb)
plot(fit_xgb)

# ---- Threshold selection by F2 on OOF predictions ----
F_beta <- function(precision, recall, beta = 2){
  (1 + beta^2) * precision * recall / (beta^2 * precision + recall + 1e-12)
}

find_best_threshold_F2 <- function(probs, actuals){
  thresholds <- seq(0.05, 0.95, by = 0.01)
  y <- factor(actuals, levels = c("No","Yes"))
  scores <- sapply(thresholds, function(t){
    pred <- factor(ifelse(probs >= t, "Yes","No"), levels = c("No","Yes"))
    cm <- table(pred, y)
    precision <- cm["Yes","Yes"] / (cm["Yes","Yes"] + cm["Yes","No"] + 1e-12)
    recall    <- cm["Yes","Yes"] / (cm["Yes","Yes"] + cm["No","Yes"] + 1e-12)
    F_beta(precision, recall, beta = 2)
  })
  thresholds[which.max(scores)]
}

oof <- fit_xgb$pred
prob_cols <- setdiff(names(oof), c("pred","obs","rowIndex","Resample"))
yes_col <- prob_cols[grepl("^Yes$", prob_cols)]
if(length(yes_col) == 0) yes_col <- prob_cols[grepl("Yes", prob_cols)][1]

best_thresh <- find_best_threshold_F2(oof[[yes_col]], oof$obs)
cat("Best threshold (F2 on OOF):", best_thresh, "\n")

# ---- KPI helpers ----
get_kpis_from_cm <- function(cm){
  bc <- cm$byClass
  overall <- cm$overall
  c(
    Accuracy    = unname(overall["Accuracy"]),
    Kappa       = unname(overall["Kappa"]),
    Sensitivity = unname(bc["Sensitivity"]),      # Recall(Yes)
    Specificity = unname(bc["Specificity"]),
    Precision   = unname(bc["Precision"]),
    F1          = unname(bc["F1"]),
    BalancedAcc = unname(bc["Balanced Accuracy"])
  )
}

get_auc <- function(probs, actuals){
  roc_obj <- roc(response = actuals, predictor = probs, levels = c("No","Yes"), direction = "<")
  as.numeric(auc(roc_obj))
}

get_f2_from_cm <- function(cm){
  bc <- cm$byClass
  F_beta(unname(bc["Precision"]), unname(bc["Sensitivity"]), beta = 2)
}

# ---- Evaluate on train2/test2 ----
prob_train2 <- predict(fit_xgb, newdata = train2_x, type = "prob")[,"Yes"]
pred_train2 <- factor(ifelse(prob_train2 >= best_thresh, "Yes","No"), levels = c("No","Yes"))
cm_train2 <- confusionMatrix(pred_train2, train2_x$Exited, positive = "Yes")

prob_test2 <- predict(fit_xgb, newdata = test2_x, type = "prob")[,"Yes"]
pred_test2 <- factor(ifelse(prob_test2 >= best_thresh, "Yes","No"), levels = c("No","Yes"))
cm_test2 <- confusionMatrix(pred_test2, test2_x$Exited, positive = "Yes")

# ---- Refit on full train with best grid (stable dummies) ----
best_grid <- fit_xgb$bestTune

dmy_full <- dummyVars(Exited ~ ., data = train, fullRank = TRUE)

train_full_x <- as.data.frame(predict(dmy_full, newdata = train))
train_full_x$Exited <- train$Exited

final_xgb <- train(
  Exited ~ .,
  data = train_full_x,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl_xgb,
  tuneGrid = best_grid
)

print(final_xgb)

prob_train_full <- predict(final_xgb, newdata = train_full_x, type = "prob")[,"Yes"]
pred_train_full <- factor(ifelse(prob_train_full >= best_thresh, "Yes","No"), levels = c("No","Yes"))
cm_train_full <- confusionMatrix(pred_train_full, train_full_x$Exited, positive = "Yes")

# ---- KPI table ----
kpi_table <- rbind(
  Train2 = c(get_kpis_from_cm(cm_train2), AUC = get_auc(prob_train2, train2_x$Exited), F2 = get_f2_from_cm(cm_train2)),
  Test2  = c(get_kpis_from_cm(cm_test2),  AUC = get_auc(prob_test2,  test2_x$Exited),  F2 = get_f2_from_cm(cm_test2)),
  Train_Full = c(get_kpis_from_cm(cm_train_full), AUC = get_auc(prob_train_full, train_full_x$Exited), F2 = get_f2_from_cm(cm_train_full))
)
print(kpi_table)

# ---- Predict on test (submission set) ----
test_full_x <- as.data.frame(predict(dmy_full, newdata = test))
prob_test_full <- predict(final_xgb, newdata = test_full_x, type = "prob")[,"Yes"]
train_rate_yes <- mean(train$Exited == "Yes")

thr_match_rate <- as.numeric(quantile(prob_test_full, probs = 1 - train_rate_yes))

class_test_xgb_rate <- factor(ifelse(prob_test_full >= thr_match_rate, "Yes", "No"),
                              levels = c("No","Yes"))

table(class_test_xgb_rate)
prop.table(table(class_test_xgb_rate))

submission <- data.frame(
  ID     = data$ID[7001:10000],
  Exited = class_test_xgb
)
write.csv(submission, "submission_data_plus_sl.csv", row.names = FALSE)
