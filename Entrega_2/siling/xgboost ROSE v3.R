library(caret)
library(fastDummies)
library(ROSE)
library(MLmetrics)
library(xgboost)

set.seed(123)

## 1. 读数据 + 基础清理 ----
mydata <- data_reducida_plus
mydata$group <- NULL   # 如果有这个变量就先去掉

## 确认 Exited 原始形式（跑一次看看就好）
# print(table(mydata$Exited))
# str(mydata$Exited)

## 假设 Exited 原来是数值 0/1，如果其实是字符 "0"/"1"，把 levels 改成 c("0","1")
mydata$Exited <- factor(
  mydata$Exited,
  levels = c(0, 1),
  labels = c("No", "Yes")
)

## 2. 对指定的分类变量做 dummies（不动 Exited） ----
cat_cols <- c(
  "Gender", "EducationLevel", "LoanStatus", "Geography",
  "HasCrCard", "IsActiveMember", "CustomerSegment",
  "MaritalStatus", "SavingsAccountFlag"
)

## 如果这里报某些列不存在，就从 cat_cols 里删掉那些名字。
mydata <- fastDummies::dummy_cols(
  mydata,
  select_columns         = cat_cols,
  remove_first_dummy     = TRUE,
  remove_selected_columns = TRUE
)

## 确保列名合法
names(mydata) <- make.names(names(mydata))

## 3. 划分 7000 / 3000 ----
train_all <- mydata[1:7000, ]      # 用来训练和验证（有 Exited）
holdout   <- mydata[7001:10000, ]  # 用来生成 Kaggle 提交（也暂时带 Exited，但我们不用它）

## 4. 从 train_all 再划分 train2 / test2 做内部验证 ----
set.seed(123)
index  <- createDataPartition(train_all$Exited, p = 0.8, list = FALSE)
train2 <- train_all[index, ]
test2  <- train_all[-index, ]

## 5. 指标函数 + trainControl（ROSE） ----
f1_recall_summary <- function(data, lev = NULL, model = NULL) {
  # 如果这一折里只有一个类别，直接返回 NA，避免报错
  if (length(unique(data$obs)) < 2) {
    return(c(F1 = NA, Recall = NA, Precision = NA))
  }
  precision <- Precision(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  recall    <- Recall(   y_true = data$obs, y_pred = data$pred, positive = "Yes")
  f1        <- F1_Score( y_true = data$obs, y_pred = data$pred, positive = "Yes")
  c(F1 = f1, Recall = recall, Precision = precision)
}

ctrl_boot_auc <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = f1_recall_summary,
  sampling        = "rose" 
)

## 6. 在 train2 上调参 ----
set.seed(123)
fit_xgb_rose <- train(
  Exited ~ .,
  data       = train2,
  method     = "xgbTree",
  trControl  = ctrl_boot_auc,
  metric     = "F1",
  preProcess = c("scale"),
  verbosity  = 0
)

## 看一下每组参数的 F1
print(fit_xgb_rose$results[, c("eta","max_depth","nrounds","F1","Recall","Precision")])

## 7. 在 test2 上自动找最佳阈值 ----
test2_prob <- predict(fit_xgb_rose, newdata = test2, type = "prob")[, "Yes"]
y_true     <- test2$Exited

thr_seq <- seq(0.05, 0.5, by = 0.005)
f1_vec  <- sapply(thr_seq, function(th) {
  y_pred <- ifelse(test2_prob > th, "Yes", "No")
  y_pred <- factor(y_pred, levels = c("No","Yes"))
  F1_Score(y_true = y_true, y_pred = y_pred, positive = "Yes")
})

best_idx <- which.max(f1_vec)
best_thr <- thr_seq[best_idx]
best_thr   # 看看最优阈值是多少

## 8. 用 best_thr 在 train2 / test2 上算一次指标 ----
train2_prob <- predict(fit_xgb_rose, newdata = train2, type = "prob")[, "Yes"]
test2_prob  <- predict(fit_xgb_rose, newdata = test2,  type = "prob")[, "Yes"]

train2_pred <- factor(ifelse(train2_prob > best_thr, "Yes", "No"), levels = c("No","Yes"))
test2_pred  <- factor(ifelse(test2_prob  > best_thr, "Yes", "No"), levels = c("No","Yes"))

conf_train2 <- confusionMatrix(train2_pred, train2$Exited, positive = "Yes")
conf_test2  <- confusionMatrix(test2_pred,  test2$Exited,  positive = "Yes")

f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  2 * (precision * recall) / (precision + recall)
}

kpis <- data.frame(
  Dataset            = c("Train2", "Test2"),
  Error_rate         = c(1 - conf_train2$overall["Accuracy"],
                         1 - conf_test2$overall["Accuracy"]),
  Accuracy           = c(conf_train2$overall["Accuracy"],
                         conf_test2$overall["Accuracy"]),
  Precision          = c(conf_train2$byClass["Pos Pred Value"],
                         conf_test2$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train2$byClass["Sensitivity"],
                         conf_test2$byClass["Sensitivity"]),
  Specificity        = c(conf_train2$byClass["Specificity"],
                         conf_test2$byClass["Specificity"]),
  F1_Score           = c(f1_score(conf_train2), f1_score(conf_test2))
)

print(kpis)

## 9. 用全部 train_all 重新训练最终模型 ----
set.seed(123)
fit_final_xgb_rose <- train(
  Exited ~ .,
  data       = train_all,
  method     = "xgbTree",
  trControl  = ctrl_boot_auc,
  metric     = "F1",
  preProcess = c("scale"),
  verbosity  = 0
)

## 10. 在 holdout 上预测 + 生成提交 ----
pred_holdout_prob  <- predict(fit_final_xgb_rose, newdata = holdout, type = "prob")[, "Yes"]
pred_holdout_class <- ifelse(pred_holdout_prob > best_thr, "Yes", "No")

submission <- data.frame(
  ID     = holdout$ID,          # 确保 data_reducida_plus 里有 ID 这一列
  Exited = pred_holdout_class
)

write.csv(submission, "xgboost_ROSE_noFAMD.csv", row.names = FALSE)
