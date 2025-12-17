library(tidyverse)
library(lightgbm)
library(xgboost)
library(vip)
library(caret)

### data
df <- data_reducida
str(df)
summary(df)
summary(df$Exited)
df <- df[,-8]


#### feat engineering #####

# df <- df %>%
#   mutate(Age = case_when(
#     Age >= 18 & Age <= 37 ~ "18-37",
#     Age >= 38 & Age <= 57 ~ "38-57",
#     Age >= 58 & Age <= 77 ~ "58-77",
#     Age >= 78 & Age <= 97 ~ "78-97",
#     TRUE ~ as.character(NA) 
#   ))

# df$Age <- factor(df$Age, levels = c("18-37", "38-57", "58-77", "78-97"))
## as factor
df <- df %>% 
  mutate_if(is.character, as.factor)

## receta con dummies, normalizacion y eliminar var con varianza 0
library(recipes)
rec <- recipe(Exited ~ ., data = df) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors())

set.seed(689)
# SEPARAR TRAIN Y TEST
train <- df[1:7000, ]
test  <- df[7001:10000, ]

# LABELS PARA EXITED
train$Exited <- factor(train$Exited, levels = c("0","1")) 
test$Exited  <- factor(test$Exited,  levels = c("0","1"), labels = c("No","Yes")) 

# PARTICION TRAIN2/TEST2
set.seed(689)
index <- createDataPartition(train$Exited, p = 0.8, list = FALSE)
train2 <- train[index, ]
test2  <- train[-index, ]


# XGBoost
x_train <- model.matrix(Exited ~ ., data = train2)[, -1] 
y_train <- ifelse(train2$Exited == "Yes", 1, 0)

library(MLmetrics)

f1_recall_summary <- function(data, lev = NULL, model = NULL) {
  precision <- Precision(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  recall <- Recall(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  f1 <- F1_Score(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  c(F1 = f1, Recall = recall, Precision = precision)
}

xgb_model <- xgboost(data = x_train,
                     label = y_train,
                     objective = "binary:logistic",
                     nrounds = 300,
                     eta = 0.05,
                     max_depth = 6)

# 2. XGBoost grid
xgb_grid <- expand.grid(
  nrounds = 150,
  max_depth = c(2, 3, 5),
  eta = c(0.1, 0.4),
  gamma = c(3, 5, 6),             
  colsample_bytree = c(0.8, 0.7),
  min_child_weight = 1,
  subsample = c(0.7, 0.8)
)


ctrl_boot_auc <- trainControl(method = "cv", 
                              number = 5,         
                              classProbs = TRUE,
                              summaryFunction = f1_recall_summary
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
#   nrounds  max_depth    eta     gamma   colsample_bytree  min_child_weight subsample
#    65      150          5 0.4     5              0.7                1       0.7

train_pred_prob <- predict(fit_tuning, newdata = train2, type = "prob")
test_pred_prob  <- predict(fit_tuning, newdata = test2,  type = "prob")

train_pred_cut <- ifelse(train_pred_prob$Yes > 0.2071429, "Yes", "No")
test_pred_cut  <- ifelse(test_pred_prob$Yes > 0.2071429, "Yes", "No")

train_pred_cut <- factor(train_pred_cut, levels = c("No", "Yes"))
test_pred_cut  <- factor(test_pred_cut, levels = c("No", "Yes"))

# cm
conf_train <- confusionMatrix(train_pred_cut, train2$Exited, positive = "Yes")
conf_test  <- confusionMatrix(test_pred_cut, test2$Exited, positive = "Yes")

# F1
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
  Error_rate = c(1 - conf_train$overall["Accuracy"], 1 - conf_test$overall["Accuracy"]),
  Accuracy = c(conf_train$overall["Accuracy"], conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Pos Pred Value"], conf_test$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"], conf_test$byClass["Sensitivity"]),
  Specificity = c(conf_train$byClass["Specificity"], conf_test$byClass["Specificity"]),
  F1_Score = c(f1_train, f1_test)
)
kpis







# modelo
xgb_grid_no_tuning <- expand.grid(
  nrounds = 150,
  max_depth = 5,
  eta = 0.4,
  gamma = 5,              
  colsample_bytree = 0.7,
  min_child_weight = 1,
  subsample = 0.7
)
set.seed(689)
fit_final_glm <- train(
  Exited ~ ., 
  data = train, 
  method = "xgbTree",
  trControl = ctrl_boot_auc, 
  metric = "F1",
  preProcess = c("scale"),
  verbosity = 0,
  tuneGrid = xgb_grid_no_tuning
)


train_pred_prob <- predict(fit_final_glm, newdata = train, type = "prob")
train_pred_cut <- ifelse(train_pred_prob$Yes > 0.2071429, "Yes", "No")
train_pred_cut <- factor(train_pred_cut, levels = c("No", "Yes"))
conf_train <- confusionMatrix(train_pred_cut, train$Exited, positive = "Yes")
f1_score(conf_train)

# resultado
pred_kaggle_prob <- predict(fit_final_glm, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > 0.2071429, "Yes", "No")
test$ID <- data$ID[7001:10000]
submission <- data.frame(ID = test$ID, Exited = pred_kaggle_class)
write.csv(submission, "xgboost_tuning3_kaggle.csv", row.names = FALSE)
