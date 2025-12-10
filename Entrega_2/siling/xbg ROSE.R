library(caret)
library(fastDummies)
library(ROSE)
library(xgboost)
library(pROC)
library(MLmetrics)

## 1. preprocessing ---------------------------------------------------------
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

## 3. ROSE ---------------------------------------
set.seed(123)
rose_train <- ROSE(Exited ~ ., data = train, seed = 123)$data

rose_train$Exited <- factor(rose_train$Exited,
                            levels = c("Yes","No"))

table(train$Exited)
table(rose_train$Exited)

## 4. definir traing con xgboost  ----------------------------------------
ctrl_xgb <- trainControl(
  method          = "cv",
  number          = 5, 
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter     = FALSE
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

## 5. ROSE  xgboost -------------------------------
set.seed(123)
rose_train <- ROSE(Exited ~ ., data = train, seed = 123)$data
rose_train$Exited <- factor(rose_train$Exited, levels = c("Yes","No"))

set.seed(123)
fit_xgb <- train(
  Exited ~ .,
  data      = rose_train,
  method    = "xgbTree",
  metric    = "ROC", 
  trControl = ctrl_xgb,
  tuneGrid  = tuneGrid_xgb
)

prob_test <- predict(fit_xgb, newdata = test, type = "prob")[, "Yes"]

pred_test <- factor(
  ifelse(prob_test > 0.5, "Yes", "No"),
  levels = levels(test$Exited)
)

cm_test <- confusionMatrix(pred_test, test$Exited, positive = "Yes")
cm_test

submission <- data.frame(
  ID     = data_reducida$ID[7001:10000],
  Exited = pred_test
)

write.csv(submission, "submission_xgb_rose_simple.csv", row.names = FALSE)
