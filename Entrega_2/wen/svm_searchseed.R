library(e1071)
library(mlbench)
library(ggplot2)
library(ISLR)
library(caret)
mydata<-data_reducida
#dummifico data reducido
x<-mydata[,-3] #quito la respuesta
x<-x[,1:4] # cojo solo las cat
x <- fastDummies::dummy_cols(x, 
                             remove_first_dummy = TRUE,  
                             remove_selected_columns = TRUE)
x<-cbind(x,mydata[,6:7]) # adjunto las numericas
x$Exited<-mydata$Exited # añado la respuesta
mydata<-x
mydata$hasB<-ifelse(mydata$Balance==0,0,1)
mydata$Balance<-NULL

# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("0","1"),
                       labels = c("No","Yes"))

# PARTICION TRAIN2/TEST2
semillas <- sample(1:1000, 100)
f1_train <- numeric(length(semillas))
f1_test  <- numeric(length(semillas))

for (i in seq_along(semillas)) {
  
  set.seed(semillas[i])
  index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
  train2 <- train[index, ] # train interno
  test2  <- train[-index, ] # test interno
  
  svm.model <- svm(Exited ~ ., data = train2, cost = 10, kernel="radial", 
                   gamma = 0.02,scale = TRUE,,probability=TRUE,class.weights = c("No"=1, "Yes"=3))
  
  train_pred_prob <- predict(svm.model, newdata = train2, probability = TRUE)
  test_pred_prob  <- predict(svm.model, newdata = test2,  probability = TRUE)
  train_values <- attr(train_pred_prob, "probabilities")[, "Yes"]
  test_values <- attr(test_pred_prob, "probabilities")[, "Yes"]
  ###search best p
  probs <- predict(svm.model, test2, probability = TRUE)
  p1 <- attr(probs, "probabilities")[, "Yes"]
  
  thresholds <- seq(0.1, 0.5, by = 0.02)
  
  library(MLmetrics)
  
  f1s <- sapply(thresholds, function(t){
    preds <- ifelse(p1 > t, "Yes", "No")
    F1_Score(test2$Exited, preds, positive = "Yes")
  })
  
  best_t <- thresholds[which.max(f1s)]
  ####################
  
  train_pred_cut <- ifelse(train_values > best_t, "Yes", "No")
  test_pred_cut  <- ifelse(test_values > best_t, "Yes", "No")
  # Pasamos a clase: yes/no
  train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))
  test_pred_cut  <- factor(test_pred_cut,  levels = c("No","Yes"))
  
  # Matrices de confusión
  conf_train <- confusionMatrix(train_pred_cut, train2$Exited,positive = "Yes")
  conf_test  <- confusionMatrix(test_pred_cut, test2$Exited,positive = "Yes")
  
  # F1-score function
  f1_score <- function(cm){
    precision <- cm$byClass["Precision"]
    recall    <- cm$byClass["Sensitivity"]
    f1 <- 2 * (precision * recall) / (precision + recall)
    return(as.numeric(f1))
  }
  f1_train[i] <- f1_score(conf_train)
  f1_test[i]  <- f1_score(conf_test)
  
  cat("Semilla:", semillas[i],
      "| F1 Train:", round(f1_train[i], 4),
      "| F1 Test:", round(f1_test[i], 4), "\n")

}  
