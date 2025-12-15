library(caret)
library(DAAG)
library(mlbench)
library(pROC)
library(printr)
library(randomForest)
library(ranger)

mydata<-data_reducida
#########
# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("0","1"),  
                       labels = c("No","Yes")) 
# PARTICION TRAIN2/TEST2
semillas <- sample(1:1000, 50)
f1_train <- numeric(length(semillas))
f1_test  <- numeric(length(semillas))

for (i in seq_along(semillas)) {
  
  set.seed(semillas[i])
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

mtry.class <- sqrt(ncol(train2) - 1)
tuneGrid <- data.frame(mtry = 5)
rf.caret <- train(Exited ~ ., data = train2,method = "rf",
                  tuneGrid = tuneGrid,nodesize=20,maxnodes=45)

ptest <- predict(rf.caret, test2, type = 'prob')
ptrain <- predict(rf.caret, train2, type = 'prob')

# Probabilidades de la clase positiva
probs <- ptest[, "Yes"]   # cambia "1" si tu clase positiva tiene otro nombre
y_true <- test2$Exited      # variable real (ajusta el nombre)
f1_score <- function(y_true, y_pred, positive = "Yes") {
  tp <- sum(y_true == positive & y_pred == positive)
  fp <- sum(y_true != positive & y_pred == positive)
  fn <- sum(y_true == positive & y_pred != positive)
  
  if ((2 * tp + fp + fn) == 0) {
    return(0)
  }
  
  2 * tp / (2 * tp + fp + fn)
}

ps <- seq(0.01, 0.99, by = 0.01)

best_p <- 0
best_f1 <- 0

for (p in ps) {
  y_pred <- ifelse(probs >= p, "Yes", "No")
  f1 <- f1_score(y_true, y_pred)
  
  if (f1 > best_f1) {
    best_f1 <- f1
    best_p <- p
  }
}

ptrain <- ifelse(ptrain[,2] > best_p, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > best_p, "Yes", "No")
ptest <- factor(ptest, levels = c("No", "Yes"))

conf_train<-confusionMatrix(ptrain, train2$Exited, positive="Yes")
conf_test<-confusionMatrix(ptest, test2$Exited, positive="Yes")

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

