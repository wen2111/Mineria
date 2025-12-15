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
semillas <- sample(1:1000, 10)
f1_train <- numeric(length(semillas))
f1_test  <- numeric(length(semillas))

for (i in seq_along(semillas)) {
  
  set.seed(semillas[i])
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

mtry.class <- sqrt(ncol(train2) - 1)
tuneGrid <- data.frame(mtry = floor(c(mtry.class/2, mtry.class, 2*mtry.class)))
rf.caret <- train(Exited ~ ., data = train2,method = "rf",
                  tuneGrid = tuneGrid,nodesize=15,maxnodes=25)

ptest <- predict(rf.caret, test2, type = 'prob')
ptrain <- predict(rf.caret, train2, type = 'prob')

# ROC y umbral óptimo
roc_obj <- roc(test2_coord$Exited, probs_test$Yes, percent = TRUE)
coords_opt <- coords(roc_obj, "best", ret = c("threshold"), best.method = "closest.topleft")
umbral <- coords_opt$threshold

ptrain <- ifelse(ptrain[,2] > umbral, "Yes", "No")
ptrain <- factor(ptrain, levels = c("No", "Yes"))

ptest <- ifelse(ptest[,2] > umbral, "Yes", "No")
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
