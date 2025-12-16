library(e1071)
library(mlbench)
library(ggplot2)
library(ISLR)


###########################################3
mydata <- data_transformada
vars <- c(
  "Age",
  "EstimatedSalary",
  "AvgTransactionAmount",
  "CreditScore",
  "DigitalEngagementScore",
  "Balance",
  "NumOfProducts_grupo",
  "TransactionFrequency",
  "Tenure",
  "NetPromoterScore",
  "Geography",
  "Gender",
  "IsActiveMember",
  "Exited"
)
mydata<-mydata[,vars]
#################################################3
library(fastDummies)

# columnas que quieres transformar
cols_to_dummy <- names(mydata)[c(7, 11, 12)]

# crear dummies
df_dummy <- dummy_cols(
  mydata,
  select_columns = cols_to_dummy,       # columnas específicas
  remove_first_dummy = TRUE,            # elimina la primera dummy (evita multicolinealidad)
  remove_selected_columns = TRUE        # elimina las columnas originales
)

mydata<-df_dummy
#####################################################
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
set.seed(666)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

####### (Añadido) smote
set.seed(666)
train_control <- trainControl(
  method = "cv",
  number = 10,
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)
svm.model <- train(Exited ~ ., data = train2,
                   method = "svmRadial",
                   trControl = train_control,
                   metric = "ROC",
                   probability = TRUE) 


print(svm.model)

####### (añadido) Smote con library(DMwR)
# library(DWmR)
# train_smote <- SMOTE(Exited ~ ., data = train2, perc.over = 150, perc.under = 100)
######################################################

# svm.model <- svm(Exited ~ ., data = train2, cost = 10, kernel="radial", 
#                 gamma = 0.02,scale = TRUE,,probability=TRUE,class.weights = c("No"=1, "Yes"=3))

train_pred_prob <- predict(svm.model, newdata = train2, probability = TRUE)
test_pred_prob  <- predict(svm.model, newdata = test2, probability = TRUE)
train_values <- attr(train_pred_prob, "probabilities")[, "Yes"]
test_values <- attr(test_pred_prob, "probabilities")[, "Yes"]

############3
probs <- predict(svm.model, test2, probability = TRUE)
p1 <- attr(probs, "probabilities")[, "Yes"]

thresholds <- seq(0.1, 0.5, by = 0.02)

library(MLmetrics)

f1s <- sapply(thresholds, function(t){
  preds <- ifelse(p1 > t, "Yes", "No")
  F1_Score(test2$Exited, preds, positive = "Yes")
})
max(f1s)

best_t <- thresholds[which.max(f1s)]
best_t
####################

train_pred_cut <- ifelse(train_values > best_t, "Yes", "No")
test_pred_cut  <- ifelse(test_values > best_t, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))
test_pred_cut  <- factor(test_pred_cut,  levels = c("No","Yes"))


length(train_pred_cut)  # should match length of train2$Exited
length(train2$Exited)

length(test_pred_cut)  # should match length of test2$Exited
length(test2$Exited)


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


f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)


kpis <- data.frame(
  Dataset = c("Train2", "Test2"),
  Error_rate = c(1 - conf_train$overall["Accuracy"],
                 1 - conf_test$overall["Accuracy"]),
  Accuracy = c(conf_train$overall["Accuracy"],
               conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Pos Pred Value"],
                conf_test$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"],
                         conf_test$byClass["Sensitivity"]),
  Specificity = c(conf_train$byClass["Specificity"],
                  conf_test$byClass["Specificity"]),
  F1_Score = c(f1_train, f1_test)
)

kpis















######################################## final

svm.model <- svm(Exited ~ ., data = train, cost = 10, kernel="radial", 
                 gamma = 0.02,scale = TRUE,probability=TRUE,class.weights = c("No"=1, "Yes"=3))

train_pred_prob <- predict(svm.model, newdata = train, probability = TRUE)
train_values <- attr(train_pred_prob, "probabilities")[, "Yes"]

############
probs <- predict(svm.model, train, probability = TRUE)
p1 <- attr(probs, "probabilities")[, "Yes"]

thresholds <- seq(0.1, 0.5, by = 0.02)

library(MLmetrics)

f1s <- sapply(thresholds, function(t){
  preds <- ifelse(p1 > t, "Yes", "No")
  F1_Score(train$Exited, preds, positive = "Yes")
})

best_t <- thresholds[which.max(f1s)]
best_t
max(f1s)
####################
train_pred_cut <- ifelse(train_values > 0.24, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))
# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train$Exited,positive = "Yes")
(f1_train <- f1_score(conf_train))

#pred
kaggle<-predict(svm.model,test[,-13],probability = TRUE)
kaggle<-attr(kaggle, "probabilities")[, "Yes"]
kaggle <- ifelse(kaggle > 0.26, "Yes", "No")
test$ID<-data$ID[7001:10000]
submission <- data.frame(ID = test$ID, Exited = kaggle)
write.csv(submission, "svm2.csv", row.names = FALSE)
prop.table(table(submission$Exited))
