#######################################################
################ Bayes sin balancear ##################
############## con punto de corte 0.2 #################
#######################################################

# librerias
library("caret")
library("naivebayes")
library("reshape")
library("ggplot2")
library(e1071)
library(klaR)
library(naniar)
library(dplyr)

#DATA
load("dataaaaaaaaaaaaaa.RData")
load("data.RData")
mydata <- data_reducida
mydata <- mydata %>% select(Exited, everything())


# Extraccion de train i test(kaggle)
train <- subset(mydata, group == "train") # 7000 obs
test <- subset(mydata,group == "test") # 3000 obs variable respuesta vacia

train$group <- NULL
test$group <- NULL
test$ID <- test_data$ID

# levels para "exited" porque lo exige bayes
train$Exited <- factor(train$Exited,
                                levels = c("1","0"),
                                labels = c("Yes","No"))

# TRAIN I TEST NUESTRO 

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno


# MODELADO

control <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        repeats = 10, 
                        verboseIter = FALSE)

hiperparametros <- data.frame(usekernel = FALSE, fL = 0, adjust=0)
# tener cuidado con la x, hay que tener en cuenta el rango de datos
set.seed(123)
mod <- train(y=train2$Exited, x= train2[,c(2:6)],
              data = train2, 
              method = "nb", 
              tuneGrid = hiperparametros, 
              metric = "Accuracy",	
              trControl = control)
# Predicciones
train_prob <- predict(mod, train2, type = "prob")
test_prob  <- predict(mod, test2, type = "prob")

# Punto de corte 0.2
threshold <- 0.2

train_pred_cut <- factor(ifelse(train_prob$Yes >= threshold, "Yes", "No"),
                         levels = c("Yes","No"))
test_pred_cut <- factor(ifelse(test_prob$Yes >= threshold, "Yes", "No"),
                        levels = c("Yes","No"))


# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train2$Exited)
conf_test  <- confusionMatrix(test_pred_cut, test2$Exited)

# F1-score
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)

# KPIs
data.frame(
  Dataset = c("Train", "Test"),
  Error_rate = c(1-conf_train$overall["Accuracy"], 1-conf_test$overall["Accuracy"]),
  Accuracy = c(conf_train$overall["Accuracy"], conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Pos Pred Value"], conf_test$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"], conf_test$byClass["Sensitivity"]),
  Specificity = c(conf_train$byClass["Specificity"], conf_test$byClass["Specificity"]),
  F1_Score = c(f1_train, f1_test)
)

# Predicciones sobre el test de Kaggle

# Entrenamiento con todo el train
mod_kaggle <- train(
  y = train$Exited,
  x = train[, 2:6],  # todas las variables predictoras
  method = "nb",
  tuneGrid = hiperparametros,
  metric = "Accuracy",
  trControl = control
)

mod_kaggle

pred_kaggle <- predict(mod_kaggle, newdata = test, type = "raw")

submission <- data.frame(ID = test$ID, Exited = pred_kaggle)
write.csv(submission, "submission.csv", row.names = FALSE)
