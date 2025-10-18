#################################
######### NAIVE BAYES ###########
#################################

# librerias
library("caret")
library("naivebayes")
library("reshape")
library("ggplot2")
library(e1071)

load("dataaaaaaaaaaaaaa.RData")

# Pasos:
# 1) subset train/test de imputado, transformada y reducida
# 2) particion train en train2/test2
# 3) modelado data desbalanceada
# 3) gestionar desbalanceo
# 4) modelado data balanceada

#_______________________________________________________________________________
# DATA IMPUTADO
#_______________________________________________________________________________

# subset y preparacion
train_imputado <- subset(data_imputado, group == "train")
test_imputado  <- subset(data_imputado, group == "test")

vars_drop <- c("ID", "Surname", "group")
train_imputado <- train_imputado[, !(names(train_imputado) %in% vars_drop)]
test_imputado  <- test_imputado[,  !(names(test_imputado) %in% vars_drop)]

# levels para "exited"
train_imputado$Exited <- factor(train_imputado$Exited,
                                levels = c("0","1"),
                                labels = c("No","Yes"))
test_imputado$Exited  <- factor(test_imputado$Exited,
                                levels = c("0","1"),
                                labels = c("No","Yes"))

# particion
set.seed(123)
index <- createDataPartition(train_imputado$Exited, p = 0.7, list = FALSE)
train_imputado2 <- train_imputado[index, ]
test_imputado2  <- train_imputado[-index, ]

# modelo data_imputado sin balancear
set.seed(123)
mod7 <- train(
  Exited ~ .,                 
  data = train_imputado2,
  method = "nb",
  metric = "Accuracy",
  trControl = trainControl(
    classProbs = TRUE,
    method = "cv",
    number = 10
  )
)

# predicciones
pred_train <- predict(mod7, train_model_data)
pred_test  <- predict(mod7, test_model_data)

# confusion matrix
conf_train <- confusionMatrix(pred_train, train_model_data$Exited, positive="1")
conf_test  <- confusionMatrix(pred_test,  test_model_data$Exited,  positive="1")

conf_train
conf_test

# F1-score
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(f1)
}

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)

f1_train
f1_test


#_______________________________________________________________________________
# DATA TRANSFORMADA
#_______________________________________________________________________________

data_transformada$Exited <- factor(data_transformada$Exited,
                                   levels = c("0","1"),
                                   labels = c("No","Yes"))

# subset
train_transformada <- subset(data_transformada, group == "train")
test_transformada  <- subset(data_transformada, group == "test")


#_______________________________________________________________________________
# DATA REDUCIDA
#_______________________________________________________________________________

# añadir "group" a data_reducida
data_reducida$group <- data_imputado$group
identical(rownames(data_reducida), rownames(data_imputado)) # comprobar que coinciden las filas

data_reducida$Exited <- factor(data_reducida$Exited,
                               levels = c("0","1"),
                               labels = c("No","Yes"))

# subset
train_reducida <- subset(data_reducida, group == "train")
test_reducida  <- subset(data_reducida, group == "test")
