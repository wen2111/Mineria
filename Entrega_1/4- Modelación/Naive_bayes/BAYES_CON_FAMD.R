# Bayes
# librerias
library("caret")
library("naivebayes")
library("ggplot2")
library(e1071)
library(klaR)
library(dplyr)
library(FactoMineR)
library(factoextra)
load("dataaaaaaaaaaaaaa.RData")
load("data_imputado.RData") # SUBMISSION ID

#DATA 
data<-data_reducida
data <- data %>% select(Exited, everything()) # poner exited en la primera columna
data$ID <- data_imputado$ID # necesaria en test kaggle para hacer el submission, luego se quita de train


# Extraccion de train i test(kaggle)
train <- subset(data, group == "train") # 7000 obs
test <- subset(data,group == "test") # 3000 obs variable respuesta vacia

vars_drop <- c("group", "ID")
train <- train[, !(names(train) %in% vars_drop)]

# levels para "exited" porque lo exige bayes
train$Exited <- factor(train$Exited,
                                levels = c("1","0"),
                                labels = c("Yes","No"))

# FAMD

train_famd <- FAMD(train[, !names(train) %in% "Exited"],ncp = 25,
                   graph = FALSE)
train_famd_coord <- as.data.frame(train_famd$ind$coord)
train_famd_coord$Exited <- train$Exited

test_famd <- FAMD(test[, !names(test) %in% "Exited"],ncp=25,
                  graph = FALSE)
test_famd_coord <- as.data.frame(test_famd$ind$coord)
test_famd_coord$Exited <- test$Exited

train<-train_famd_coord
test<-test_famd_coord
# get_eigenvalue(train_famd)
# TRAIN I TEST NUESTRO

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno


# MODELADO

control <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        repeats = 10, 
                        verboseIter = FALSE,
                        sampling = "up")

hiperparametros <- data.frame(usekernel = FALSE, fL = 0, adjust=0)
# tener cuidado con la x, hay que tener en cuenta el rango de datos de train2
#set.seed(123)
mod <- train(y=train2$Exited, x= train2[,c(1:8)],
              data = train2, 
              method = "nb", 
              tuneGrid = hiperparametros, 
              metric = "Accuracy",	
              trControl = control)
# Predicciones
train_pred <- predict(mod, train2, type = "raw")
test_pred  <- predict(mod, test2, type = "raw")

# Matrices de confusión
conf_train <- confusionMatrix(train_pred, train2$Exited)
conf_test  <- confusionMatrix(test_pred, test2$Exited)

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

###### CONCLUSION: F1-Score<0.5 pero es de lo mas alto que hemos conseguido

# Predicciones sobre el test de Kaggle

# Entrenamiento con todo el train
mod_kaggle <- train(
  y = train$Exited,
  x = train[, 2:7],  # todas las variables predictoras
  data = train,
  method = "nb",
  tuneGrid = hiperparametros,
  metric = "Accuracy",
  trControl = control
)

pred_kaggle <- predict(mod_kaggle, newdata = test, type = "raw")

submission <- data.frame(ID = test$ID, Exited = pred_kaggle)
write.csv(submission, "submission.csv", row.names = FALSE)


