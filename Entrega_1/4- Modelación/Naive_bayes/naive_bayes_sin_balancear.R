#################################
######### NAIVE BAYES ###########
######## SIN BALANCEAR ##########
#################################

# librerias
library("caret")
library("naivebayes")
library("reshape")
library("ggplot2")
library(e1071)
library(klaR)
library(naniar)

load("dataaaaaaaaaaaaaa.RData") # <- contiene imputada, transformada y reducida

# Pasos:
# 1) subset train/test
# 2) particion train en train2/test2
# 3) modelado data desbalanceada

#_______________________________________________________________________________
# DATA IMPUTADO
#_______________________________________________________________________________

# subset y preparacion
train_imputado <- subset(data_imputado, group == "train") # 7000 obs
test_imputado  <- subset(data_imputado,
                         group == "test") # 3000 obs variable respuesta vacia

vars_drop <- c("ID", "Surname", "group")
train_imputado <- train_imputado[, !(names(train_imputado) %in% vars_drop)]
test_imputado  <- test_imputado[,  !(names(test_imputado) %in% vars_drop)]

# levels para "exited"
train_imputado$Exited <- factor(train_imputado$Exited,
                                levels = c("1","0"),
                                labels = c("Yes","No"))

# niveles en orden correcto para la conf matrix
levels(train_imputado$Exited)
# Debería mostrar: [1] "Yes" "No"


# particion
set.seed(123)
index <- createDataPartition(train_imputado$Exited, p = 0.7, list = FALSE)
train_imputado2 <- train_imputado[index, ] # train interno
test_imputado2  <- train_imputado[-index, ] # test interno

# verifico no hay missings por si a caso
gg_miss_var(train_imputado2)
gg_miss_var(test_imputado2)

# verifico coinciden proporciones de "Exited"
prop.table(table(train_imputado$Exited))
prop.table(table(train_imputado2$Exited))
prop.table(table(test_imputado2$Exited))

# MODELADO

# cross-validation: se asume Normalidad, Laplace=0
control= trainControl(method="repeatedcv",repeats=3)
hiperparametros <- data.frame(usekernel = FALSE, fL = 1, adjust=0)

mod7 <- train(y=train_imputado2$Exited, x= train_imputado2[,c(2:21)],
                   data = train_imputado2, 
                   method = "nb", 
                   tuneGrid = hiperparametros, 
                   metric = "Accuracy",	
                   trControl = control)
mod7 

# Predicciones
train_pred <- predict(mod7, train_imputado2, type = "raw")
test_pred  <- predict(mod7, test_imputado2, type = "raw")

# Matrices de confusión
conf_train <- confusionMatrix(train_pred, train_imputado2$Exited)
conf_test  <- confusionMatrix(test_pred, test_imputado2$Exited)

# F1-score
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)

f1_train
f1_test

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

###### CONCLUSION: F1-Score<0.5 descartado #######

# Predicciones sobre el test de Kaggle

# Entrenamiento con todo el train
mod_kaggle <- train(
  y = train_imputado$Exited,
  x = train_imputado[, 2:21],  # todas las variables predictoras
  data = train_imputado,
  method = "nb",
  tuneGrid = hiperparametros,
  metric = "Accuracy",
  trControl = control
)

mod_kaggle

pred_kaggle <- predict(mod_kaggle, newdata = test_imputado, type = "raw")

submission <- data.frame(ID = test_imputado$ID, Exited = pred_kaggle)
write.csv(submission, "submission.csv", row.names = FALSE)


#_______________________________________________________________________________
# DATA TRANSFORMADA
#_______________________________________________________________________________

# subset y preparacion
train_transformada <- subset(data_transformada, group == "train") # 7000 obs
test_transformada  <- subset(data_transformada,
                         group == "test") # 3000 obs variable respuesta vacia

vars_drop <- c("ID", "Surname", "group")
train_transformada <- train_transformada[, !(names(train_transformada) %in% vars_drop)]
test_transformada  <- test_transformada[,  !(names(test_transformada) %in% vars_drop)]

# levels para "exited"
train_transformada$Exited <- factor(train_transformada$Exited,
                                levels = c("1","0"),
                                labels = c("Yes","No"))

# niveles en orden correcto para la conf matrix
levels(train_transformada$Exited)
# Debería mostrar: [1] "Yes" "No"


# particion
set.seed(123)
index <- createDataPartition(train_transformada$Exited, p = 0.7, list = FALSE)
train_transformada2 <- train_transformada[index, ] # train interno
test_transformada2  <- train_transformada[-index, ] # test interno

# verifico no hay missings por si a caso
gg_miss_var(train_transformada2)
gg_miss_var(test_transformada2)

# verifico coinciden proporciones de "Exited"
prop.table(table(train_transformada$Exited))
prop.table(table(train_transformada2$Exited))
prop.table(table(test_transformada2$Exited))

# MODELADO

# cross-validation: se asume Normalidad, Laplace=0
control= trainControl(method="repeatedcv",repeats=3)
hiperparametros <- data.frame(usekernel = FALSE, fL = 1, adjust=0)

mod8 <- train(y=train_transformada2$Exited, x= train_transformada2[,c(2:21)],
              data = train_transformada2, 
              method = "nb", 
              tuneGrid = hiperparametros, 
              metric = "Accuracy",	
              trControl = control)
mod8 

# Predicciones
train_pred <- predict(mod8, train_transformada2, type = "raw")
test_pred  <- predict(mod8, test_transformada2, type = "raw")

# Matrices de confusión
conf_train <- confusionMatrix(train_pred, train_transformada2$Exited)
conf_test  <- confusionMatrix(test_pred, test_transformada2$Exited)

# F1-score
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)

f1_train
f1_test

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

###### CONCLUSION: F1-Score<0.5 descartado #######

#_______________________________________________________________________________
# DATA REDUCIDA
#_______________________________________________________________________________

# añadir "group" a data_reducida (no estaba)
data_reducida$group <- data_imputado$group
identical(rownames(data_reducida), rownames(data_imputado)) # comprobar que coinciden las filas

# subset y preparacion
train_reducida <- subset(data_reducida, group == "train") # 7000 obs
test_reducida  <- subset(data_reducida,
                             group == "test") # 3000 obs variable respuesta vacia

vars_drop <- c("ID", "Surname", "group")
train_reducida <- train_reducida[, !(names(train_reducida) %in% vars_drop)]
test_reducida  <- test_reducida[,  !(names(test_reducida) %in% vars_drop)]

# levels para "exited"
train_reducida$Exited <- factor(train_reducida$Exited,
                                    levels = c("1","0"),
                                    labels = c("Yes","No"))

# niveles en orden correcto para la conf matrix
levels(train_reducida$Exited)
# Debería mostrar: [1] "Yes" "No"


# particion
data<-da
set.seed(123)
index <- createDataPartition(train_reducida$Exited, p = 0.7, list = FALSE)
train_reducida2 <- train_reducida[index, ] # train interno
test_reducida2  <- train_reducida[-index, ] # test interno

# verifico no hay missings por si a caso
gg_miss_var(train_reducida2)
gg_miss_var(test_reducida2)

# verifico coinciden proporciones de "Exited"
prop.table(table(train_reducida$Exited))
prop.table(table(train_reducida2$Exited))
prop.table(table(test_reducida2$Exited))

# MODELADO

# cross-validation: se asume Normalidad, Laplace=0
control= trainControl(method="repeatedcv",repeats=3)
hiperparametros <- data.frame(usekernel = FALSE, fL = 1, adjust=0)

mod9 <- train(
  y = train_reducida2$Exited, 
  x = train_reducida2[, !names(train_reducida2) %in% "Exited"],  # Todas menos la variable respuesta
  method = "nb", 
  tuneGrid = hiperparametros, 
  metric = "Accuracy",	
  trControl = control
)
mod9 

# Predicciones
train_pred <- predict(mod9, train_reducida2, type = "raw")
test_pred  <- predict(mod9, test_reducida2, type = "raw")

# Matrices de confusión
conf_train <- confusionMatrix(train_pred, train_reducida2$Exited)
conf_test  <- confusionMatrix(test_pred, test_reducida2$Exited)

# F1-score
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)

f1_train
f1_test

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

###### CONCLUSION: F1-Score<0.5 descartado #######
