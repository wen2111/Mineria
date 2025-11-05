#################################
######### NAIVE BAYES ###########
#########    FAMD    ############
#################################

# librerias
library("FactoMineR")
library("factoextra")
library("caret")
library("naivebayes")
library("reshape")
library("ggplot2")
library(e1071)
library(klaR)
library(naniar)


load("dataaaaaaaaaaaaaa.RData")

# FAMD para data reducida

######################### preparacion ##########################################

train_reducida <- subset(data_reducida, group == "train") # 7000 obs
test_reducida  <- subset(data_reducida,
                         group == "test") # 3000 obs variable respuesta vacia

vars_drop <- c("ID", "Surname", "group")
train_reducida <- train_reducida[, !(names(train_reducida) %in% vars_drop)]
test_reducida  <- test_reducida[,  !(names(test_reducida) %in% vars_drop)]

train_reducida$Exited <- factor(train_reducida$Exited,
                                levels = c("1","0"),
                                labels = c("Yes","No"))

######################## particion train2/test2 ################################

set.seed(123)
index <- createDataPartition(train_reducida$Exited, p = 0.7, list = FALSE)
train_reducida2 <- train_reducida[index, ] # train interno
test_reducida2  <- train_reducida[-index, ] # test interno

############################### FAMD solo con train ############################

# Esto APRENDE la transformación con train
famd_reducida <- FAMD(train_reducida2, ncp = 10, graph = FALSE)

fviz_screeplot(famd_reducida, addlabels = TRUE, ylim = c(0, 20)) +
  ggtitle("Varianza explicada por dimensión (FAMD)")

train_famd_coord <- as.data.frame(famd_reducida$ind$coord)
train_famd_coord$Exited <- train_reducida2$Exited  # Mantener la variable respuesta

# Esto APLICA la MISMA transformación a test
# Usando predict para proyectar test en las mismas dimensiones
test_famd_coord <- as.data.frame(predict.FAMD(famd_reducida, test_reducida2)$coord)
test_famd_coord$Exited <- test_reducida2$Exited

# esto es porque daba un error porque los nombre de las dims no eran iguales en train y test
names(test_famd_coord) <- c("Dim.1", "Dim.2", "Dim.3", "Dim.4", "Dim.5", 
                            "Dim.6", "Dim.7", "Dim.8", "Dim.9", "Exited")

############################### modelado #######################################

# cross-validation
control <- trainControl(method = "repeatedcv", repeats = 3)
hiperparametros <- data.frame(usekernel = FALSE, fL = 1, adjust = 0)

mod_nb_famd <- train(
  y = train_famd_coord$Exited, 
  x = train_famd_coord[, 1:9],  # Usar las primeras 9 dimensiones
  method = "nb", 
  tuneGrid = hiperparametros, 
  metric = "Accuracy",	
  trControl = control
)

# Predicciones
train_pred <- predict(mod_nb_famd, train_famd_coord, type = "raw")
test_pred <- predict(mod_nb_famd, test_famd_coord, type = "raw")

# Matrices de confusión
conf_train <- confusionMatrix(train_pred, train_famd_coord$Exited)
conf_test <- confusionMatrix(test_pred, test_famd_coord$Exited)

# F1-score
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}

f1_train <- f1_score(conf_train)
f1_test <- f1_score(conf_test)

# KPIs
resultados_famd <- data.frame(
  Dataset = c("Train", "Test"),
  Error_rate = c(1 - conf_train$overall["Accuracy"], 1 - conf_test$overall["Accuracy"]),
  Accuracy = c(conf_train$overall["Accuracy"], conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Pos Pred Value"], conf_test$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"], conf_test$byClass["Sensitivity"]),
  Specificity = c(conf_train$byClass["Specificity"], conf_test$byClass["Specificity"]),
  F1_Score = c(f1_train, f1_test)
)
resultados_famd


################## Predicciones sobre el test de Kaggle ########################


################## Predicciones sobre el test de Kaggle ########################

# 1. Preparar los datos - REMOVER Exited del test set
test_reducida_sin_exited <- test_reducida[, !names(test_reducida) %in% c("Exited")]

# 2. Aplicar FAMD a TODO el conjunto de entrenamiento (train_reducida)
famd_completo <- FAMD(train_reducida, ncp = 10, graph = FALSE)

# 3. Obtener coordenadas del train completo
train_famd_completo <- as.data.frame(famd_completo$ind$coord)
train_famd_completo$Exited <- train_reducida$Exited

# Corregir nombres (si es necesario)
names(train_famd_completo) <- c("Dim.1", "Dim.2", "Dim.3", "Dim.4", "Dim.5", 
                                "Dim.6", "Dim.7", "Dim.8", "Dim.9", "Dim.10", "Exited")

# 4. Aplicar la MISMA transformación FAMD al test de Kaggle (SIN la variable Exited)
test_famd_kaggle <- as.data.frame(predict.FAMD(famd_completo, test_reducida_sin_exited)$coord)

# Corregir nombres del test
names(test_famd_kaggle) <- c("Dim.1", "Dim.2", "Dim.3", "Dim.4", "Dim.5", 
                             "Dim.6", "Dim.7", "Dim.8", "Dim.9", "Dim.10")

# 5. Entrenar el modelo con TODO el train (FAMD version)
mod_kaggle_famd <- train(
  y = train_famd_completo$Exited,
  x = train_famd_completo[, 1:9],  # Usar las 9 dimensiones del FAMD
  method = "nb",
  tuneGrid = hiperparametros,
  metric = "Accuracy",
  trControl = control
)

# 6. Hacer predicciones sobre el test de Kaggle (transformado con FAMD)
pred_kaggle <- predict(mod_kaggle_famd, newdata = test_famd_kaggle[, 1:9], type = "raw")

# 7. Crear el archivo de submission
submission <- data.frame(
  ID = test_reducida$ID, 
  Exited = pred_kaggle
)

# Verificar las primeras filas
head(submission)
table(submission$Exited)  # Ver distribución de las predicciones

# 8. Guardar el archivo
write.csv(submission, "submission_famd_nb.csv", row.names = FALSE)



