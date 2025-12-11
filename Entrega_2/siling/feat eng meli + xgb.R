library(caret)
library(dplyr)
library(ggplot2)


load("data_reducida_con_ID.RData")

mydata <- data_reducida

################## PREPARACION #################################################
train <- subset(mydata, group == "train") 
test  <- subset(mydata, group == "test") 

variables_eliminar <- c("group", "Surname", "ID")
train <- train[, !names(train) %in% variables_eliminar]
test_submission_id <- test$ID 
test <- test[, !names(test) %in% variables_eliminar]
train$Exited <- factor(train$Exited, levels = c("0","1"), labels = c("No","Yes"))

################### FEAT. ENGINEERING ##########################################

# variable dummy HasBalance
train$HasBalance <- factor(ifelse(train$Balance > 0, "Yes", "No"), 
                           levels = c("No", "Yes")) # Definimos el orden: No=Base, Yes=Objetivo
test$HasBalance <- factor(ifelse(test$Balance > 0, "Yes", "No"), 
                          levels = c("No", "Yes"))

# interaccion Gender:Geography

# 1. PREPARACIÓN: Usamos 'train' (7000 obs)
# Necesitamos convertir Exited a número (0 y 1) para calcular el promedio (tasa)
# Si es "Yes" vale 1, si es "No" vale 0.
train_temp <- train
train_temp$Exited_Num <- ifelse(train_temp$Exited == "Yes", 1, 0)
# 2. TABLA RESUMEN: Calculamos la tasa de Churn por Grupo
interaction_data <- train_temp %>%
  group_by(Geography, Gender) %>%
  summarise(
    Total_Clientes = n(),
    Clientes_Perdidos = sum(Exited_Num),
    Tasa_Churn = mean(Exited_Num)
  )

print(interaction_data)

# 3. VISUALIZACIÓN: HEATMAP (Mapa de Calor)
# Esto te mostrará en ROJO INTENSO dónde está el problema
ggplot(interaction_data, aes(x = Geography, y = Gender, fill = Tasa_Churn)) +
  geom_tile(color = "white") +
  geom_text(aes(label = scales::percent(Tasa_Churn, accuracy = 0.1)), 
            color = "black", size = 5, fontface = "bold") +
  scale_fill_gradient(low = "#ffcccc", high = "#cc0000") +
  labs(title = "Mapa de Calor: ¿Quién se va más?",
       fill = "Tasa de Abandono") +
  theme_minimal() +
  theme(axis.text = element_text(size = 12))
# Eliminar la variable temporal
rm(train_temp)

################## PARTICION ###################################################
set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] 
test2  <- train[-index, ] 

# xgb model con caret
set.seed(123)
library(xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(train2[, -which(names(train2) == "Exited")]), label = train2$Exited)
param <- list(max_depth = 6, eta = 0.3, objective = "binary:logistic")
mod_xgb <- xgb.train(param, dtrain, nrounds = 100)


print(mod_xgb$finalModel)

# Predecimos
probs_test <- predict(mod_xgb, newdata = test2, type = "prob")[,2] # prob de "Yes"

# --- VISUALIZACIÓN predicciones ---
# Ahora el histograma debería tener las barras altas a la IZQUIERDA (cerca del 0)
# porque la probabilidad de irse (Yes) es baja para la mayoría.
hist(probs_test, 
     main = "Distribución de Probabilidades de CHURN (Yes)", 
     col = "salmon", 
     xlim = c(0,1),
     xlab = "Probabilidad de abandono")

################## PREDICCIONES TRAIN Y TEST ###################################
probs_train_xgb <- predict(mod_xgb, newdata = train2, type = "prob")[, "Yes"]
probs_test_xgb  <- predict(mod_xgb, newdata = test2, type = "prob")[, "Yes"]
################## APLICACION UMBRAL MANUAL ####################################
umbral_manual <- 0.2

pred_train_xgb <- factor(ifelse(probs_train_xgb > umbral_manual, "Yes", "No"), 
                        levels = c("Yes", "No"))
pred_test_xgb <- factor(ifelse(probs_test_xgb > umbral_manual, "Yes", "No"), 
                       levels = c("Yes", "No"))

################## MATRICES DE CONFUSION #######################################
conf_train_xgb <- caret::confusionMatrix(pred_train_xgb, train2$Exited, 
                                        positive = "Yes", mode = "prec_recall")
conf_test_xgb  <- caret::confusionMatrix(pred_test_xgb, test2$Exited, 
                                        positive = "Yes", mode = "prec_recall")

################## DATAFRAME KPIS ##############################################
resultados_xgb <- data.frame(
  Dataset = c("Train", "Test"),
  Accuracy = c(conf_train_xgb$overall["Accuracy"], conf_test_xgb$overall["Accuracy"]),
  Precision = c(conf_train_xgb$byClass["Precision"], conf_test_xgb$byClass["Precision"]),
  Specificity = c(conf_train_xgb$byClass["Specificity"], conf_test_xgb$byClass["Specificity"]),
  Recall_Sensitivity = c(conf_train_xgb$byClass["Sensitivity"], conf_test_xgb$byClass["Sensitivity"]),
  F1_Score = c(conf_train_xgb$byClass["F1"], conf_test_xgb$byClass["F1"])
)

print(resultados_xgb)



################# con diferentes umbrales en test2 ##########################
thresholds <- seq(0.1, 0.5, by = 0.05)
results <- data.frame()

for (thresh in thresholds) {
  pred <- factor(ifelse(probs_test > thresh, "Yes", "No"), levels = c("No", "Yes"))
  conf <- confusionMatrix(pred, test2$Exited, positive = "Yes")
  
  results <- rbind(results, data.frame(
    Threshold = thresh,
    Accuracy = conf$overall["Accuracy"],
    Precision = conf$byClass["Precision"],
    Recall = conf$byClass["Sensitivity"],
    F1 = conf$byClass["F1"]
  ))
}

print(results)
