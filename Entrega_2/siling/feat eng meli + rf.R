library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)

load("data_reducida_con_ID.RData")

mydata <- data_reducida

################## PREPARACION ########################
train <- subset(mydata, group == "train") 
test  <- subset(mydata, group == "test") 

variables_eliminar <- c("group", "Surname", "ID")
train <- train[, !names(train) %in% variables_eliminar]
test_submission_id <- test$ID 
test <- test[, !names(test) %in% variables_eliminar]
train$Exited <- factor(train$Exited, levels = c("0","1"), labels = c("No","Yes"))

################### FEAT. ENGINEERING ####################

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

# rf model con caret
set.seed(123)
mod_rf<- train(Exited ~ . + I(Age^2) + Geography:Gender + HasBalance,
                      data = train2, 
                      method = "rf", 
                      trControl = trainControl(method = "cv", number = 10),
                      tuneLength = 5)

print(mod_rf$finalModel)

# Predecimos
probs_test <- predict(mod_rf, newdata = test2, type = "prob")[,2] # prob de "Yes"

# --- VISUALIZACIÓN predicciones ---
# Ahora el histograma debería tener las barras altas a la IZQUIERDA (cerca del 0)
# porque la probabilidad de irse (Yes) es baja para la mayoría.
hist(probs_test, 
     main = "Distribución de Probabilidades de CHURN (Yes)", 
     col = "salmon", 
     xlim = c(0,1),
     xlab = "Probabilidad de abandono")

################## PREDICCIONES TRAIN Y TEST ###################################
probs_train_rf <- predict(mod_rf, newdata = train2, type = "prob")[, "Yes"]
probs_test_rf  <- predict(mod_rf, newdata = test2, type = "prob")[, "Yes"]
################## APLICACION UMBRAL MANUAL ####################################
umbral_manual <- 0.2

pred_train_rf <- factor(ifelse(probs_train_rf > umbral_manual, "Yes", "No"), 
                         levels = c("Yes", "No"))
pred_test_rf <- factor(ifelse(probs_test_rf > umbral_manual, "Yes", "No"), 
                         levels = c("Yes", "No"))

################## MATRICES DE CONFUSION #######################################
conf_train_rf <- caret::confusionMatrix(pred_train_rf, train2$Exited, 
                                     positive = "Yes", mode = "prec_recall")
conf_test_rf  <- caret::confusionMatrix(pred_test_rf, test2$Exited, 
                                     positive = "Yes", mode = "prec_recall")

################## DATAFRAME KPIS ##############################################
resultados_rf <- data.frame(
  Dataset = c("Train", "Test"),
  Accuracy = c(conf_train_rf$overall["Accuracy"], conf_test_rf$overall["Accuracy"]),
  Precision = c(conf_train_rf$byClass["Precision"], conf_test_rf$byClass["Precision"]),
  Specificity = c(conf_train_rf$byClass["Specificity"], conf_test_rf$byClass["Specificity"]),
  Recall_Sensitivity = c(conf_train_rf$byClass["Sensitivity"], conf_test_rf$byClass["Sensitivity"]),
  F1_Score = c(conf_train_rf$byClass["F1"], conf_test_rf$byClass["F1"])
)

print(resultados_rf)



# diferentes umbrales
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
