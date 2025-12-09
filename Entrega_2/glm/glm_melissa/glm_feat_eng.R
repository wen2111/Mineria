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
test <- test[, !names(test) %in% c("group", "Surname", "ID")]

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
train$Exited_Num <- ifelse(train$Exited == "Yes", 1, 0)

# 2. TABLA RESUMEN: Calculamos la tasa de Churn por Grupo
interaction_data <- train %>%
  group_by(Geography, Gender) %>%
  summarise(
    Total_Clientes = n(),
    Clientes_Perdidos = sum(Exited_Num),
    Tasa_Churn = mean(Exited_Num) # Esto nos da el % (ej: 0.32)
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

################## PARTICION ###################################################
set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] 
test2  <- train[-index, ] 

set.seed(123)
mod_glm <- glm(Exited ~ . + I(Age^2) + Geography:Gender + HasBalance,
               data = train2, family = "binomial")

# Predecimos
probs_test <- predict(mod_glm, newdata = test2, type = "response")

# --- VISUALIZACIÓN predicciones ---
# Ahora el histograma debería tener las barras altas a la IZQUIERDA (cerca del 0)
# porque la probabilidad de irse (Yes) es baja para la mayoría.
hist(probs_test, 
     main = "Distribución de Probabilidades de CHURN (Yes)", 
     col = "salmon", 
     xlim = c(0,1),
     xlab = "Probabilidad de abandono")

################## PREDICCIONES TRAIN Y TEST ###################################
probs_train <- predict(mod_glm, newdata = train2, type = "response")
probs_test  <- predict(mod_glm, newdata = test2, type = "response")

################## APLICACION UMBRAL MANUAL ####################################
umbral_manual <- 0.2

pred_train_glm <- factor(ifelse(probs_train > umbral_manual, "Yes", "No"), 
                         levels = c("No", "Yes"))
pred_test_glm  <- factor(ifelse(probs_test > umbral_manual, "Yes", "No"), 
                         levels = c("No", "Yes"))

################## MATRICES DE CONFUSION #######################################
conf_train <- caret::confusionMatrix(pred_train_glm, train2$Exited, 
                                     positive = "Yes", mode = "prec_recall")
conf_test  <- caret::confusionMatrix(pred_test_glm, test2$Exited, 
                                     positive = "Yes", mode = "prec_recall")

################## DATAFRAME KPIS ##############################################
resultados_glm <- data.frame(
  Dataset = c("Train", "Test"),
  Accuracy = c(conf_train$overall["Accuracy"], conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Precision"], conf_test$byClass["Precision"]),
  Specificity = c(conf_train$byClass["Specificity"], conf_test$byClass["Specificity"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"], conf_test$byClass["Sensitivity"]),
  F1_Score = c(conf_train$byClass["F1"], conf_test$byClass["F1"])
)

print(resultados_glm)
