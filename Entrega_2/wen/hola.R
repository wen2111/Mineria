library(caret)
library(ROSE)
library(smotefamily)
data <- data_reducida

# SEPARAR TRAIN Y TEST
train <- subset(data, group == "train") # 7000 obs
test <- subset(data,group == "test") # 3000 obs variable respuesta vacia

# ELIMINAR "GROUP"
train$group <- NULL
test$group  <- NULL

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("0","1"),
                       labels = c("No","Yes"))
table(train$Exited)

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)

train2 <- train[index,]     # Entrenamiento interno
test2  <- train[-index, ]    # Validación interna
#str(data_imputado)
# BALANCEO SOLO EN TRAIN2
train2_balanceada <- ROSE(
  Exited ~ .,
  data = train2,
  p = 0.4,      # 40% Yes, 60% No
  seed = 123
)$data

# COMPROBAR BALANCEO
(table(train2$Exited))                     # Antes del balanceo
(table(train2_balanceada$Exited))  # Después del balanceo
################################ FIN ###########################################
# MODELOS GLM

# Logit
modelo_logit <- glm(Exited ~ ., data = train2_balanceada, family = binomial(link = "logit"))

# Probit
modelo_probit <- glm(Exited ~ ., data = train2_balanceada, family = binomial(link = "probit"))

# matrix para train2
pred_logit_train2 <- predict(modelo_logit, newdata = train2_balanceada, type = "response")
pred_probit_train2 <- predict(modelo_probit, newdata = train2_balanceada, type = "response")

class_logit_train2 <- ifelse(pred_logit_train2 > 0.5, 1, 0)
class_probit_train2 <- ifelse(pred_probit_train2 > 0.5, 1, 0)

class_logit_train2 <- factor(class_logit_train2,
                       levels = c("0","1"),
                       labels = c("No","Yes"))
class_probit_train2 <- factor(class_probit_train2,
                             levels = c("0","1"),
                             labels = c("No","Yes"))

(conf_logit_train2 <- confusionMatrix(class_logit_train2, train2_balanceada$Exited,positive = "Yes"))
(conf_probit_train2 <- confusionMatrix(class_probit_train2, train2_balanceada$Exited,positive = "Yes"))

# Función para calcular F1 desde confusionMatrix
calcular_f1 <- function(conf){
  precision <- conf$byClass["Precision"]
  recall    <- conf$byClass["Recall"]
  f1        <- 2 * (precision * recall) / (precision + recall)
  return(f1)
}

(f1_logit_train2 <- calcular_f1(conf_logit_train2))
(f1_probit_train2 <- calcular_f1(conf_probit_train2))

# Predicciones de TEST2
pred_logit_test2  <- predict(modelo_logit,  newdata = test2, type = "response")
pred_probit_test2 <- predict(modelo_probit, newdata = test2, type = "response")

library(caret)

mejor_umbral_f1 <- function(prob, real){
  
  real <- as.factor(real)
  umbrales <- seq(0.01, 1, by = 0.01)
  
  f1_scores <- sapply(umbrales, function(t){
    pred <- ifelse(prob > t, "Yes", "No")
    pred <- factor(pred, levels = levels(real))
    F1 <- caret::F_meas(pred, real)
    return(F1)
  })
  
  # Mejor umbral y mejor F1
  umbral_opt <- umbrales[which.max(f1_scores)]
  f1_opt     <- max(f1_scores)
  
  return(list(
    umbral = umbral_opt,
    F1     = f1_opt,
    tabla  = data.frame(umbral = umbrales, f1 = f1_scores)
  ))
}

res_probit <- mejor_umbral_f1(pred_probit_test2, test2$Exited)
res_probit$umbral
res_probit$F1


# Convertir a clases segun umbral
class_logit_test2  <- ifelse(pred_logit_test2  > 0.4, 1, 0)
class_probit_test2 <- ifelse(pred_probit_test2 > 0.4, 1, 0)

# Matriz de confusión
# Convertir predicción binaria a "No"/"Yes"
class_logit_test2_factor <- ifelse(class_logit_test2 == 1, "Yes", "No")
class_logit_test2_factor <- factor(class_logit_test2_factor,
                                   levels = c("No", "Yes"))

test2$Exited <- factor(test2$Exited, 
                                   levels = c("No", "Yes"))

# Ahora sí funciona
z <- confusionMatrix(class_logit_test2_factor, test2$Exited, positive = "Yes")

a<-confusionMatrix(class_probit_test2, test2$Exited,positive = "Yes")

# --- F1 de test2
f1_logit_test2  <- calcular_f1(z)
f1_probit_test2 <- calcular_f1(a)

cat("F1 Logit - test2:", f1_logit_test2, "\n")
cat("F1 Probit - test2:", f1_probit_test2, "\n")

##########  hacer load de data porque no tenemos id de test
#predicciones: para probit
pred_test_probit <- predict(modelo_probit, newdata = test, type = "response")

# Predicción binaria final
class_test_probit <- ifelse(pred_test_probit > 0.67 , "Yes", "No")
class_test_probit <- factor(class_test_probit, levels = c("No", "Yes"))


submission <- data.frame(ID = data$ID[7001:10000], Exited = class_test_probit)
write.csv(submission, "submission.csv", row.names = FALSE)


###

