# hacer un load de data_imputado balanceada i dataaaaaaaaaaaaaaaaaaaaaaa
library(caret)
hola<-data_imputada_balanceada[,c(1,3,8,9,13,20)]
train  <- hola[1:7000, ]
test   <- data_reducida[7001:10000, -c(5,8)]

train$Exited<-factor(ifelse(train$Exited %in% c("Yes", 1), "1", "0"), levels = c("0","1"))

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)

train2 <- train[index,]     # Entrenamiento interno
test2  <- train[-index, ]    # Validación interna

# MODELOS GLM

# Logit
modelo_logit <- glm(Exited ~ ., data = train2, family = binomial(link = "logit"))

# Probit
modelo_probit <- glm(Exited ~ ., data = train2, family = binomial(link = "probit"))

# matrix para train2
pred_logit_train2 <- predict(modelo_logit, newdata = train2, type = "response")
pred_probit_train2 <- predict(modelo_probit, newdata = train2, type = "response")

class_logit_train2 <- ifelse(pred_logit_train2 > (1/3), 1, 0)
class_probit_train2 <- ifelse(pred_probit_train2 > (1/3), 1, 0)

(conf_logit_train2 <- confusionMatrix(as.factor(class_logit_train2), train2$Exited,positive = "1"))
(conf_probit_train2 <- confusionMatrix(as.factor(class_probit_train2), train2$Exited,positive = "1"))

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

# Convertir a clases segun umbral
class_logit_test2  <- ifelse(pred_logit_test2  > (1/3), 1, 0)
class_probit_test2 <- ifelse(pred_probit_test2 > (1/3), 1, 0)

# Matriz de confusión
z<-confusionMatrix(as.factor(class_logit_test2),  test2$Exited,positive = "1")
a<-confusionMatrix(as.factor(class_probit_test2), test2$Exited,positive = "1")

# --- F1 de test2
f1_logit_test2  <- calcular_f1(a)
f1_probit_test2 <- calcular_f1(z)

cat("F1 Logit - test2:", f1_logit_test2, "\n")
cat("F1 Probit - test2:", f1_probit_test2, "\n")

##########  hacer load de data porque no tenemos id de test
#predicciones: para probit
pred_test_probit <- predict(modelo_probit, newdata = test, type = "response")

# Predicción binaria final
class_test_probit <- ifelse(pred_test_probit > 0.5 , "Yes", "No")
class_test_probit <- factor(class_test_probit, levels = c("No", "Yes"))


submission <- data.frame(ID = data$ID[7001:10000], Exited = class_test_probit)
write.csv(submission, "submission.csv", row.names = FALSE)


###

table(class_test_probit)

