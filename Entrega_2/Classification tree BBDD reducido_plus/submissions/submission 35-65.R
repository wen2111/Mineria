predicciones <- read.csv("~/GitHub/Mineria/Entrega_1/4- Modelación/submissions/grup5_sub2.csv")
load("~/GitHub/Mineria/DATA/data_imputado.RData")
ID<-data_imputado$ID
load("~/GitHub/Mineria/DATA/dataaaaaaaaaaaaaa.RData")
data_reducida_plus$ID<-ID
datatest_reducido<-data_reducida_plus[7001:10000,]
datatest_reducido$group<-NULL

data_reducida_plus$ID<-NULL
data_reducida_plus$group<-NULL
datatrainfull<-data_reducida_plus[0:7000,]
library(ROSE)
library(caret)
library(dplyr)

# Configurar semillas para reproducibilidad
set.seed(123)
seeds <- vector(mode = "list", length = 6)
for(i in 1:5) seeds[[i]] <- sample.int(1000, 10)
seeds[[6]] <- sample.int(1000, 1)

# Balancear datos a 50-50 con ROSE
data_balanced <- ROSE(Exited ~ ., data = datatrainfull, p = 0.35, seed = 123)$data

# Entrenar modelo
model <- train(Exited ~ ., 
               method = "rpart", 
               data = data_balanced,
               tuneLength = 10, 
               trControl = trainControl(
                 method = "cv", 
                 number = 5,
                 seeds = seeds
               ))
model

predicciones <- predict(model, newdata = datatest_reducido)
predicciones_factor <- ifelse(predicciones == "1", "Yes", "No")

# Crear dataframe con ID y predicción
resultado_final <- data.frame(
  ID = datatest_reducido$ID,
  Exited = predicciones_factor
)

# Guardar en Excel
write.csv(resultado_final, "~/GitHub/Mineria/Entrega_2/Classification tree BBDD reducido_plus/submit_reducida_plus_balance35.csv", row.names = FALSE)
