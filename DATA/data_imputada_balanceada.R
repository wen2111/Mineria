#######################################################
############## DATA IMPUTADA BALANCEADA ###############
#######################################################

library("caret")
library(ggplot2)
library(ROSE)

load("dataaaaaaaaaaaaaa.RData")

data <- data_imputado

# SEPARAR TRAIN Y TEST
train <- subset(data, group == "train") # 7000 obs
test <- subset(data,group == "test") # 3000 obs variable respuesta vacia

# ELIMINAR "GROUP"
train$group <- NULL
test$group  <- NULL

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                                levels = c("1","0"),
                                labels = c("Yes","No"))

# BALANCEO SOLO EN TRAIN (7000 OBS)
data_imputada_balanceada <- ROSE(
  Exited ~ .,
  data = train,
  p = 0.4,      # 40% Yes, 60% No
  seed = 123
)$data


# COMPROBAR BALANCEO
prop.table(table(train$Exited))                     # Antes del balanceo
prop.table(table(data_imputada_balanceada$Exited))  # Después del balanceo

ggplot(data_imputada_balanceada, aes(x = Exited)) +
  geom_bar()

save(data_imputada_balanceada, file = "imputada_balanceada.RData")


################################ FIN ###########################################