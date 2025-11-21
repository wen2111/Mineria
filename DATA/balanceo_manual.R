#######################################################
############## DATA reducida BALANCEADA ###############
#######################################################

library("caret")
library(ggplot2)

load("dataaaaaaaaaaaaaa.RData")

data <- data_reducida

# SEPARAR TRAIN Y TEST
train <- subset(data, group == "train") # 7000 obs
test <- subset(data, group == "test")   # 3000 obs

# ELIMINAR "GROUP"
train$group <- NULL
test$group  <- NULL

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("1","0"),
                       labels = c("Yes","No"))

#######################################################
############## BALANCEO SOLO EN TRAIN #################
#######################################################

set.seed(123)

# Separar las clases
yes <- subset(train, Exited == "Yes")  # ~20%
no  <- subset(train, Exited == "No")   # ~80%

# Tamaño total deseado = mismo tamaño del train original
N_total <- nrow(train)  # 7000

# Proporción deseada 40% Yes - 60% No
n_yes <- round(0.40 * N_total)  # 2800
n_no  <- round(0.60 * N_total)  # 4200

# Generar el nuevo conjunto balanceado
yes_bal <- yes[sample(nrow(yes), n_yes, replace = TRUE), ]   # oversampling Yes
no_bal  <- no[sample(nrow(no),  n_no,  replace = TRUE), ]    # undersampling No

# Dataset final balanceado
data_reducida_balanceada <- rbind(yes_bal, no_bal)

#######################################################
############## COMPROBAR BALANCEO #####################
#######################################################

prop.table(table(train$Exited))                     # Antes
prop.table(table(data_reducida_balanceada$Exited))  # Después

ggplot(data_reducida_balanceada, aes(x = Exited)) +
  geom_bar()

save(data_reducida_balanceada, file = "reducida_balanceada.RData")

#######################################################
############## COMPROBAR QUE NO HAYA COSAS RARAS ######
#######################################################

load("reducida_balanceada.RData")
str(data_reducida_balanceada)
summary(data_reducida_balanceada)
