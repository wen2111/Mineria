###############################################
##### GLM BOOTSTRAP REDUCIDA SIN BALANCEAR ####
###############################################

library(caret)

load("/cloud/project/dataaaaaaaaaaaaaa.RData")

mydata <- data_reducida

# SEPARAR TRAIN Y TEST
train <- subset(mydata, group == "train") # 7000 obs
test <- subset(mydata, group == "test")   # 3000 obs

# ELIMINAR VARIABLES NO NECESARIAS
variables_eliminar <- c("group", "Surname", "ID")
train <- train[, !names(train) %in% variables_eliminar]
test <- test[, !names(test) %in% c("group", "Surname")] # Mantener ID en test si es necesario para el submit

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("1","0"),
                       labels = c("Yes","No"))

# PARTICION TRAIN2/TEST2

set.seed(123)
index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

# BOOTSTRAP

ctrl_boot_auc <- trainControl(method = "boot", 
                              number = 200,         # 200 samples
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)
fit_boot_auc <- train(Exited ~ ., data=train2, 
                      method = "glm", family = "binomial",
                      trControl = ctrl_boot_auc, metric = "ROC")
auc_boot <- fit_boot_auc$results$ROC
cat('Area under curve (Bootstrap):', round(as.numeric(auc_boot),3), '\n')

# Accuracy metric
ctrl_boot_acc <- trainControl(method = "boot", 
                              number = 200, 
                              classProbs = TRUE,
                              summaryFunction = defaultSummary)
fit_boot_acc <- train(Exited ~ ., data=train2, 
                      method = "glm", family = "binomial",
                      trControl = ctrl_boot_acc, metric = "Accuracy")
acc_boot <- fit_boot_acc$results$Accuracy


cat('Area under curve (Bootstrap):', round(as.numeric(auc_boot),3), '\n')


