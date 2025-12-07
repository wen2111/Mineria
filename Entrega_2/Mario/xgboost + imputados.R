load("~/GitHub/Mineria/DATA/dataaaaaaaaaaaaaa.RData")
#xgboost
library(caret)
library(FactoMineR)  

mydata <- data_reducida
mydata$group<-NULL

#dummifico data reducido
x<-mydata[,-3] #quito la respuesta
x<-x[,1:4] # cojo solo las cat
x <- fastDummies::dummy_cols(x, 
                             remove_first_dummy = TRUE,  
                             remove_selected_columns = TRUE)
x<-cbind(x,mydata[,6:7]) # adjunto las numericas
x$Exited<-mydata$Exited # añado la respuesta
mydata<-x

# Discretizamos Balance
cortes <- c(-Inf, 0, 30000, 60000, 90000, 120000, 150000, 180000, 210000, 240000, Inf) 
etiquetas <- c("Cero", "0-30k", "30k-60k", "60k-90k", "90k-120k", "120k-150k", 
               "150k-180k", "180k-210k", "210k-240k", ">240k")
mydata$Balance<- cut(
  data_reducida$Balance, 
  breaks = cortes, 
  labels = etiquetas, 
  right = TRUE, 
  include.lowest = TRUE
)
# SEPARAR TRAIN Y TEST
train <- mydata[1:7000,]
test <- mydata[7001:10000,]  # 3000 obs

# LABELS PARA EXITED
train$Exited <- factor(train$Exited,
                       levels = c("0","1"),
                       labels = c("No","Yes"))

# PARTICION TRAIN2/TEST2

set.seed(123)

index <- createDataPartition(train$Exited, p = 0.7, list = FALSE)
train2 <- train[index, ] # train interno
test2  <- train[-index, ] # test interno

library(MLmetrics)

f1_recall_summary <- function(data, lev = NULL, model = NULL) {
  precision <- Precision(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  recall <- Recall(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  f1 <- F1_Score(y_true = data$obs, y_pred = data$pred, positive = "Yes")
  c(F1 = f1, Recall = recall, Precision = precision)
}

ctrl_boot_auc <- trainControl(method = "cv", 
                              number = 5 ,         
                              classProbs = TRUE,
                              summaryFunction = f1_recall_summary
                              #, sampling = "up"
)

fit_boot_auc <- train(Exited ~ ., data=train2, 
                      method = "xgbTree",
                      trControl = ctrl_boot_auc, metric = "F1",
                      preProcess = c("scale"),verbosity = 0
)

#fit_boot_auc$results$F1
# Predicciones probabilísticas
train_pred_prob <- predict(fit_boot_auc, newdata = train2, type = "prob")
test_pred_prob  <- predict(fit_boot_auc, newdata = test2,  type = "prob")

train_pred_cut <- ifelse(train_pred_prob$Yes > 0.2071429, "Yes", "No")
test_pred_cut  <- ifelse(test_pred_prob$Yes > 0.2071429, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))
test_pred_cut  <- factor(test_pred_cut,  levels = c("No","Yes"))

# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train2$Exited,positive = "Yes")
conf_test  <- confusionMatrix(test_pred_cut, test2$Exited,positive = "Yes")

# F1-score function
f1_score <- function(cm){
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  return(as.numeric(f1))
}


f1_train <- f1_score(conf_train)
f1_test  <- f1_score(conf_test)


kpis <- data.frame(
  Dataset = c("Train2", "Test2"),
  Error_rate = c(1 - conf_train$overall["Accuracy"],
                 1 - conf_test$overall["Accuracy"]),
  Accuracy = c(conf_train$overall["Accuracy"],
               conf_test$overall["Accuracy"]),
  Precision = c(conf_train$byClass["Pos Pred Value"],
                conf_test$byClass["Pos Pred Value"]),
  Recall_Sensitivity = c(conf_train$byClass["Sensitivity"],
                         conf_test$byClass["Sensitivity"]),
  Specificity = c(conf_train$byClass["Specificity"],
                  conf_test$byClass["Specificity"]),
  F1_Score = c(f1_train, f1_test)
)

kpis

#final train
set.seed(123)
fit_final_glm <- train(Exited ~ ., data=train, 
                       method = "xgbTree",
                       trControl = ctrl_boot_auc, metric = "F1",
                       preProcess = c("scale"),verbosity = 0
)

train_pred_prob <- predict(fit_final_glm, newdata = train, type = "prob")
train_pred_cut <- ifelse(train_pred_prob$Yes > 0.2071429, "Yes", "No")
# Pasamos a clase: yes/no
train_pred_cut <- factor(train_pred_cut, levels = c("No","Yes"))

# Matrices de confusión
conf_train <- confusionMatrix(train_pred_cut, train$Exited,positive = "Yes")
conf_train
f1_score(conf_train)

# prediccion
pred_kaggle_prob <- predict(fit_final_glm, newdata = test, type = "prob")
pred_kaggle_class <- ifelse(pred_kaggle_prob$Yes > 0.2071429, "Yes", "No")
pred_kaggle_class2 <- ifelse(pred_kaggle_prob$Yes > 0.205921, "Yes", "No")

#aplicamos los imputados
test_df_imputed <- readRDS("~/GitHub/Mineria/DATA/A NUEVOS TEST CON IMPUTADOS DE REPETIDOS/test_df_imputed.rds")
library(dplyr)
#ID originales de test
load("~/GitHub/Mineria/DATA/data.RData")
id<-data[7001:10000,]$ID
# Unir por ID y actualizar Exited
submission <- data.frame(ID = id, Exited = pred_kaggle_class)
table(submission$Exited)
submission <- submission %>%
  left_join(
    test_df_imputed %>% select(ID, Exited) %>% rename(Exited_new = Exited),
    by = "ID"
  ) %>%
  mutate(
    Exited = if_else(!is.na(Exited_new), Exited_new, Exited)
  ) %>%
  select(-Exited_new)
table(submission$Exited)

#Para ver la cantidad de Exited diferentes del modelo y la imputacion por AR
library(dplyr)
comparacion <- data_ar_imputed2 %>%
  mutate(Exited = as.character(Exited)) %>%
  inner_join(
    submission %>% mutate(Exited = as.character(Exited)),
    by = "ID",
    suffix = c("_imputed", "_submission")
  )

diferentes <- comparacion %>%
  filter(Exited_imputed != Exited_submission)

n_diferentes <- nrow(diferentes)
n_diferentes
#submission
write.csv(submission, "~/GitHub/Mineria/Entrega_2/Mario/xgboost_imputing_1.csv", row.names = FALSE)
