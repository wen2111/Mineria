train_data<-read.csv2("DATA/train.csv", sep=",")
test_data<-read.csv2("DATA/test.csv", sep=",")
submit_data<-read.csv2("DATA/sample_submission.csv", sep=",")

summary(test_data)
summary(train_data)

train_data$group<-"train"
test_data$group<-"test"
train_data<-train_data[,-1]
test_data$Exited<-NA
common_columns <- intersect(names(train_data), names(test_data))
common_columns
train_data <- train_data[,common_columns]
test_data <- test_data[,  common_columns]
data<-rbind(train_data,test_data)
data <- data[, c("Exited", setdiff(names(data), "Exited"))]
