train_data<-read.csv2("train.csv", sep=",")
test_data<-read.csv2("test.csv", sep=",")
#submit_data<-read.csv2("sample_submission.csv", sep=",")
train_data$group<-"train"
test_data$group<-"test"
train_data<-train_data[,-1]
test_data$Exited<-NA
common_columns <- intersect(names(train_data), names(test_data))
#common_columns
train_data <- train_data[,common_columns]
test_data <- test_data[,  common_columns]
data<-rbind(train_data,test_data)
data <- data[, c("Exited", setdiff(names(data), "Exited"))]
varCat<-c("Geography", "Gender", "MaritalStatus", "EducationLevel","HasCrCard", 
          "SavingsAccountFlag", "LoanStatus","CustomerSegment","Exited","group" ,"IsActiveMember")
varNum<-c("Age", "CreditScore", "Tenure","EstimatedSalary", "Balance", "NumOfProducts",
          "TransactionFrequency","AvgTransactionAmount","DigitalEngagementScore",
          "ComplaintsCount","NetPromoterScore")
data[varCat] <- lapply(data[varCat], factor)
data[varNum] <- lapply(data[varNum], as.numeric)
str(data)

