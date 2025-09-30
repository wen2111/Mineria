# install.packages("psych")
# install.packages("dlookr")


# Variables categoricas y numéricas
## paso 0  prueba : data<-train_data

varCat <- c(
  "CustomerSegment", "Gender", "MaritalStatus", "EducationLevel", 
  "HasCrCard", "SavingsAccountFlag", "LoanStatus", "TransactionFrequency", 
  "DigitalEngagementScore", "ComplaintsCount", "NetPromoterScore", "Exited"
)
data[varCat] <- lapply(data[varCat], as.factor)

varNum <- c(
  "EstimatedSalary", "CreditScore", "Tenure", "Balance", "NumOfProducts"
)


## ==== AED ====
# install.packages("skimr")
# install.packages("tidyverse")
library(skimr)
library(tidyverse)

## Podem visualitzar un descriptiu de les dades 
skim(data)

