#Repetidos: 

load("~/Documents/GitHub/Mineria/DATA/dataaaaaaaaaaaaaa.RData")
#First start with the transformed data: 

data_transformada_n<-data_transformada
summary(data_transformada_n)
duplicated(data_transformada_n) 
data_transformada_n[duplicated(data_transformada_n), ]#no hay duplicados 

#Observo en la base de datos con na, para identificar que individuos tienen 
#el mismo surname 

load("~/Documents/GitHub/Mineria/DATA/data.RData")

data_sur<-data
duplicated(data_sur) 
data_sur[duplicated(data_sur), ]#No hay filas identicas
#Mirar aquellos individuos que tienen el mismo apellido. 
data_sur$surname_clean <- trimws(tolower(data_sur$Surname))
data_sur[duplicated(data_sur$surname_clean), ]
which(duplicated(data_sur$surname_clean) | duplicated(data_sur$surname_clean, fromLast = TRUE))
sum(duplicated(data_sur$surname_clean) | duplicated(data_sur$surname_clean, fromLast = TRUE))
#numero total de individuos tienen surnames iguales 
#Al no tener el nombre del individuo, buscamos individuos con muchas caracteristicas iguales

#Volvemos a la base de datos transformada despues de ser imputado: 
#Para poder eliminar filas redundantes, aquellas con valores iguales, 
#Se elige un conjunto razonable de columnas “estables” y se eliminan filas 
#repetidas según ese conjunto. 

#Las columas a continuación, son aquellas las columnas que no deberían cambiar nunca 
#para una misma persona son:

keys <- c("Gender", "Age", "Geography", "MaritalStatus", "EducationLevel")

data_sin_duplicados <- data_transformada_n[!(duplicated(data_transformada_n[, keys]) & data_transformada_n$group == "train"), ]
sum(data_sin_duplicados$group=="train")

#Un total de 7473

# Más estricto: 
keys_e <- c("Gender", "Age", "Geography", "MaritalStatus", "EducationLevel", "LoanStatus")
data_sin_duplicados_e <- data_transformada_n[!(duplicated(data_transformada_n[, keys_e]) & data_transformada_n$group == "train"), ]

save(data_sin_duplicados, file = "data_transformada_r.RData")
save(data_sin_duplicados_e, file = "data_transformada_re.RData")
# Guardar ambos datasets en un único archivo RData
#save(data_sin_duplicados, data_sin_duplicados_e, file = "data_transformada_todos.RData")




