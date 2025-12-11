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

# Mantener la primera ocurrencia de cada duplicado dentro de 'train'
duplicado_train <- duplicated(data_transformada_n[data_transformada_n$group == "train", keys])

# Crear un vector lógico para todas las filas
keep <- rep(TRUE, nrow(data_transformada_n))

# Marcar como FALSE las filas duplicadas de 'train' (excepto la primera)
keep[data_transformada_n$group == "train"][duplicado_train] <- FALSE

# Filtrar los datos
data_sin_duplicados <- data_transformada_n[keep, ]
sum(data_sin_duplicados)

# Más estricto: 
keys_e <- c("Gender", "Age", "Geography", "MaritalStatus")
# Identificar duplicados dentro de 'train'
duplicado_train <- duplicated(data_transformada_n[data_transformada_n$group == "train", keys_e])

# Crear vector lógico para mantener todas las filas por defecto
keep <- rep(TRUE, nrow(data_transformada_n))

# Marcar como FALSE las filas duplicadas de 'train' (excepto la primera)
keep[data_transformada_n$group == "train"][duplicado_train] <- FALSE

# Filtrar el dataframe
data_sin_duplicados_e <- data_transformada_n[keep, ]

# Revisar
sum(data_sin_duplicados_e$group == "train")

# Más base columnas: 

keys_f <- c("Gender", "Age", "Geography", "MaritalStatus", "EducationLevel", "LoanStatus","HasCrCard","NetPromoterScore","CustomerSegment","LoanStatus")
data_sin_duplicados_f <- data_transformada_n[!(duplicated(data_transformada_n[, keys_f]) & data_transformada_n$group == "train"), ]


save(data_sin_duplicados, file = "data_transformada_r.RData")
save(data_sin_duplicados_e, file = "data_transformada_re.RData")
save(data_sin_duplicados_f, file = "data_transformada_fi.RData")
# Guardar ambos datasets en un único archivo RData
#save(data_sin_duplicados, data_sin_duplicados_e, file = "data_transformada_todos.RData")

# Con la base de datos reducido 
data_reducida_r<-data_reducida
reducido <- data_reducida_r[data_reducida_r$grupo == "train" & 
                                     duplicated(data_reducida_r[data_reducida_r$grupo == "train", ]), ]
wich
# Ver resultados
duplicados_test
# Con la de reducido plus



