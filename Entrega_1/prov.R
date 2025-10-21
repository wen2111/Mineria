# provisional
summary(data_reducida) # poner la data transformada

# categoroizar el resto denumericas i aplciar el codigo tenieno en cuenta los va

# categorizar
cortes <- c(17, 30, 50, Inf)
etiquetas <- c("Joven", "Mediana", "Mayor")
data_reducida$edad_cat <- cut(data_reducida$Age, breaks = cortes, labels = etiquetas, right = TRUE)

cortes <- c(0, 1, 97054, 127638, Inf)
etiquetas <- c("Nada", "Medio", "Alto", "Muy alto")
data_reducida$Balance_cat <- cut(data_reducida$Balance,
                      breaks = cortes,
                      labels = etiquetas,
                      right = FALSE,
                      include.lowest = TRUE)
data<-data_reducida[,-c(6,7)]

data2<-as(data,"transactions")

# retocar los parametros
rules = apriori (data2, parameter = list (support=0.25, confidence=0.9, maxlen = 10, minlen=2))
rules
summary(rules)
inspect(sort(x = rules, decreasing = TRUE, by = "confidence"))

filtrado_reglas <- subset(x = rules,
                          subset = rhs %in% c("Exited=0", "Exited=1"))
inspect(filtrado_reglas)
