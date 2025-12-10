#######################EJECUTAR ASSOCIATION RULES PRIMERO##################
#######################HASTA TENER DATA_TR####################

#########################PREPARACION DE LOS DATOS#####################
library(dplyr)
library(arules)

# ===============================
# 1️⃣ Preparar data_ar
# ===============================
data_ar <- subset(data_transformada, select = -c(group))
data_ar <- data_ar[7001:10000, ]

# Convertir caracteres a factor
data_ar <- data_ar %>%
  mutate(across(where(is.character), as.factor))

# Discretizaciones
data_ar <- data_ar %>%
  mutate(
    Tenure = cut(Tenure, breaks = c(0, 3, 6, 10),
                 labels = c("Nuevo (0-3 años)", "Medio (4-6 años)", "Antiguo (7-10 años)"),
                 include.lowest = TRUE),
    NetPromoterScore = cut(NetPromoterScore, breaks = c(-1, 6, 8, 10),
                           labels = c("0-6", "7-8", "9-10"), include.lowest = TRUE),
    TransactionFrequency = cut(TransactionFrequency,
                               breaks = c(0, 20, 30, 40, max(TransactionFrequency, na.rm = TRUE)),
                               labels = c("0-20", "21-30", "31-40", "41+"), include.lowest = TRUE),
    Age = cut(Age, breaks = c(0, 25, 35, 45, 55, 65, 100),
              labels = c("18-25", "26-35", "36-45", "46-55", "56-65", "65+"), include.lowest = TRUE),
    EstimatedSalary = cut(EstimatedSalary,
                          breaks = c(0, 30000, 60000, 90000, 120000, 150000, 180000, max(EstimatedSalary, na.rm = TRUE)),
                          labels = c("0-30K", "31-60K", "61-90K", "91-120K", "121-150K", "151-180K", "180K+"),
                          include.lowest = TRUE),
    AvgTransactionAmount = cut(AvgTransactionAmount,
                               breaks = quantile(AvgTransactionAmount, probs = c(0, 0.5, 0.8, 0.95, 1), na.rm = TRUE),
                               labels = c("Bajo (0-50%)", "Medio (51-80%)", "Alto (81-95%)", "Muy Alto (96-100%)"),
                               include.lowest = TRUE),
    DigitalEngagementScore = cut(DigitalEngagementScore, breaks = c(0, 25, 50, 75, 100),
                                 labels = c("0-25", "26-50", "51-75", "76-100"), include.lowest = TRUE),
    CreditScore = cut(CreditScore, breaks = c(300, 580, 670, 740, 800, 850),
                      labels = c("Muy Bajo (300-579)", "Bajo (580-669)", "Medio (670-739)", "Bueno (740-799)", "Excelente (800-850)"),
                      include.lowest = TRUE),
    Balance = cut(Balance, breaks = c(0, 1000, 5000, 15000, 50000, Inf),
                  labels = c("Muy Bajo (0-1K)", "Bajo (1-5K)", "Medio (5-15K)", "Alto (15-50K)", "Muy Alto (50K+)"),
                  include.lowest = TRUE)
  )

# ===============================
# 2️⃣ Convertir a transacciones
# ===============================
data_ar_trans <- as(data_ar, "transactions")
rules = apriori (data_tr, parameter = list (support=0.0125, confidence=0.5, maxlen = 5, minlen=2))
summary(rules)
reglas_Noredund <- rules[!is.redundant(x = rules, measure = "confidence")]


reglas1 <- subset(reglas_Noredund, rhs %pin% "Exited=1")

# Inicializamos vector lógico para marcar filas a imputar
rows_to_impute <- rep(FALSE, length(data_ar_trans))

# Iterar sobre las reglas
for(i in seq_along(reglas1)) {
  lhs_rule <- lhs(reglas1[i])
  # is.subset verifica qué transacciones contienen todos los items del LHS
  match_rows <- is.subset(lhs_rule, data_ar_trans)
  # is.subset devuelve una matriz lógica: [regla, transacción]
  rows_to_impute <- rows_to_impute | as.logical(match_rows)
}

# Imputar Exited = 1 donde corresponda
data_ar_imputed <- data_ar
data_ar_imputed$Exited[rows_to_impute] <- 1
id<-data[7001:10000,]$ID
data_ar_imputed$ID<-id
data_ar_imputed2<-data_ar_imputed[,c("ID", "Exited")]
data_ar_imputed2 <- data_ar_imputed2 %>%
  filter(!is.na(Exited))
data_ar_imputed2$Exited<-ifelse(data_ar_imputed2==1, "Yes", "")
#Guardamos los datos
saveRDS(
  data_ar_imputed2,
  "~/GitHub/Mineria/DATA/A NUEVOS TEST CON IMPUTADOS DE REPETIDOS/test_df_imputed.rds"
)
