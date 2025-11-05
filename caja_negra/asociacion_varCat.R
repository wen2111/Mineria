# asociacion entre variables categoricas

# Paquetes necesarios
# install.packages(c("reshape2"))
library(ggplot2)
library(reshape2)

# variables categóricas (sin la variable objetivo)
varCatIndep <- c("CustomerSegment", "Gender", "MaritalStatus", "EducationLevel",
                 "HasCrCard", "SavingsAccountFlag", "LoanStatus",
                 "TransactionFrequency", "DigitalEngagementScore",
                 "ComplaintsCount", "NetPromoterScore")

# matriz vacía para guardar los p-valores
p_matrix <- matrix(NA, nrow = length(varCatIndep), ncol = length(varCatIndep),
                   dimnames = list(varCatIndep, varCatIndep))

# Bucle test de chi-cuadrado
for (i in 1:(length(varCatIndep)-1)) {
  for (j in (i+1):length(varCatIndep)) {
    tbl <- table(data[[varCatIndep[i]]], data[[varCatIndep[j]]])
    chi <- suppressWarnings(chisq.test(tbl))  # evitar warnings de celdas pequeñas
    p_matrix[i, j] <- chi$p.value
    p_matrix[j, i] <- chi$p.value
  }
}

# formato largo para graficar
p_df <- melt(p_matrix, na.rm = TRUE)
names(p_df) <- c("Var1", "Var2", "p_value")

# heatmap
ggplot(p_df, aes(x = Var1, y = Var2, fill = p_value)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "red", high = "white",
                      name = "p-value",
                      limits = c(0, 1),
                      na.value = "grey90") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Asociación entre variables categóricas (Chi-cuadrado)",
       x = "Variable",
       y = "Variable")
