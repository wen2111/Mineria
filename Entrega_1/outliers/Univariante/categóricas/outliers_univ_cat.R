# outliers variables categoricas

# librerias
library(ggplot2)
library(dplyr)
library(scales)

# variables a tratar: 10 (sin "group")
names(Filter(is.factor, data))
sum(sapply(data, is.factor))
varCat_sin_group_sin_group <- setdiff(varCat, "group")

# niveles por variable
niveles_por_variable <- sapply(data[varCat_sin_group_sin_group], function(x) length(unique(x)))
niveles_por_variable

total_modalidades <- sum(sapply(data[varCat_sin_group], function(x) length(unique(x))))
total_modalidades # pone 37 porque cuenta NA's como una modalidad para cada varCat

# Umbrales
umbral_bajo <- 0.05  # <1% del total
umbral_alto <- 0.9   # >90% del total

resumen_list <- list()
for (var in varCat_sin_group) {
  freq <- prop.table(table(data[[var]]))
  df_freq <- as.data.frame(freq)
  names(df_freq) <- c("Categoria", "Proporcion")
  
  df_freq <- df_freq %>%
    mutate(Outlier = case_when(
      Proporcion < umbral_bajo ~ "Muy baja",
      Proporcion > umbral_alto ~ "Muy alta",
      TRUE ~ "Normal"
    ))
  
  resumen_list[[var]] <- df_freq %>%
    mutate(Variable = var) %>%
    select(Variable, Categoria, Proporcion, Outlier)

resumen_outliers_df <- bind_rows(resumen_list)
View(resumen_outliers_df) # hay 27 modalidades

# barplots
p <- ggplot(df_freq, aes(x = Categoria, y = Proporcion, fill = Outlier)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = percent(Proporcion)), vjust = -0.5) +
    scale_fill_manual(values = c("Muy baja" = "red", "Muy alta" = "orange", "Normal" = "steelblue")) +
    labs(y = "Proporción", title = paste("Distribución de", var)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
}

# conclusion: no hay modalidades atipicas

