# outliers variables categoricas

# librerias
library(ggplot2)
library(dplyr)
library(scales)

# variables a tratar: 12
names(Filter(is.factor, data))
sum(sapply(data, is.factor))

# niveles por variable
niveles_por_variable <- sapply(data[varCat], function(x) length(unique(x)))
niveles_por_variable

# Umbrales
umbral_bajo <- 0.05  # <1% del total
umbral_alto <- 0.9   # >90% del total

resumen_list <- list()
for (var in varCat) {
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
  View(resumen_outliers_df)
  
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

