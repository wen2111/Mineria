# install.packages("psych")
# install.packages("dlookr")

## ==== 2. Análisis exploratorio ====

# 2.1 Análisis exploratorio de una variable
# 2.1.1 Numérica
library(psych)
psych::describe(data[, varNum])


## 2.1.1.2 Gráficos
### base
par(mfrow = c(2, 4))  
for (var in varNum) {
  hist(data[, var], main = paste0("Histograma variable ", var))
  boxplot(data[, var], main = paste0("Boxplot variable ", var))
}

### ggplot2
install.packages("ggplot2")
install.packages("patchwork")
library(ggplot2)
library(patchwork)
# Create a list of ggplot objects
plots <- list()

for (var in varNum) {
  binwidth_value <- diff(range(data[[var]])) / 30
  
  histo <- ggplot(data, aes(x = .data[[var]])) + 
    geom_histogram(aes(y = ..density..), colour = "black", fill = "white", binwidth = binwidth_value) +
    geom_density(alpha = .2, fill = "#FF6666") +
    geom_vline(aes(xintercept = mean(.data[[var]], na.rm = TRUE)),
               color = "blue", linetype = "dashed", linewidth = 1) +
    ggtitle(paste("Histograma de", var))
  
  boxp <- ggplot(data, aes(x = .data[[var]])) + 
    geom_boxplot(outlier.colour = "red", outlier.shape = 8,
                 outlier.size = 4) +
    ggtitle(paste("Boxplot de", var))
  
  plots <- append(plots, list(histo, boxp))
}

# Use patchwork to combine plots in a 2-column layout
dev.new()
final_plot <- Reduce(`+`, plots) + plot_layout(ncol = 2) 
final_plot


# 2.1.2 Categórica
## 2.1.2.1 Descriptivo
for (var in varCat) {
  tablaAbs <- data.frame(table(data[, var]))
  tablaFreq <- data.frame(table(data[, var])/sum(table(data[, var])))
  m <- match(tablaAbs$Var1, tablaFreq$Var1)
  tablaAbs[, "FreqRel"] <- tablaFreq[m, "Freq"]
  colnames(tablaAbs) <- c("Categoria", "FreqAbs", "FreqRel")
  
  cat("===============", var, "===================================\n")
  print(tablaAbs)
  cat("==================================================\n")
}

## 2.1.2.2 Gráficos
### base
par(mfrow = c(2, 3))  
for (var in varCat) {
  barplot(table(data[, var]))
}
par(mfrow = c(1, 1))  

### ggplot2
library(ggplot2)
library(gridExtra)

plots <- list()  # lista vacía
i <- 1           # índice

for (var in varCat) {
  tabla <- data.frame(table(data[, var]) / sum(table(data[, var])))
  
  p <- ggplot(data = tabla, aes(x = Var1, y = Freq)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = paste0(round(Freq * 100, 2), "%")),
              vjust = 1.6, color = "white", size = 3.5) +
    theme_minimal() +
    labs(title = paste("Distribución de", var), x = var, y = "Proporción")
  
  plots[[i]] <- p
  i <- i + 1
}

# Mostrar todos los gráficos en un grid (ejemplo con 2 columnas)
grid.arrange(grobs = plots, ncol = 2)


