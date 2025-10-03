# install.packages("psych")
# install.packages("dlookr")
# install.packages("ggplot2")
# install.packages("patchwork")
# install.packages("ggcorplot")

## Cosas a decidir

# Categorizar la variable compliancounts, mayoria son 0 i num of products
# decidir como maximo cuantos missings por fila

## ==== Análisis exploratorio ====

# 1 Análisis exploratorio de una variable ! FALTA INTERPRETAR

# 1.1 Numérica
library(psych)
psych::describe(data[, varNum])

##Gráficos
### base
par(mfrow = c(2, 4))  
for (var in varNum) {
  hist(data[, var], main = paste0("Histograma ", var))
  boxplot(data[, var], main = paste0("Boxplot ", var))
}

### ggplot2 
library(ggplot2)
library(patchwork)

# Crear listas separadas
plots_histo <- list()
plots_box <- list()

for (var in varNum) {
  binwidth_value <- diff(range(data[[var]])) / 30
  
  histo <- ggplot(data, aes(x = .data[[var]])) + 
    geom_histogram(aes(y = ..density..), colour = "black", fill = "white", binwidth = binwidth_value) +
    geom_density(alpha = .2, fill = "#FF6666") +
    geom_vline(aes(xintercept = mean(.data[[var]], na.rm = TRUE)),
               color = "blue", linetype = "dashed", linewidth = 1) +
    ggtitle(paste("Histograma de", var))
  
  boxp <- ggplot(data, aes(x = .data[[var]])) + 
    geom_boxplot(outlier.colour = "red", outlier.shape = 8, outlier.size = 4) +
    ggtitle(paste("Boxplot de", var))
  
  plots_histo <- append(plots_histo, list(histo))
  plots_box <- append(plots_box, list(boxp))
}

dev.new()
final_histo <- Reduce(`+`, plots_histo) + plot_layout(ncol = 2)
final_box <- Reduce(`+`, plots_box) + plot_layout(ncol = 2)
final_histo
final_box

# 1.2 Categórica
## Descriptivo
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

## Gráficos
### base
par(mfrow = c(2, 3))  
for (var in varCat) {
  barplot(table(data[, var]),main = var)
}
par(mfrow = c(1, 1))  

### ggplot2
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
dev.new()
grid.arrange(grobs = plots, ncol = 2)

# 2 Análisis exploratorio de una bivariante ! NO ESTA ACABADO I FALTA INTERPRETAR

## Num VS Nun

cor(data[, varNum],use = "pairwise.complete.obs")
corPlot(data[, varNum])

# Correlacion negativa entre ComplaintsCount i NetPromoterScore -0.69
# Correlación negativa entre Balance i NumOfProducts de -0.31
# El resto tienen un valor muy bajo en valor absoluto.

## Cat VS Cat

for (varc1 in varCat) {
  for (varc2 in varCat) {
    if (varc1 != varc2) {
      prop_table <- prop.table(table(data[, varc1], data[, varc2]))
      cat("=============", varc1, " vs. ", varc2, "=========================\n")
      print(prop_table)
    }
  }
}

par(mfrow = c(3, 3))  
for (varc1 in varCat) {
  for (varc2 in varCat) {
    if (varc1 != varc2) {
      prop_table <- prop.table(table(data[, varc1], data[, varc2]))
      barplot(prop_table, beside = TRUE,main = paste0(varc1,"&",varc2))
    }
  }
}


## Num VS Cat

ggplot(data, aes(x = factor(Exited), y = NetPromoterScore, fill = factor(Exited))) +
  geom_boxplot(alpha = 0.7) +
  labs(x = "Exited", 
       y = "Net Promoter Score",
       title = "Distribución de NetPromoterScore por estado de salida (Exited)") +
  scale_fill_manual(values = c("#66CC99", "#FF6666"), 
                    name = "Exited", 
                    labels = c("No", "Sí")) +
  theme_minimal()

#violinplot
ggplot(data, aes(x = factor(Exited), y = NetPromoterScore, fill = factor(Exited))) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white") +
  labs(x = "Exited", 
       y = "Net Promoter Score",
       title = "Distribución (violin plot) de NetPromoterScore por Exited") +
  scale_fill_manual(values = c("#66CC99", "#FF6666"), 
                    name = "Exited", 
                    labels = c("No", "Sí")) +
  theme_minimal()

#intento para todas las numericas con exited

for (var in varNum) {
  # Crear gráfico filtrando NA sobre la marcha
  p <- ggplot(data, aes(x = factor(Exited), y = .data[[var]], fill = factor(Exited))) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA, na.rm = TRUE) +
    labs(x = "Exited", 
         y = var,
         title = paste("Distribución de", var, "por Exited")) +
    scale_fill_manual(values = c("#66CC99", "#FF6666"), 
                      name = "Exited", 
                      labels = c("No", "Sí")) +
    theme_minimal()
  
  print(p)
}


# Variable numérica balance i las categroicas
numVar <- "Balance"
for (var in varCat) {
  p <- ggplot(data, aes(x = .data[[var]], y = .data[[numVar]], fill = .data[[var]])) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA, na.rm = TRUE) +
    labs(x = var,
         y = numVar,
         title = paste("Distribución de", numVar, "por", var)) +
    theme_minimal() +
    theme(legend.position = "none")  # opcional: quitar leyenda
  
  print(p)
}
# LO MISMO PERO quitar el boxplot de NA
for (var in varCat) {
  # Filtrar filas donde la categoría o balance no sean NA
  data_filtrada <- data[!is.na(data[[var]]) & !is.na(data[[numVar]]), ]
  
  p <- ggplot(data_filtrada, aes(x = .data[[var]], y = .data[[numVar]], fill = .data[[var]])) +
    geom_boxplot(alpha = 0.7, outlier.shape = NA) +
    labs(x = var,
         y = numVar,
         title = paste("Distribución de", numVar, "por", var)) +
    theme_minimal() +
    theme(legend.position = "none")  # opcional: quitar leyenda
  
  print(p)
}

# Aplicar tapply a todas las variables categóricas de varCat para media
resultados <- lapply(varCat, function(var) {
  tapply(data[[numVar]], data[[var]], mean, na.rm = TRUE)
})
names(resultados) <- varCat
resultados

# Missings

# Se sabe que los missings són aleatorios por los profes

library(visdat)

vis_miss(data)
# hay un 30% de missings en cada variable
#el id con un 21%, debido que en el test no exite missings.

na_por_fila <- rowSums(is.na(data))
filas_15_miss <- which(na_por_fila >= 11)
s<-data[filas_15_miss,]
sum(s$group=="test") # hay 212 individuos con 11 o mas missings


## Aleatoriedad de los missings
install.packages("naniar")
library(naniar)
naniar::mcar_test(data)
# p-valor = 0.387 > 0.05, por lo que no se rechaza H0: los missings son aleatorios (MCAR)