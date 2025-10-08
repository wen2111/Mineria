# install.packages("psych")
# install.packages("dlookr")
# install.packages("ggplot2")
# install.packages("patchwork")
# install.packages("ggcorplot")

## Cosas a decidir

# Categorizar la variable complaincounts, mayoria son 0 i num of products, binaria i 3 grupos de num of prodtos
# decidir como maximo cuantos missings por fila <- nbo hacer nada de momento
# cargar las variables no signicativas para la variable respuesta. hacer modelo reducido
## ==== Análisis exploratorio ====

# 1 Análisis exploratorio de una variable 

# 1.1 Numérica
library(psych)
psych::describe(data[, varNum])

# Balance: Hay muchos 0 (cuentas inactivas)
# ComplaintsCount: la mayoría tiene 0 reclamos, y unos pocos con muchos reclamos, kurtosis 7,51.>3 Agrupar en rango.
# AvgTransactionAmount: la mayoría gasta poco, algunos tienen importes

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

# La variable respuesta Exited: 21% abandonó el banco, dataset desbalanceado
# La mayoría son alemanes, con educación universitaria, sin productos de ahorro ni inversión.



# 2 Análisis exploratorio de una bivariante 

## Num VS Num

cor(data[, varNum],use = "pairwise.complete.obs")
corPlot(data[, varNum])

# Correlacion negativa entre ComplaintsCount i NetPromoterScore -0.69
# Correlación negativa entre Balance i NumOfProducts de -0.31.
# El resto tienen un valor muy bajo en valor absoluto.

## Cat VS Cat

# taules fetes amb el 100% per columna. És a dir, del grup 1, A+B+C sumen 100, del grup 2 A+B+C=100%.

for (varc1 in varCat) {
  for (varc2 in varCat) {
    if (varc1 != varc2) {
      prop_table <- prop.table(table(data[, varc1], data[, varc2]),margin = 2)
      cat("=============", varc1, " vs. ", varc2, "=========================\n")
      print(prop_table)
    }
  }
}

par(mfrow = c(2, 2))  
for (varc1 in varCat) {
  for (varc2 in varCat) {
    if (varc1 != varc2) {
      prop_table <- prop.table(table(data[, varc1], data[, varc2]),margin =2 )
      barplot(prop_table, beside = TRUE,main = paste0(varc1,"&",varc2))
    }
  }
}

# Desbalanceig entre les combinacions de MaritalStatus, 
# LoanSatatus,Education level, SavingAccount i CustumerSegem.
# no s'observa res rellevant. Masses combinacions.
# Considero que no es tan important estudiar les relaciones entre les varibales.

# relacion entre exited (var resposta) i las categoricas
par(mfrow = c(3, 3)) 
cat<-varCat[-10]
v<-"Exited"
for (varc1 in cat) {
    if (varc1 != v) {
      prop_table <- prop.table(table(data[, v], data[, varc1]),margin = 2)
      print(prop_table)
      barplot(prop_table, beside = TRUE,main = paste0(v,"&",varc1))
    }
}

for (varc1 in cat) {
  if (varc1 != v) {
    tab <- table(data[[v]], data[[varc1]])  # filas = Exited, columnas = categorías
    test <- chisq.test(tab, correct = FALSE)
    
    cat("\nVariable:", varc1, "\n")
    cat("Chi-squared =", round(test$statistic, 3),
        "df =", test$parameter,
        "p-value =", signif(test$p.value, 5), "\n")
    
    if (test$p.value < 0.05) cat("-> Diferencias significativas entre columnas\n")
  }
}

# demostrat pel test.
# Els d'origen alemany tenen 1/3 de prob de marxar.
# Les dones amb mes prob.
# Els membres no actius tenen una prob més elevada.

## Num VS Cat

# todas las numericas con exited ( var resposta)

for (var in varNum) {
  p <- ggplot(data, aes(x = factor(Exited), y = .data[[var]], fill = factor(Exited))) +
    geom_boxplot(alpha = 0.7, na.rm = TRUE) +
    labs(x = "Exited", 
         y = var,
         title = paste("Distribución de", var, "por Exited")) +
    scale_fill_manual(values = c("#66CC99", "#FF6666"), 
                      name = "Exited", 
                      labels = c("No", "Sí")) +
    theme_minimal()
  
  print(p)
}

# Misma dsitribución tanto para los que marchan como los que no.
# Salvo la variable age,creditscore,num of products, Balance,Estimated salary.

# test de medianas

resultados_mediana <- data.frame(
  Variable = character(),
  Mediana_Exited0 = numeric(),
  Mediana_Exited1 = numeric(),
  p_value = numeric(),
  stringsAsFactors = FALSE
)

for (var in varNum) {
  medianas <- tapply(data[[var]], data$Exited, median, na.rm = TRUE)
  p_val <- wilcox.test(data[[var]] ~ data$Exited)$p.value
  
  resultados_mediana <- rbind(resultados_mediana, data.frame(
    Variable = var,
    Mediana_Exited0 = medianas["0"],
    Mediana_Exited1 = medianas["1"],
    p_value = p_val
  ))
}

resultados_mediana

# mismos resultados que antes.

# Missings

# Se sabe que los missings són aleatorios por los profes

library(visdat)

vis_miss(data)

# hay un 30% de missings en cada variable
#el id con un 21%, debido que en el test no exite missings.

na_por_fila <- rowSums(is.na(data))
filas_miss <- which(na_por_fila >= 15)
s<-data[filas_miss,]
sum(s$group=="test") # hay 212 individuos con 11 o mas missings


## Aleatoriedad de los missings
install.packages("naniar")
library(naniar)
naniar::mcar_test(data)
# p-valor = 0.387 > 0.05, por lo que no se rechaza H0: los missings son aleatorios (MCAR)
