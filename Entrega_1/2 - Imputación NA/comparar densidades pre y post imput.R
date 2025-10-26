density_before_after <- function(before, after) {
  require(ggplot2)
  require(dplyr)
  require(tidyr)
  require(purrr)
  
  # Identificar variables numéricas comunes
  numeric_vars_before <- before |> 
    select(where(is.numeric)) |>
    names()
  
  numeric_vars_after <- after |> 
    select(where(is.numeric)) |>
    names()
  
  # Variables comunes entre ambos datasets
  common_vars <- intersect(numeric_vars_before, numeric_vars_after)
  
  if (length(common_vars) == 0) {
    stop("No hay variables numéricas comunes entre los datasets")
  }
  
  # Crear datos combinados
  density_df <- map_dfr(common_vars, function(var) {
    data.frame(
      variable = var,
      value = c(before[[var]], after[[var]]),
      dataset = rep(c("Original", "Imputado"), 
                    c(nrow(before), nrow(after)))
    )
  }) |>
    filter(!is.na(value))
  
  # Crear el gráfico
  ggplot(density_df, aes(x = value, color = dataset, fill = dataset)) +
    facet_wrap(~variable, scales = "free") +
    geom_density(alpha = 0.3) +
    scale_color_manual(values = c("#1f77b4", "#ff7f0e")) +
    scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
    labs(title = "Comparación de Densidades: Original vs Imputado",
         x = "Valores", 
         y = "Densidad") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

mass_before_after <- function(before, after, max_categories = 10) {
  require(ggplot2)
  require(dplyr)
  require(tidyr)
  require(purrr)
  
  # Identificar variables categóricas comunes
  categorical_vars_before <- before |> 
    select(where(~ is.factor(.) | is.character(.))) |>
    names()
  
  categorical_vars_after <- after |> 
    select(where(~ is.factor(.) | is.character(.))) |>
    names()
  
  # Variables comunes entre ambos datasets
  common_vars <- intersect(categorical_vars_before, categorical_vars_after)
  
  if (length(common_vars) == 0) {
    stop("No hay variables categóricas comunes entre los datasets")
  }
  
  # Crear datos combinados
  mass_df <- map_dfr(common_vars, function(var) {
    
    # Convertir a factor si es character
    before_vec <- if(is.character(before[[var]])) {
      as.factor(before[[var]])
    } else {
      before[[var]]
    }
    
    after_vec <- if(is.character(after[[var]])) {
      as.factor(after[[var]])
    } else {
      after[[var]]
    }
    
    # Calcular proporciones
    prop_before <- proportions(table(before_vec))
    prop_after <- proportions(table(after_vec))
    
    data.frame(
      variable = var,
      category = c(names(prop_before), names(prop_after)),
      proportion = c(prop_before, prop_after),
      dataset = rep(c("Original", "Imputado"), 
                    c(length(prop_before), length(prop_after)))
    )
  }) |>
    filter(!is.na(category))
  
  # Filtrar categorías si hay demasiadas
  if (max_categories > 0) {
    mass_df <- mass_df |>
      group_by(variable) |>
      slice_max(proportion, n = max_categories) |>
      ungroup()
  }
  
  # Crear el gráfico
  ggplot(mass_df, aes(x = category, y = proportion, 
                      color = dataset, fill = dataset)) +
    facet_wrap(~variable, scales = "free") +
    geom_col(alpha = 0.6, width = 0.7, position = "dodge") +
    scale_color_manual(values = c("#1f77b4", "#ff7f0e")) +
    scale_fill_manual(values = c("#1f77b4", "#ff7f0e")) +
    labs(title = "Comparación de Distribuciones Categóricas: Original vs Imputado",
         x = "Categorías", 
         y = "Proporción") +
    theme_minimal() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1))
}