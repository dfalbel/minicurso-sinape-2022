
# Dados daqui: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

# Ler dados

col_names <- janitor::make_clean_names(c("Clump Thickness",
"Uniformity of Cell Size",
"Uniformity of Cell Shape",
"Marginal Adhesion",
"Single Epithelial Cell Size",
"Bare Nuclei",
"Bland Chromatin",
"Normal Nucleoli",
"Mitoses",
"Class"
))

col_names <- c(
  "id", "diag",
  paste0(col_names, "_mean"),
  paste0(col_names, "_se"),
  paste0(col_names, "_worst")
)

data <- readr::read_csv("dados/wdbc.data",
                col_names = col_names, na = "?")

# Definir o modelo:

library(keras)
library(tensorflow)

input <- layer_input(shape = shape(30))
output <- input %>%
  layer_dense(units = 5, activation = "relu") %>%
  layer_dense(units = 1)
model <- keras_model(input, output)


model %>% compile(
  loss = loss_binary_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(),
  metrics = "accuracy"
)

# Ajustar o modelo

model %>%
  fit(
    x = as.matrix(data[-c(1,2)]),
    y = as.numeric(data$diag == "M"),
    batch_size = 10,
    epochs = 100,
    validation_split = 0.2
  )

get_weights(model)
predict(model, as.matrix(data[-c(1,2)]))
