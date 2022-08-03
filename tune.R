
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


# Variáveis:

library(tfruns)

FLAGS <- flags(
  flag_integer('hidden_layer_1', 5, 'Size of the first hidden layer'),
  flag_integer("hidden_layer_2", 0, "Size of the second hidden layer")
)

# Definir o modelo:

library(keras)
library(tensorflow)

input <- layer_input(shape = shape(30))
output <- input %>%
  layer_dense(units = FLAGS$hidden_layer_1, activation = "relu")

if (FLAGS$hidden_layer_2 > 0) {
  output <- output %>%
    layer_dense(units = FLAGS$hidden_layer_2, activation = "relu")
}

output <- output %>%
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
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(patience = 5)
    )
  )

