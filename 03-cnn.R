# Carregando os dados

library(keras)
library(tensorflow)

arquivos <- fs::dir_ls("dados/images/", glob = "*.jpg")
classes <- arquivos %>%
  fs::path_file() %>%
  stringr::str_extract("(.*)_") %>%
  stringr::str_sub(end = -2)

all_class <- unique(classes)
classes_int <- match(classes, all_class) - 1L

library(tfdatasets)

make_dataset <- function(arquivos, classes_int) {
  tensor_slices_dataset(list(arq = arquivos, classe = classes_int)) %>%
    dataset_map(function(x) {
      img <- x$arq %>%
        tf$io$read_file() %>%
        tf$image$decode_jpeg(channels = 3) %>%
        tf$image$resize(c(32L, 32L)) %>%
        tf$image$convert_image_dtype(tf$float32)
      list(img, x$classe)
    }, num_parallel_calls = tf$data$AUTOTUNE) %>%
    dataset_batch(32) %>%
    dataset_prefetch(tf$data$AUTOTUNE)
}


id_train <- sample.int(length(arquivos), 0.8*length(arquivos))

train_dataset <- make_dataset(
  arquivos = arquivos[id_train],
  classes_int = classes_int[id_train]
)

valid_dataset <- make_dataset(
  arquivos = arquivos[-id_train],
  classes_int = classes_int[-id_train]
)

coro::collect(train_dataset, 2)[[1]][[1]][4,,,] %>%
  as.array() %>%
  as.raster(max = 255) %>%
  plot()

# Construir o modelo

input <- layer_input(shape = shape(32, 32, 3))
output <- input %>%
  layer_rescaling(scale = 1/255) %>%

  layer_conv_2d(
    filter = 16, kernel_size = c(3,3), padding = "same",
    input_shape = c(32, 32, 3)
  ) %>%
  layer_activation_leaky_relu(0.1) %>%

  # Second hidden layer
  layer_conv_2d(filter = 32, kernel_size = c(3,3)) %>%
  layer_activation_leaky_relu(0.1) %>%

  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%

  # 2 additional hidden 2D convolutional layers
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_conv_2d(filter = 64, kernel_size = c(3,3)) %>%
  layer_activation_leaky_relu(0.1) %>%

  # Use max pooling once more
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%

  # Flatten max filtered output into feature vector
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(256) %>%
  layer_activation_leaky_relu(0.1) %>%
  layer_dropout(0.5) %>%

  # Outputs from dense layer are projected onto 10 unit output layer
  layer_dense(37)

model <- keras_model(input, output)
model %>%
  compile(
    loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
    optimizer = "adam",
    metrics = "accuracy"
  )


model %>%
  fit(
    train_dataset,
    validation_data = valid_dataset
  )
