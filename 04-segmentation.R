# Carregando os dados

library(keras)
library(tensorflow)
library(tfdatasets)

arquivos <- fs::dir_ls("dados/images/", glob = "*.jpg")
trimaps <- fs::path(
  "dados/annotations/trimaps/",
  fs::path_file(arquivos)
)
fs::path_ext(trimaps) <- "png"

img_size <- c(32L, 32L)

read_trimap <- function(trimap) {
  tr <- trimap %>%
    tf$io$read_file() %>%
    tf$image$decode_png() %>%
    tf$image$resize(img_size)
  tr - 1L
}

display_target <- function(target_array) {
  normalized_array <- (target_array) * 127
  normalized_array <- tf$image$grayscale_to_rgb(as_tensor(normalized_array))
  normalized_array <- as.raster(as.array(normalized_array), max = 255)
  plot(normalized_array)
}

display_target(as.array(read_trimap(trimaps[1])))

read_img <- function(arq) {
  arq %>%
    tf$io$read_file() %>%
    tf$image$decode_jpeg(channels = 3) %>%
    tf$image$resize(img_size) %>%
    tf$image$convert_image_dtype(tf$float32)
}

read_img(arquivos[2]) %>%
  as.array() %>%
  as.raster(max = 255) %>%
  plot()


make_dataset <- function(arquivos, trimaps) {
  list(arquivos = arquivos, trimaps = trimaps) %>%
    tensor_slices_dataset() %>%
    dataset_map(function(x) {
      list(
        read_img(x$arquivos),
        read_trimap(x$trimaps)
      )
    }, num_parallel_calls = tf$data$AUTOTUNE) %>%
    dataset_batch(32) %>%
    dataset_prefetch(tf$data$AUTOTUNE)
}

id_train <- sample.int(length(arquivos), 0.8*length(arquivos))

train_dataset <- make_dataset(
  arquivos = arquivos[id_train],
  trimaps = trimaps[id_train]
)

valid_dataset <- make_dataset(
  arquivos = arquivos[-id_train],
  trimaps = trimaps[-id_train]
)

coro::collect(train_dataset, 2)[[1]][[1]][4,,,] %>%
  as.array() %>%
  as.raster(max = 255) %>%
  plot()

# Construir o modelo

get_model <- function(img_size, num_classes) {
  input <- layer_input(shape = c(img_size, 3))
  output <- input %>%
    layer_rescaling(scale = 1/255) %>%
    layer_conv_2d(filters = 64, kernel_size = 3, strides = 2, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 128, kernel_size = 3, strides = 2, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 256, kernel_size = 3, strides = 2, activation = "relu",
                  padding = "same") %>%
    layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu",
                  padding = "same") %>%


    layer_conv_2d_transpose(filters = 256, kernel_size = 3, activation = "relu",
                            padding = "same") %>%
    layer_conv_2d_transpose(filters = 256, kernel_size = 3, activation = "relu",
                            padding = "same", strides = 2) %>%
    layer_conv_2d_transpose(filters = 128, kernel_size = 3, activation = "relu",
                            padding = "same") %>%
    layer_conv_2d_transpose(filters = 128, kernel_size = 3, activation = "relu",
                            padding = "same", strides = 2) %>%
    layer_conv_2d_transpose(filters = 64, kernel_size = 3, activation = "relu",
                            padding = "same") %>%
    layer_conv_2d_transpose(filters = 64, kernel_size = 3, activation = "relu",
                            padding = "same", strides = 2) %>%

    layer_conv_2d(num_classes, 3, activation="softmax", padding="same")


  keras_model(input, output)
}

model <- get_model(img_size=img_size, num_classes=3)
model

model %>%
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "rmsprop"
  )


model %>%
  fit(
    train_dataset,
    validation_data = valid_dataset
  )
