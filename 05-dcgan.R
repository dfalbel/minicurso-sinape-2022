library(keras)
library(tensorflow)
library(tfdatasets)
library(raster)

# Carregando dados

mnist <- dataset_mnist()

make_generator <- function() {
  input <- layer_input(shape = shape(100))

  output <- input %>%
    layer_dense(7*7*256, use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu() %>%

    layer_reshape(c(7, 7, 256)) %>%

    layer_conv_2d_transpose(128, c(5,5), strides = c(1,1), padding = "same",
                            use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu() %>%

    layer_conv_2d_transpose(64, c(5,5), strides = c(2,2), padding = "same",
                            use_bias = FALSE) %>%
    layer_batch_normalization() %>%
    layer_activation_leaky_relu() %>%

    layer_conv_2d_transpose(1, c(5,5), strides = c(2,2), padding = "same",
                            use_bias = FALSE, activation = "tanh")

  keras_model(input, output)
}

gen <- make_generator()

generate_image <- function(generator) {
  noise <- rnorm(100)
  generated_image <- predict(generator, matrix(noise, nrow = 1))
  plot(raster(generated_image[1,,,1]))
  generated_image
}

generate_image(gen)

make_discriminator <- function() {
  input <- layer_input(shape = shape(28, 28, 1))
  output <- input %>%
    layer_conv_2d(64, c(5,5), strides = c(2,2), padding = "same") %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%

    layer_conv_2d(128, c(5,5), strides = c(2,2), padding = "same") %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%

    layer_flatten() %>%
    layer_dense(1)

  keras_model(input, output)
}

disc <- make_discriminator()
decision <- disc(generate_image(gen))
tf$sigmoid(decision)

cross_entropy <- loss_binary_crossentropy(from_logits = TRUE)

discriminator_loss <- function(real_output, fake_output) {
  real_loss <- cross_entropy(tf$ones_like(real_output), real_output)
  fake_loss <- cross_entropy(tf$zeros_like(fake_output), fake_output)
  total_loss <- real_loss + fake_loss
  total_loss
}

generator_loss <- function(fake_output) {
  cross_entropy(tf$ones_like(fake_output), fake_output)
}

DCGAN <- new_model_class(
  "DCGAN",
  initialize = function() {
    super()$`__init__`()
    self$generator <- make_generator()
    self$discriminator <- make_discriminator()
  },
  compile = function() {
    super()$compile()
    self$g_optimizer <- optimizer_adam(1e-4)
    self$d_optimizer <- optimizer_adam(1e-4)

    self$d_loss_metric <- tf$keras$metrics$Mean(name = "d_loss")
    self$g_loss_metric <- tf$keras$metrics$Mean(name = "g_loss")
  },
  train_step = function(images) {
    noise <- tf$random$normal(shape(256, 100))

    with(tf$GradientTape() %as% tape, {
      generated_images <- self$generator(noise, training=TRUE)
      fake_output <- self$discriminator(generated_images, training=TRUE)
      gen_loss <- generator_loss(fake_output)
    })

    grads <- tape$gradient(gen_loss, self$generator$trainable_variables)
    self$g_optimizer$apply_gradients(
      zip_lists(grads, self$generator$trainable_variables)
    )

    with(tf$GradientTape() %as% tape, {
      generated_images <- self$generator(noise, training=TRUE)
      fake_output <- self$discriminator(generated_images, training=TRUE)
      real_output <- self$discriminator(images, training=TRUE)
      disc_loss <- discriminator_loss(real_output, fake_output)
    })

    grads <-tape$gradient(disc_loss, self$discriminator$trainable_variables)
    self$d_optimizer$apply_gradients(
      zip_lists(grads, self$discriminator$trainable_variables)
    )

    self$d_loss_metric$update_state(disc_loss)
    self$g_loss_metric$update_state(gen_loss)
    list(
      d_loss = self$d_loss_metric$result(),
      g_loss = self$g_loss_metric$result()
    )
  }
)

dataset <- tensor_slices_dataset(mnist$train) %>%
  dataset_map(function(x) {
    (x$x/255 - 0.5)*2
  }) %>%
  dataset_shuffle(buffer_size = 10000) %>%
  dataset_batch(256) %>%
  dataset_prefetch()

model <- DCGAN()
model %>% compile()
model %>% fit(dataset, epochs = 50)

invisible(generate_image(model$generator))
