library(keras)
library(tensorflow)
library(tfdatasets)
library(raster)

# Carregando dados

mnist <- dataset_mnist()

make_generator <- function() {
  input_noise <- layer_input(shape = shape(100))
  input_class <- layer_input(shape = shape(), dtype = "int64")

  latent_class <- input_class %>%
    layer_embedding(input_dim = 10, output_dim = 100)

  latent_noise <- input_noise %>%
    layer_dense(units = 100, activation = "relu")

  latent_input <- layer_concatenate(list(latent_noise, latent_class))

  output <- latent_input %>%
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

  keras_model(list(input_noise, input_class), output)
}

gen <- make_generator()


generate_image <- function(generator, class) {
  noise <- rnorm(100)
  generated_image <- predict(generator, list(matrix(noise, nrow = 1), class))
  plot(raster(generated_image[1,,,1]))
  generated_image
}

generate_image(gen, 1L)

make_discriminator <- function() {
  input <- layer_input(shape = shape(28, 28, 1))
  vector <- input %>%
    layer_conv_2d(64, c(5,5), strides = c(2,2), padding = "same") %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%

    layer_conv_2d(128, c(5,5), strides = c(2,2), padding = "same") %>%
    layer_activation_leaky_relu() %>%
    layer_dropout(0.3) %>%

    layer_flatten()


  prob <- vector %>%
    layer_dense(1)

  class <- vector %>%
    layer_dense(10)

  keras_model(input, list(prob = prob, class = class))
}

disc <- make_discriminator()
decision <- disc(generate_image(gen, 1))
decision

cross_entropy <- loss_binary_crossentropy(from_logits = TRUE)

discriminator_loss <- function(real_output, fake_output) {
  real_loss <- cross_entropy(tf$ones_like(real_output), real_output)
  fake_loss <- cross_entropy(tf$zeros_like(fake_output), fake_output)
  total_loss <- real_loss + fake_loss
  total_loss
}

classification_loss <- function(target_class, predicted) {
  loss_sparse_categorical_crossentropy(target_class, predicted, from_logits = TRUE)
}

generator_loss <- function(fake_output) {
  cross_entropy(tf$ones_like(fake_output), fake_output)
}

ACGAN <- new_model_class(
  "ACGAN",
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
  train_step = function(data) {
    images <- data[[1]]
    images_class <- data[[2]]

    noise <- tf$random$normal(shape(256, 100))
    class <- tf$random$uniform(shape(256), minval = 0L, maxval = 9L,  dtype = "int64")

    with(tf$GradientTape() %as% tape, {
      generated_images <- self$generator(list(noise, class), training=TRUE)
      fake_output <- self$discriminator(generated_images, training=TRUE)
      gen_loss <-
        generator_loss(fake_output$prob) +
        classification_loss(class, fake_output$class)
    })

    grads <- tape$gradient(gen_loss, self$generator$trainable_variables)
    self$g_optimizer$apply_gradients(
      zip_lists(grads, self$generator$trainable_variables)
    )

    with(tf$GradientTape() %as% tape, {
      generated_images <- self$generator(list(noise, class), training=TRUE)
      fake_output <- self$discriminator(generated_images, training=TRUE)
      real_output <- self$discriminator(images, training=TRUE)
      disc_loss <- discriminator_loss(real_output$prob, fake_output$prob) +
        classification_loss(images_class, real_output$class)
    })

    grads <- tape$gradient(disc_loss, self$discriminator$trainable_variables)
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
    list(
      (x$x/255 - 0.5)*2,
      x$y
    )
  }) %>%
  dataset_shuffle(buffer_size = 10000) %>%
  dataset_batch(256) %>%
  dataset_prefetch()

model <- ACGAN()
model %>% compile()
model %>% fit(dataset, epochs = 50)

invisible(generate_image(model$generator, 6L))

save_model_tf(model, "acgan")
save_model_weights_tf(model, "acgan-checkpoint")



