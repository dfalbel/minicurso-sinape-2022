
library(tfruns)

runs <- list()

for (i in 1:5) {
  runs[[i]] <- tuning_run(
    "tune.R",
    flags = list(
      hidden_layer_1 = c(5, 10, 15),
      hidden_layer_2 = c(0, 5, 10, 15)
    ),
    confirm = FALSE
  )
}

library(tidyverse)

runs %>%
  bind_rows() %>%
  group_by(flag_hidden_layer_1, flag_hidden_layer_2) %>%
  summarise(
    mean = mean(metric_val_accuracy),
    sd = sd(metric_val_accuracy)
  ) %>%
  arrange(desc(mean))
