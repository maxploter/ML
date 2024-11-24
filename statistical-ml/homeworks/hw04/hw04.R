library(tidymodels)
library(mgcv)
library(readr)

set.seed(39692)

df <- read_csv("hw4_train.csv")

rec <- recipe(y ~ ., data = df) |> 
  step_dummy(all_nominal_predictors())

df_baked <- rec |> prep(df) |> bake(new_data=NULL)

wf_gam_1 <- workflow() |>
  add_recipe(recipe(y ~ ., data = df_baked)) |>
  add_model(
    gen_additive_mod(mode = "regression"),
    formula = y ~ splines::bs(X1, knots = c(1.776703, 3.097543, 4.128191, 5.065609, 6.022864, 7.034800, 8.375370)) +
      splines::ns(X2, df = 6) + s(X9) + X7 + X6 + X14
  )

final_model <- fit(wf_gam_1, data = df_baked)

test_df <- read_csv("hw4_test.csv")

test_baked <- rec |> prep(df) |>  bake(new_data = test_df)

my_predictions <- predict(final_model, new_data = test_baked)$.pred

print(my_predictions)
