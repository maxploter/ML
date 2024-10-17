library(tidyverse)
library(tidymodels)
library(GGally)

prefix <- "/Users/maksimploter/projects/ML/statistical-ml/labs/lab01/Datasets of Lab 1-20240905/"

auto_data <- read_delim(paste0(prefix, "Auto.csv"), delim = ",")

azn_data <- read_delim(paste0(prefix, "AZN.csv"), delim = ",", skip=1)

amzn_data <- read_delim(paste0(prefix, "AMZN.csv"), delim = "\t",
                        n_max=253, 
                        na = c("NA", ".."),
                        locale = locale(decimal_mark = ','))

amzn_data <- amzn_data |> mutate(Date = as.Date(Date, format = "%Y-%m-%d"))

summary(auto_data)

shoesize_data <- read_delim(paste0(prefix, "shoesize.txt"), delim = "\t", 
                            locale = locale(decimal_mark = ','))

summary(shoesize_data)


ggplot(auto_data)+geom_point(aes(x=weight,y=mpg, color=factor(cylinders)))

ggplot(auto_data)+geom_boxplot(aes(y=mpg, x=factor(cylinders)))

ggpairs(shoesize_data, progress=FALSE)


stock_prices=azn_data |> left_join(amzn_data, by="Date", suffix=c(".AZN", ".AMZN"))                             


auto_train=auto_data |> filter(row_number()!=n())
auto_newx=auto_data |> filter(row_number()==n()) |> select(-mpg)
