load("ploter_hw7.Rdata")

df = read_delim("hw7_train.csv", delim = ",")

splitted=initial_split(df,prop=0.75, strata = decision)
train_df=training(splitted)
test_df=testing(splitted)

# Use the loaded prediction function for inference
predicted_classes <- my_predictions(fitted_workflow, new_data = test_df)

# Print the predictions
print(predicted_classes)
