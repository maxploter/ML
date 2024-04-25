import argparse
import torch

from transformers import AutoModelForTokenClassification, AutoTokenizer

# Task 4
# load your trained model and its tokenizer from the local folder
output_dir = 'results'
model = AutoModelForTokenClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)


def merge_subwords_and_tags(tokens, labels):
    merged_tokens = []
    merged_labels = []
    temp_token = ""
    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            temp_token += token[2:]
        else:
            if temp_token:  # Push the previous token to the list
                merged_tokens.append(temp_token)
                merged_labels.append(merged_label)
            temp_token = token
            print(label)
            merged_label = label if label.startswith("B-") else label
    if temp_token:  # Don't forget to add the last token
        merged_tokens.append(temp_token)
        merged_labels.append(merged_label)
    return merged_tokens, merged_labels


def combine_entities(tokens, labels):
    combined_entities = []
    current_entity = ""
    current_tag = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:  # Save the current entity
                combined_entities.append((current_entity, current_tag))
            current_entity = token  # Start new entity
            current_tag = label[2:]  # Remove the B- prefix
        elif label.startswith("I-") and current_tag == label[2:]:
            current_entity += " " + token  # Continue the entity
        else:
            if current_entity:  # Save the current entity
                combined_entities.append((current_entity, current_tag))
                current_entity = ""
                current_tag = None
            if label == "O":  # Single token not part of an entity
                combined_entities.append((token, "O"))
    if current_entity:  # Final entity
        combined_entities.append((current_entity, current_tag))
    return combined_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input")  # positional argument
    args = parser.parse_args()

    # Task: tag an arbitrary user input string
    # Example:
    #   Input: "Some Company did something"
    #   Output: [("Some Company", "ORG"), ("did", "0"), ("something", "0")]
    # 1. tokenize the input (the return format should be pytorch)
    # 2. pass the tokenized input through the model
    # 3. get the tag predictions for each token using argmax
    # 4. using `id2label` from the model's config (model.config) convert the predicted indices into labels (i.e. NER tags)
    # 5. convert indices (`input_ids`) in the tokenizer's output to tokens using `convert_ids_to_tokens`
    # 6. align the tokens and the NER tags:
    #   6.1 the output should be a list of tuples: [(word, tag), ...]
    #   6.2 the tokenizer splits the input text into subword units, so you should merge them back into words
    #       parts of a bigger word can be identified by `##` in front of the token
    #       the first part of a word consisting of subword units doesn't have `##` in front of it
    #    Example: ["Some", "Comp", "##any"], ["B-ORG", "I-ORG", "0"] -> [("Some", "B-ORG"), ("Company", "I-ORG")]
    # 7. some named entities consist of several words: the beginning is indicated by "B-" in the name of the tag,
    #    while "I-" means the continuation of the previous entity
    #    combine such sequences into a single tuple (words, tag)
    #    Example: [("Some", "B-ORG"), ("Company", "I-ORG")] -> [("Some Company", "ORG")]
    # 8. filter out "[CLS]" and "[SEP]" tokens from the output
    # 9. print out the input and the tagged output

    inputs = tokenizer(args.user_input, return_tensors="pt")
    outputs = model(inputs["input_ids"])
    predictions = torch.argmax(outputs.logits, dim=2)
    labels = [model.config.id2label[prediction.item()] for prediction in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    assert len(tokens) == len(labels)

    tokens_labels = [(token, label) for token, label in zip(tokens, labels) if token not in ["[CLS]", "[SEP]"]]
    merged_tokens, merged_labels = merge_subwords_and_tags(*zip(*tokens_labels))
    combined_entities = combine_entities(merged_tokens, merged_labels)

    # Print out the input and the tagged output
    print("Input:", args.user_input)
    print("Tagged Output:", combined_entities)