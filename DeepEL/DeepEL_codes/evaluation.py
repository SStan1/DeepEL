import json
import os

input_dir_path = '/content/drive/MyDrive/FYP/formal_Experiment1/Result_for_validation'
output_file_path = os.path.join(input_dir_path, 'combined_results.txt')

# Initialize a variable to hold all evaluation results
all_evaluation_results = ''

# Iterate over all files in the directory
for filename in os.listdir(input_dir_path):
    if filename.endswith('.json'):
        input_file_path = os.path.join(input_dir_path, filename)

        # Initialize counters for this file
        llm_correct = 0        # Number of times the LLM's judgment was correct
        llm_incorrect = 0      # Number of times the LLM's judgment was incorrect
        total_cases = 0        # Total number of valid test cases (excluding empty ones)

        false_positives = 0    # LLM judged incorrect replacement as correct
        false_negatives = 0    # LLM judged correct replacement as incorrect

        true_positives = 0     # LLM correctly judged correct replacement as correct
        true_negatives = 0     # LLM correctly judged incorrect replacement as incorrect

        # Step 1: Read the input data
        with open(input_file_path, 'r', encoding='utf-8') as f:
             data = json.load(f)

        # Step 2: Process documents
        for doc_key, doc_value in data.items():
            sentence = doc_value['sentence']            # Original sentence
            # Sentence after replacement
            entity_mentions = doc_value['entities']['entity_mentions']  # Original sentence entities
            processed_entity_names = doc_value['entities'].get('processed_entity_names', [])
            predict_entity_names = doc_value['entities'].get('predict_entity_names', [])
            validation_data = doc_value.get('validation_data', [])

            # Ensure the lengths of processed_entity_names and predict_entity_names are consistent
            max_len = max(len(processed_entity_names), len(predict_entity_names))
            processed_entity_names.extend([''] * (max_len - len(processed_entity_names)))
            predict_entity_names.extend([''] * (max_len - len(predict_entity_names)))

            # For each validation item
            for item in validation_data:
                predicted_entity = item['entity']
                validation_result = item['validation_result']  # 'Yes' or 'No'
                llm_judgment = validation_result.strip().lower() == 'yes'

                # Find idx where predict_entity_names[idx] == predicted_entity
                try:
                    idx = predict_entity_names.index(predicted_entity)
                except ValueError:
                    print(f"Predicted entity '{predicted_entity}' not found in predict_entity_names.")
                    continue  # Skip this item

                # Get processed_entity
                processed_entity = processed_entity_names[idx]

                # Skip cases where either the predicted or processed entity is empty
                if not predicted_entity or not processed_entity:
                    continue  # This case is ignored

                # Determine if the replacement is correct
                is_replacement_correct = processed_entity == predicted_entity

                # Compare llm_judgment with is_replacement_correct
                if llm_judgment and is_replacement_correct:
                    llm_correct += 1
                    true_positives += 1  # Correct replacement judged as correct
                elif not llm_judgment and not is_replacement_correct:
                    llm_correct += 1
                    true_negatives += 1  # Incorrect replacement judged as incorrect
                elif llm_judgment and not is_replacement_correct:
                    llm_incorrect += 1
                    false_positives += 1  # Incorrect replacement judged as correct
                elif not llm_judgment and is_replacement_correct:
                    llm_incorrect += 1
                    false_negatives += 1  # Correct replacement judged as incorrect

                total_cases += 1  # Increment only for valid test cases (those that are not skipped)

        # Compute accuracy
        accuracy = llm_correct / total_cases * 100 if total_cases > 0 else 0

        # Prepare the evaluation results
        evaluation_results = f"Evaluation Results for file: {filename}\n"
        evaluation_results += f"Total test cases: {total_cases}\n"
        evaluation_results += f"LLM correct judgments: {llm_correct}\n"
        evaluation_results += f" - True positives (correct replacements judged as correct): {true_positives}\n"
        evaluation_results += f" - True negatives (incorrect replacements judged as incorrect): {true_negatives}\n"
        evaluation_results += f"LLM incorrect judgments: {llm_incorrect}\n"
        evaluation_results += f" - False positives (incorrect replacements judged as correct): {false_positives}\n"
        evaluation_results += f" - False negatives (correct replacements judged as incorrect): {false_negatives}\n"
        evaluation_results += f"LLM accuracy: {accuracy:.2f}%\n"
        evaluation_results += "-" * 50 + "\n"

        # Append the evaluation results to the combined string
        all_evaluation_results += evaluation_results

# After processing all files, write the combined results to a single output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(all_evaluation_results)

# Optionally, print the combined results
print(all_evaluation_results)
