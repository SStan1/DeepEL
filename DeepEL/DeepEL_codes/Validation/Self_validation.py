import json
import argparse
import os
import openai
import time
import re
from tqdm import tqdm

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process and validate entity replacements')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--output_file', type=str, required=True, help='Output file name')
    parser.add_argument('--api_base', type=str, default='', help='API base URL')
    parser.add_argument('--api_key', type=str, default='', help='API key')
    
    args = parser.parse_args()
    
    # Set up API configuration
    openai.api_base = args.api_base
    openai.api_key = args.api_key
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_file_path = os.path.join(args.output_dir, args.output_file)
    
    # Read the input JSON data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Step 1: Process entities and replace them in sentences
    processed_data = process_and_replace_entities(data)
    
    # Step 2: Validate replacements using LLM
    validation_results = validate_replacements(processed_data)
    
    # Step 3: Save final results
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=4)
    
    print(f"Processing complete. Results saved to '{output_file_path}'.")

def process_and_replace_entities(data):
    """
    Process entities and create new sentences with replacements
    """
    for doc_key, doc_value in data.items():
        sentence = doc_value['sentence']
        entities = doc_value['entities']
        starts = entities['starts']
        ends = entities['ends']
        entity_mentions = entities['entity_mentions']
        processed_entity_names = entities['entity_names']
        predict_entity_names = entities.get('predict_entity_names', [])
        
        num_entities = len(entity_mentions)
        
        # Ensure predict_entity_names and processed_entity_names have the same length
        if len(predict_entity_names) < num_entities:
            predict_entity_names.extend([''] * (num_entities - len(predict_entity_names)))
        if len(processed_entity_names) < num_entities:
            processed_entity_names.extend([''] * (num_entities - len(processed_entity_names)))
        
        # Build new sentence with replacements
        new_sentence_parts = []
        last_idx = 0
        for idx in range(num_entities):
            start = starts[idx]
            end = ends[idx]
            new_sentence_parts.append(sentence[last_idx:start])
            
            if processed_entity_names[idx]:
                new_entity_name = predict_entity_names[idx] if predict_entity_names[idx] else entity_mentions[idx]
                new_sentence_parts.append(new_entity_name)
            else:
                new_sentence_parts.append(entity_mentions[idx])
            
            last_idx = end
        
        new_sentence_parts.append(sentence[last_idx:])
        doc_value['new_sentence'] = ''.join(new_sentence_parts)
    
    return data

def validate_replacements(data):
    """
    Validate entity replacements using LLM
    """
    llm_correct = 0
    llm_incorrect = 0
    total_cases = 0
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0
    
    def retry_request(prompt, retries=100, delay=5):
        """Retry OpenAI API call"""
        for _ in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompt}],
                )
                return response['choices'][0]['message']['content'].strip()
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")
                time.sleep(delay)
        print("Max retries reached, skipping this entity.")
        return None
    
    for doc_key, doc_value in tqdm(data.items(), desc='Validating replacements', total=len(data)):
        sentence = doc_value['sentence']
        new_sentence = doc_value['new_sentence']
        entity_mentions = doc_value['entities']['entity_mentions']
        processed_entity_names = doc_value['entities'].get('processed_entity_names', [])
        predict_entity_names = doc_value['entities'].get('predict_entity_names', [])
        entity_candidates_descriptions = doc_value['entities'].get('entity_candidates_descriptions', [])
        multi_choice_prompt_results = doc_value['entities'].get('multi_choice_prompt_results', [])
        
        # Ensure consistent lengths
        max_len = max(len(processed_entity_names), len(predict_entity_names))
        processed_entity_names.extend([''] * (max_len - len(processed_entity_names)))
        predict_entity_names.extend([''] * (max_len - len(predict_entity_names)))
        
        # Map predicted entity names to descriptions
        entity_descriptions = {}
        for idx, result in enumerate(multi_choice_prompt_results):
            match = re.search(r'\d+', result)
            if match:
                choice_idx = int(match.group(0)) - 1
                if 0 <= choice_idx < len(entity_candidates_descriptions[idx]):
                    description = entity_candidates_descriptions[idx][choice_idx]
                    predicted_entity = predict_entity_names[idx]
                    entity_descriptions[predicted_entity] = description
            else:
                continue
        
        # Validate each entity
        validation_data = []
        for idx, predicted_entity in enumerate(predict_entity_names):
            if not predicted_entity or not processed_entity_names[idx]:
                validation_data.append({
                    'entity': '',
                    'validation_prompt': '',
                    'validation_reply': '',
                    'validation_result': 'Yes'
                })
                continue
            
            original_entity = entity_mentions[idx]
            is_replacement_correct = (processed_entity_names[idx] == predicted_entity) if (processed_entity_names[idx] and predicted_entity) else False
            
            description = entity_descriptions.get(predicted_entity, "No description available.")
            prompt = f"""
Original sentence: {sentence}
Sentence after replacement: {new_sentence}

Please judge whether the entity '{predicted_entity}' in the new sentence ('Sentence after replacement')
correctly refers to the same entity as '{original_entity}' in the original sentence ('Original sentence').
Please base your judgment on the following entity descriptions in the sentence.
Answer "Yes" or "No" and briefly explain your reasoning.
If you are not sure about your answer, you should also state that.

Entities in the sentence:
{predicted_entity}: {description}
"""
            
            for entity_name in predict_entity_names:
                entity_desc = entity_descriptions.get(entity_name, "No description available.")
                prompt += f"\n- {entity_name}: {entity_desc}"
            
            llm_reply = retry_request(prompt)
            if not llm_reply:
                continue
            
            llm_judgment = "yes" in llm_reply.lower()
            
            validation_data.append({
                'entity': predicted_entity,
                'validation_prompt': prompt.strip(),
                'validation_reply': llm_reply.strip(),
                'validation_result': 'Yes' if llm_judgment else 'No'
            })
            
            # Update statistics
            if llm_judgment and is_replacement_correct:
                llm_correct += 1
                true_positives += 1
            elif not llm_judgment and not is_replacement_correct:
                llm_correct += 1
                true_negatives += 1
            elif llm_judgment and not is_replacement_correct:
                llm_incorrect += 1
                false_positives += 1
            elif not llm_judgment and is_replacement_correct:
                llm_incorrect += 1
                false_negatives += 1
            
            total_cases += 1
            time.sleep(3)
        
        doc_value['validation_data'] = validation_data
    
    # Print evaluation results
    print(f"\nTotal test cases: {total_cases}")
    print(f"LLM correct judgments: {llm_correct}")
    print(f" - True positives (correct replacements judged as correct): {true_positives}")
    print(f" - True negatives (incorrect replacements judged as incorrect): {true_negatives}")
    print(f"LLM incorrect judgments: {llm_incorrect}")
    print(f" - False positives (incorrect replacements judged as correct): {false_positives}")
    print(f" - False negatives (correct replacements judged as incorrect): {false_negatives}")
    accuracy = llm_correct / total_cases * 100 if total_cases > 0 else 0
    print(f"LLM accuracy: {accuracy:.2f}%")
    
    return data

if __name__ == '__main__':
    main()