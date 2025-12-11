import os
import json
import time
import argparse
from tqdm import tqdm
import openai
from DeepEL.openai_key import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
openai.api_base = "https://api.chatnio.net/v1"
import re
from DeepEL.openai_function import openai_chatgpt, openai_completion

def extract_answer_from_output(output):
    match = re.search(r'\b(\d+)\b', output)
    return int(match.group(1)) if match else None

def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect prompt for entity information.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--input_file",
        default='',
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        default='',
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="ace2004.json",
        type=str,
    )
    parser.add_argument(
        "--num_entity_description_characters",
        default=150,
        type=int,
    )
    parser.add_argument(
        "--openai_mode",
        default='chatgpt',
        choices=['chatgpt', 'gpt'],
        type=str,
    )
    parser.add_argument(
        "--openai_model",
        default='gpt-3.5-turbo',
        choices=[
            'gpt-3.5-turbo',
            'text-curie-001',
            'text-davinci-003',
            'gpt-4',
            'ft:gpt-3.5-turbo-0613:amrit::8VNXmmdS',
        ],
        type=str,
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args

def main():
    args = parse_args()
    openai_model = args.openai_model
    openai_mode = args.openai_mode
    openai_function = openai_chatgpt if openai_mode == 'chatgpt' else openai_completion

    input_file = args.input_file
    output_file = args.output_file
    num_entity_description_characters = args.num_entity_description_characters

    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    if os.path.isfile(output_file):
        with open(output_file) as reader:
            exist_doc_name2instance = json.load(reader)
        exist_doc_names = set(exist_doc_name2instance.keys())
    else:
        exist_doc_name2instance = {}
        exist_doc_names = set()

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if doc_name in exist_doc_names and 'multi_choice_prompts' in exist_doc_name2instance[doc_name]['entities']:
            continue

        entities = instance['entities']
        multi_choice_prompts = entities.get('multi_choice_prompts', [])
        multi_choice_prompt_results = entities.get('multi_choice_prompt_results', [])

        validation_data = instance.get('validation_data', [])
        for validation in validation_data:
            if validation.get('validation_result') != "No":
                continue

            entity = validation['entity']


            # 根据实体名称获取索引
            entity_idx = entities['predict_entity_names'].index(entity)
            entity_mention = entities['entity_mentions'][entity_idx]
            validation_prompt = validation['validation_reply']
            prompt_result = entities['prompt_results'][entity_idx]

            combined_prompt = (
                f"Original explanation: {prompt_result.strip()}\n\n"
                f"{validation_prompt.strip()}\n\n"
            )

            entity_candidates = entities['entity_candidates'][entity_idx]
            entity_candidates_description = entities['entity_candidates_descriptions'][entity_idx]

            multi_choice_prompt = ''
            for index, (entity_candidate, entity_candidate_description) in enumerate(zip(entity_candidates, entity_candidates_description)):
                description = entity_candidate + ' ' + entity_candidate_description[:num_entity_description_characters]
                multi_choice_prompt += f'({index + 1}). {description}\n'

            multi_choice_prompt = (
                f"{combined_prompt}\n\nWhich of the following entities is {entity_mention} in this sentence? "
                f"Return a number to represent your answer. \n\n{multi_choice_prompt}"
            )

      
            complete_output = openai_function(multi_choice_prompt, model=openai_model)

     
            if entity_idx < len(multi_choice_prompt_results):
                multi_choice_prompt_results[entity_idx] = complete_output
            

            if entity_idx < len(multi_choice_prompts):
                multi_choice_prompts[entity_idx] = multi_choice_prompt
            


            validation['validation_result'] = "Yes"

        entities['multi_choice_prompts'] = multi_choice_prompts
        entities['multi_choice_prompt_results'] = multi_choice_prompt_results
        doc_name2instance[doc_name]['entities'] = entities
        exist_doc_name2instance[doc_name] = instance

        with open(output_file, 'w') as writer:
            json.dump(exist_doc_name2instance, writer, indent=4)


if __name__ == '__main__':
    main()


