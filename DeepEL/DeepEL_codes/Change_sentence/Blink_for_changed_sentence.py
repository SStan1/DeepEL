import blink.main_dense as main_dense
import argparse
import os
import json
import jsonlines
from tqdm import tqdm
from DeepEL.dataset_reader import dataset_loader
import torch

torch.cuda.set_device(0)

def parse_args():
    parser = argparse.ArgumentParser(
        description='1st step to collect blink entity candidates for entity disambiguation.',
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        help="the extension file used by load_dataset function to load dataset",
        choices=["jsonl", "tsv", "oke_2015", "oke_2016", "n3", "xml", "unseen_mentions"],
        default="tsv",
        type=str,
    )
    parser.add_argument(
        "--key",
        help="the split key of aida-conll dataset",
        choices=["", "testa", "testb"],
        default="",
        type=str,
    )
    parser.add_argument(
        "--input_file",
        help="the dataset file used by load_dataset to load dataset",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="output directory",
        default="",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        help="output file",
        default="KORE50.json",
        type=str,
    )
    parser.add_argument(
        "--num_context_characters",
        help="maximum number of characters of original input sentence around mention",
        default=150,
        type=int,
    )
    parser.add_argument(
        "--blink_models_path",
        help="blink model path, must ends with /",
        default="",
        type=str,
    )
    parser.add_argument(
        "--blink_num_candidates",
        help="number of entity candidates for blink model",
        default=10,
        type=int,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file = os.path.join(args.output_dir, args.output_file)
    assert os.path.isfile(args.input_file)
    return args


def main():
    args = parse_args()
    models_path = args.blink_models_path
    config = {
        "test_entities": None,
        "test_mentions": None,
        "interactive": False,
        "top_k": args.blink_num_candidates,
        "biencoder_model": models_path + "biencoder_wiki_large.bin",
        "biencoder_config": models_path + "biencoder_wiki_large.json",
        "entity_catalogue": models_path + "entity.jsonl",
        "entity_encoding": models_path + "all_entities_large.t7",
        "crossencoder_model": models_path + "crossencoder_wiki_large.bin",
        "crossencoder_config": models_path + "crossencoder_wiki_large.json",
        "fast": False,
        "output_path": "logs/"
    }

    blink_args = argparse.Namespace(**config)
    models = main_dense.load_models(blink_args, logger=None)

    (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = models

    input_file = args.input_file
    with open(input_file) as reader:
        doc_name2instance = json.load(reader)

    output_file = args.output_file
    if os.path.isfile(output_file):
        with open(output_file) as reader:
            existing_data = json.load(reader)
        processed_docs = set(existing_data.keys())
    else:
        existing_data = {}
        processed_docs = set()

    num_context_characters = args.num_context_characters
    max_num_entity_candidates = args.blink_num_candidates

    for doc_name, instance in tqdm(doc_name2instance.items()):
        if doc_name in processed_docs:
            continue

        sentence = instance['sentence']
        entities = instance['entities']
        entity_candidates_list = []

        for (
                start,
                end,
                entity_mention,
                entity_name,
                prompt_results,
        ) in zip(
            entities['starts'],
            entities['ends'],
            entities['entity_mentions'],
            entities['entity_names'],
            entities['prompt_results']
        ):
            right_context = prompt_results[:min(len(prompt_results), num_context_characters)]
            left_context = ""
            data_to_link = [
                {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": left_context,
                    "mention": entity_mention,
                    "context_right": right_context,
                },
            ]
            _, _, _, _, _, predictions, scores, = main_dense.run(blink_args, None, *models, test_data=data_to_link)
            entity_candidates = predictions[0][:max_num_entity_candidates]
            entity_candidates_list.append(entity_candidates)

        doc_name2instance[doc_name]['entities']['blink_entity_candidates_list'] = entity_candidates_list
        existing_data[doc_name] = doc_name2instance[doc_name]

        # Incremental saving after each document is processed
        with open(output_file, 'w') as writer:
            json.dump(existing_data, writer, indent=4)


if __name__ == '__main__':
    main()
