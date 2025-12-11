import os
import json
import requests
import argparse
import jsonlines
import collections
import urllib
import rdflib
from tqdm import tqdm
from collections import defaultdict


def load_tsv(file='/nfs/yding4/EL_project/dataset/KORE50/AIDA.tsv', key='', mode='char'):
    """
    Read a file with a specific format (space- or tab-separated) and parse it into a dictionary
    (doc_name2dataset or char_doc_name2instance), which includes document name, tokens, entities, and other info.

    :param file: TSV file path
    :param key: keep only documents whose doc_name contains this key (pass an empty string to disable filtering)
    :param mode: 'token' or 'char'; determines whether entity offsets are returned at token level or character level.
    :return: dict, with doc_name as key and a dict containing 'tokens' / 'entities' and other info as value
    """

    def process_token_2_char_4_doc_name2instance(token_doc_name2instance):
        """
        Convert token-level entity indices (starts, ends) to character-level indices (starts, ends).
        """
        char_doc_name2instance = dict()

        for doc_name, instance in token_doc_name2instance.items():
            starts = []
            ends = []
            entity_mentions = []
            entity_names = []

            assert doc_name == instance['doc_name']
            tokens = instance['tokens']
            sentence = ' '.join(tokens)
            token_entities = instance['entities']

            for token_start, token_end, token_entity_mention, token_entity_name in zip(
                token_entities['starts'],
                token_entities['ends'],
                token_entities['entity_mentions'],
                token_entities['entity_names']
            ):
                # Check whether token_start and token_end fall within the valid range
                if not 0 <= token_start <= token_end < len(tokens):
                    print("[Warning] Token index out of range in doc:", doc_name)
                    print(instance)
                    continue

                # Convert token-level start/end to character-level
                if token_start == 0:
                    start = 0
                else:
                    start = len(' '.join(tokens[:token_start])) + 1

                end = len(' '.join(tokens[:token_end + 1]))
                entity_mention = sentence[start:end]

                starts.append(start)
                ends.append(end)
                entity_mentions.append(entity_mention)
                entity_names.append(token_entity_name)

            char_doc_name2instance[doc_name] = {
                'doc_name': doc_name,
                'sentence': sentence,
                'entities': {
                    "starts": starts,
                    "ends": ends,
                    "entity_mentions": entity_mentions,
                    "entity_names": entity_names,
                }
            }

        return char_doc_name2instance

    def generate_instance(
        doc_name,
        tokens,
        ner_tags,
        entity_mentions,
        entity_names,
        entity_wikipedia_ids,
    ):
        """
        Merge tokens / ner_tags / entity information of a document into a single "instance".
        Combine B/I sequences into entities and produce lists of starts/ends.
        """
        assert len(tokens) == len(ner_tags) == len(entity_mentions) == len(entity_names) == len(entity_wikipedia_ids)

        instance_starts = []
        instance_ends = []
        instance_entity_mentions = []
        instance_entity_names = []
        instance_entity_wikipedia_ids = []

        tmp_start = -1

        for index, (ner_tag, entity_mention, entity_name, entity_wikipedia_id) in enumerate(
            zip(ner_tags, entity_mentions, entity_names, entity_wikipedia_ids)
        ):
            # Skip if tag is O
            if ner_tag == 'O':
                continue
            # If tag starts with B
            elif ner_tag.startswith('B'):
                # If the next tag is not I, this entity is a single token
                if index == len(tokens) - 1 or ner_tags[index + 1] == 'O' or ner_tags[index + 1].startswith('B'):
                    instance_starts.append(index)
                    instance_ends.append(index)
                    instance_entity_mentions.append(entity_mention)
                    instance_entity_names.append(entity_name)
                    instance_entity_wikipedia_ids.append(entity_wikipedia_id)
                    tmp_start = -1
                else:
                    # Otherwise start an entity span
                    tmp_start = index
            # If tag starts with I
            elif ner_tag.startswith('I'):
                # If this is the last token or the next tag is O/B, close the entity span
                if index == len(tokens) - 1 or ner_tags[index + 1] == 'O' or ner_tags[index + 1].startswith('B'):
                    # tmp_start should not be -1 if a matching B has been seen
                    if tmp_start == -1:
                        # If tmp_start == -1, treat this token as a standalone entity
                        tmp_start = index
                    instance_starts.append(tmp_start)
                    instance_ends.append(index)
                    instance_entity_mentions.append(entity_mention)
                    instance_entity_names.append(entity_name)
                    instance_entity_wikipedia_ids.append(entity_wikipedia_id)
                    tmp_start = -1
                else:
                    # Continue if the entity has not ended
                    continue
            else:
                # Encountered a tag that is not O/B/I; log warning and skip or treat as O
                print(f"[Warning] Unexpected ner_tag '{ner_tag}' in doc '{doc_name}', index={index}. Skipping token.")
                continue

        instance = {
            'doc_name': doc_name,
            'tokens': tokens,
            'entities': {
                "starts": instance_starts,
                "ends": instance_ends,
                "entity_mentions": instance_entity_mentions,
                "entity_names": instance_entity_names,
                "entity_wikipedia_ids": instance_entity_wikipedia_ids,
            }
        }
        return instance

    doc_name2dataset = dict()
    doc_name = ''
    tokens = []
    ner_tags = []
    entity_mentions = []
    entity_names = []
    entity_wikipedia_ids = []

    with open(file) as reader:
        for line in reader:
            # When encountering -DOCSTART- (start of a new document)
            if line.startswith('-DOCSTART-'):
                # Save the previous document instance first
                if tokens:
                    assert doc_name != ''
                    if doc_name not in doc_name2dataset:
                        instance = generate_instance(
                            doc_name,
                            tokens,
                            ner_tags,
                            entity_mentions,
                            entity_names,
                            entity_wikipedia_ids,
                        )
                        # Keep the document if key is in doc_name (or key == '')
                        if (key in doc_name) or (key == ''):
                            doc_name2dataset[doc_name] = instance

                # Reset all collected data
                tokens = []
                ner_tags = []
                entity_mentions = []
                entity_names = []
                entity_wikipedia_ids = []

                # Extract doc_name
                # Line typically looks like "-DOCSTART- (1)\n"
                # Extract the content inside parentheses
                tmp_start_index = len('-DOCSTART- (')
                if line.endswith(')\n'):
                    tmp_end_index = len(')\n')
                else:
                    tmp_end_index = len('\n')
                doc_name = line[tmp_start_index: -tmp_end_index].strip()

            # Skip empty lines
            elif line.strip() == '':
                continue

            else:
                # Normal token line
                # Use split() to support both spaces and tabs
                parts = line.strip().split()

                if len(parts) == 1:
                    # [token]
                    token = parts[0]
                    ner_tag = 'O'
                    # Keep mention and name empty
                    entity_mention = ''
                    entity_name = ''
                    wiki_id = -1

                elif len(parts) == 2:
                    # [token, ner_tag]
                    token = parts[0]
                    ner_tag = parts[1]
                    entity_mention = ''
                    entity_name = ''
                    wiki_id = -1

                elif len(parts) == 4:
                    # [token, B/I, entity_name, wiki_url]
                    # According to requirement: column 3 is entity_name
                    token = parts[0]
                    ner_tag = parts[1]

                    raw_name = parts[2].encode().decode("unicode-escape")
                    if raw_name == '--NME--':
                        entity_name = ''
                    else:
                        entity_name = raw_name

                    # Use token itself as entity mention
                    entity_mention = token

                    # Column 4: wiki_url, keep or ignore as needed
                    wiki_url = parts[3]

                    wiki_id = -1  # Use -1 if ID is unavailable

                elif len(parts) in [6, 7]:
                    # [token, B/I, entity_mention, entity_name, wiki_url, wiki_id, ...]
                    token = parts[0]
                    ner_tag = parts[1]
                    entity_mention = parts[2].encode().decode("unicode-escape")
                    raw_name = parts[3].encode().decode("unicode-escape")
                    if raw_name == '--NME--':
                        entity_name = ''
                    else:
                        entity_name = raw_name

                    if parts[5].isdigit() and int(parts[5]) > 0:
                        wiki_id = int(parts[5])
                    else:
                        wiki_id = -1

                else:
                    # Not matching any of the above formats
                    print(f"[Warning] Unexpected line format: {line.strip()}")
                    continue

                # Record data
                tokens.append(token)
                ner_tags.append(ner_tag)
                entity_mentions.append(entity_mention)
                entity_names.append(entity_name)
                entity_wikipedia_ids.append(wiki_id)

    # After reaching EOF, save the last document if needed
    if tokens:
        assert doc_name != ''
        if doc_name not in doc_name2dataset:
            instance = generate_instance(
                doc_name,
                tokens,
                ner_tags,
                entity_mentions,
                entity_names,
                entity_wikipedia_ids,
            )
            if (key in doc_name) or (key == ''):
                doc_name2dataset[doc_name] = instance

    # Return based on mode
    if mode == 'token':
        return doc_name2dataset
    else:
        assert mode == 'char', 'MODE(parameter) only supports "token" and "char"'
        return process_token_2_char_4_doc_name2instance(doc_name2dataset)

def load_ttl_oke_2015(
    file='/nfs/yding4/EL_project/dataset/oke-challenge/evaluation-data/task1/evaluation-dataset-task1.ttl',
):
    def process_sen_char(s):
        assert 'sentence-' in s
        first_parts = s.split('sentence-')
        assert len(first_parts) == 2
        assert '#char=' in first_parts[1]
        second_parts = first_parts[1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type',
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: links temporary annotations to dataset base if the KB has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()

    for node_index, node in enumerate(g):
        parts = node[1].split('#')
        assert len(parts) == 2
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            tmp_str = str(node[2]).rstrip()
            if (char_end - char_start) != len(tmp_str):
                # Only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert str(node[2]).count('sentence-') == 1
            tmp_entity = str(node[2]).split('sentence-')[1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity or \
                   sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] in ['Man_4', 'His_4']
            sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = tmp_entity

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            assert str(node[0]).count('sentence-') == 1
            mention = str(node[0]).split('sentence-')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
            # print(mention, entity)
            # if not str(node[2]).startswith('http://dbpedia.org/resource/'):
            #     print(node)
        else:
            if parts[1] == 'label':
                # 'label' is not useful
                tmp_split_str = str(node[0]).split('sentence-')[1]
                tmp_str = str(node[2])
                assert tmp_split_str == tmp_str.replace(' ', '_')

    num_in = 0
    num_out = 0

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')
        if processed_tmp_entity in tmp_entity2entity:
            num_in += 1
            entity = tmp_entity2entity[processed_tmp_entity]
            # assert (char_end - char_start) == len(tmp_str)
            mention = sentence[char_start: char_end]
            doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
            doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(entity)

        else:
            num_out += 1
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out}; len(tmp_entity2entity): {len(tmp_entity2entity)}')
    # print(json.dumps(doc_name2instance, indent=4))
    return doc_name2instance


def load_unseen_mentions(file='/nfs/yding4/EL_project/dataset/unseen_mentions/test.json'):
    doc_name2instance = dict()
    with open(file) as reader:
        for index, line in enumerate(reader):
            d = json.loads(line)
            # doc_name = str(d['docId'])
            doc_name = str(index)
            mention = ' '.join(d['mention_as_list'])
            entity = d['y_title']
            sentence = d['left_context_text'] + ' ' + mention + ' ' + d['right_context_text']
            start = len( d['left_context_text']) + 1
            end = start + len(mention)

            doc_name2instance[doc_name] = {
                'sentence': sentence,
                'entities': {
                    'starts': [start],
                    'ends': [end],
                    'entity_mentions': [mention],
                    'entity_names': [entity],
                }
            }
    
    return doc_name2instance


def load_ttl_oke_2016(
    file='/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl',
):
    def process_sen_char(s):
        assert 'sentence-' in s
        first_parts = s.split('sentence-')
        assert len(first_parts) == 2
        assert '#char=' in first_parts[1]
        second_parts = first_parts[1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type',
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: links temporary annotations to dataset base if the KB has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()

    for node_index, node in enumerate(g):
        parts = node[1].split('#')
        assert len(parts) == 2
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            tmp_str = str(node[2]).rstrip()
            if (char_end - char_start) != len(tmp_str):
                # Only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            # print(node)
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert str(node[2]).count('task-1/') == 1
            tmp_entity = str(node[2]).split('task-1/')[1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity or \
                   sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] in ['Man_4', 'His_4']
            sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = tmp_entity

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            # print(node)
            assert str(node[0]).count('task-1/') == 1
            mention = str(node[0]).split('task-1/')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
            # print(mention, entity)
            # if not str(node[2]).startswith('http://dbpedia.org/resource/'):
            #     print(node)
        else:
            assert parts[1] in ['beginIndex', 'label', 'endIndex', 'referenceContext', 'type']

    num_in = 0
    num_out = 0

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')
        if processed_tmp_entity in tmp_entity2entity:
            num_in += 1
            entity = tmp_entity2entity[processed_tmp_entity]
            # assert (char_end - char_start) == len(tmp_str)
            mention = sentence[char_start: char_end]
            doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
            doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
            doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(entity)

        else:
            num_out += 1
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out}; len(tmp_entity2entity): {len(tmp_entity2entity)}')
    return doc_name2instance


def load_ttl_n3(
    file='/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl',
):
    def process_sen_char(s):
        assert s.count('/') == 5
        first_parts = s.split('/')
        assert '#char=' in first_parts[-1]
        second_parts = first_parts[-1].split('#char=')
        assert len(second_parts) == 2
        assert ',' in second_parts[1]
        sentence_index = int(second_parts[0])
        third_parts = second_parts[1].split(',')
        assert len(third_parts) == 2
        char_start, char_end = int(third_parts[0]), int(third_parts[1])
        return char_start, char_end, sentence_index

    # file = '/nfs/yding4/EL_project/dataset/oke-challenge-2016/evaluation-data/task1/evaluation-dataset-task1.ttl'
    g = rdflib.Graph()
    g.parse(file, format='ttl')

    module_list = [
        'label', 'anchorOf', 'beginIndex', 'isString', 'sameAs', 'endIndex', 'taIdentRef', 'referenceContext', 'type', 'taSource', 'hasContext', 'sourceUrl'
    ]

    # 1. isString: extracts sentence (identified by the sentence number)
    # 2. taIdentRef: extracts mentions and labelled temporary annotations
    # 3. sameAs: links temporary annotations to dataset base if the KB has corresponding entities
    sentence_index2sentence = dict()
    sent_char_index2tmp_entity = dict()
    tmp_entity2entity = dict()
    num_in = 0
    num_out = 0

    for node_index, node in enumerate(g):
        # print(node)

        parts = node[1].split('#')
        assert len(parts) == 2
        if parts[1] not in module_list:
            print(str(parts[1]))
        assert parts[1] in module_list

        if parts[1] == 'anchorOf':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            # tmp_str = str(node[2]).rstrip()
            tmp_str = str(node[2])
            if (char_end - char_start) != len(tmp_str):
                # Only one data error: 'Basel, Switzerland'
                tmp_str = tmp_str.split(',')[0]
            assert (char_end - char_start) == len(tmp_str)

        elif parts[1] == 'taIdentRef':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert '/' in str(node[2])
            tmp_entity = str(node[2]).split('/')[-1]
            assert (sentence_index, char_start, char_end) not in sent_char_index2tmp_entity
            if 'notInWiki' in str(node[2]):
                num_out += 1
            else:
                num_in += 1
                assert str(node[2].startswith('http://dbpedia.org/resource/'))

                sent_char_index2tmp_entity[(sentence_index, char_start, char_end)] = urllib.parse.unquote(tmp_entity)

        elif parts[1] == 'isString':
            char_start, char_end, sentence_index = process_sen_char(str(node[0]))
            assert sentence_index not in sentence_index2sentence
            sentence_index2sentence[sentence_index] = str(node[2])

        elif parts[1] == 'sameAs':
            assert str(node[0]).count('sentence-') == 1
            mention = str(node[0]).split('sentence-')[1]

            entity = str(node[2]).split('/')[-1]
            if mention in tmp_entity2entity:
                assert entity == tmp_entity2entity[mention]
            tmp_entity2entity[mention] = entity
        else:
            if parts[1] == 'label':
                # 'label' is not useful
                tmp_split_str = str(node[0]).split('sentence-')[1]
                tmp_str = str(node[2])
                assert tmp_split_str == tmp_str.replace(' ', '_')

    sorted_key = sorted(sent_char_index2tmp_entity.keys(), key=lambda x: (x[0], x[1], x[2]))
    doc_name2instance = dict()
    for (tmp_sent_index, char_start, char_end) in sorted_key:
        sentence = sentence_index2sentence[tmp_sent_index]
        if str(tmp_sent_index) not in doc_name2instance:
            doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
        tmp_entity = sent_char_index2tmp_entity[(tmp_sent_index, char_start, char_end)]
        processed_tmp_entity = tmp_entity.replace(' ', '_')

        # assert (char_end - char_start) == len(tmp_str)
        mention = sentence[char_start: char_end]
        doc_name2instance[str(tmp_sent_index)]['entities']['starts'].append(char_start)
        doc_name2instance[str(tmp_sent_index)]['entities']['ends'].append(char_end)
        doc_name2instance[str(tmp_sent_index)]['entities']['entity_mentions'].append(mention)
        doc_name2instance[str(tmp_sent_index)]['entities']['entity_names'].append(processed_tmp_entity)

    # print(json.dumps(doc_name2instance, indent=4))
    print(f'num_in_kb: {num_in}; num_out_kb: {num_out};')

    return doc_name2instance


def gen_anno_from_xml(
        prefix='/nfs/yding4/In_Context_EL/data',
        dataset='ace2004',
        allow_mention_shift=False,
        allow_mention_without_entity=False,
        allow_repeat_annotation=False,
        has_prob=False,    # **YD-CL** whether ED probability exist in the xml
):

    """
    This function reads a standard XML EL annotation together with its documents.
    {dataset}:
    |
    |--RawText:
    |      |
    |      |---{doc_name} (contains the txt files)
    |
    |--{dataset}.xml (annotation covering all {doc_name})
    ATTENTION:
        a. '&' is replaced with '&amp;' when reading both "txt" and "annotation". '&' is not allowed in '.xml' files.
        b. In the '.xml' annotation, 'doc_name' contains underscores '_' instead of spaces. In mentions and entity
           annotations, there are spaces instead of underscores.
    :param prefix: absolute path prefix before the dataset directory.
    :param dataset: name of the dataset, which is also the name of the '.xml' file.
    :param allow_mention_shift: allow mismatch between "txt[offset: offset + length]" and "{annotated mention}".
    If set to True: use the length of "{annotated mention}" as actual length. Search the mention from
    "offset - 10" to "offset + 100" to find it.
    If set to False: raise ERROR if a mismatch is found.
    :param allow_mention_without_entity: allow empty entity annotation, either '' or 'NIL', also called "NER annotation".
    If set to True: the empty annotation is preserved and entity becomes ''.
    If set to False: raise ERROR if an empty entity is found.
    :param allow_repeat_annotation: allow repeated annotations.
    If set to True: repeated annotations will not be considered as outputs.
    If set to False: raise ERROR if a repeated annotation is found.
    :param has_prob
    If set to True: load ED probability.
    :return:
    doc_name2txt, doc_name2anno:
    doc_name2txt: a dictionary of strings. Each doc_name corresponds to a document in the dataset.
    doc_name2anno: a dictionary of lists. Each doc_name corresponds to a document in the dataset.
    Each element (ele) in the list is a dictionary with four elements:
    ele = {
            'start': offset,    # starting position of the mention in the doc_name text.
            'end': offset + length, # ending position of the mention in the doc_name text.
            'mention_txt': cur_mention, # annotated mention.
            'entity_txt': cur_ent_title, # annotated entity. '' or 'NIL' represents empty entity annotation (NER).
        }
    """

    raw_text_prefix = os.path.join(prefix, dataset + '/' + 'RawText')
    xml_file = os.path.join(prefix, dataset + '/' + dataset + '.xml')
    doc_names = os.listdir(raw_text_prefix)

    # Collect documents for each doc_name
    doc_name2txt = dict()

    for doc_name in doc_names:
        txt_path = os.path.join(raw_text_prefix, doc_name)
        txt = ''
        with open(txt_path, 'r') as reader:
            for line in reader:
                txt += line
        doc_name2txt[doc_name] = txt.replace('&amp;', '&')

    # Collect mention/entity annotations from XML
    doc_name2anno = defaultdict(list)
    # Nested named entity recognition problem in silver + gold
    reader = open(xml_file, 'r')

    doc_str_start = 'document docName=\"'
    doc_str_end = '\">'

    line = reader.readline()
    num_el_anno = 0
    num_ner_anno = 0
    num_shift_anno = 0
    num_change_length = 0
    cur_doc_name = ''

    while line:
        if doc_str_start in line:
            start = line.find(doc_str_start)
            end = line.find(doc_str_end)
            cur_doc_name = line[start + len(doc_str_start): end]
            cur_doc_name = cur_doc_name.replace('&amp;', '&').replace(' ', '_')
            assert cur_doc_name in doc_name2txt

            # **YD** preserve empty annotation for a doc_name
            doc_name2anno[cur_doc_name]

        else:
            if '<annotation>' in line:
                line = reader.readline()

                # **YD** bug here because mention may contain newline symbols, i.e., the annotated mention spans multiple lines.
                # assert '<mention>' in line and '</mention>' in line

                assert '<mention>' in line
                new_line = line
                while '</mention>' not in new_line:
                    new_line = reader.readline()
                    line += new_line

                m_start = line.find('<mention>') + len('<mention>')
                m_end = line.find('</mention>')

                cur_mention = line[m_start: m_end]
                cur_mention = cur_mention.replace('&amp;', '&').replace('_', ' ')

                line = reader.readline()
                # assert '<wikiName>' in line and '</wikiName>' in line
                e_start = line.find('<wikiName>') + len('<wikiName>')
                e_end = line.find('</wikiName>')
                cur_ent_title = '' if '<wikiName/>' in line else line[e_start: e_end]
                cur_ent_title = cur_ent_title.replace('&amp;', '&').replace('_', ' ')

                line = reader.readline()
                assert '<offset>' in line and '</offset>' in line
                off_start = line.find('<offset>') + len('<offset>')
                off_end = line.find('</offset>')
                offset = int(line[off_start: off_end])

                line = reader.readline()
                assert '<length>' in line and '</length>' in line
                len_start = line.find('<length>') + len('<length>')
                len_end = line.find('</length>')
                length_record = int(line[len_start: len_end])
                length = len(cur_mention)

                if length != length_record:
                    print('mention', cur_mention, 'offset', offset, 'length', length, 'length_record', length_record)
                    num_change_length += 1

                if has_prob:    # **YD-CL** whether ED probability exist in the xml.
                    line = reader.readline()
                    assert '<prob>' in line and '</prob>' in line
                    prob_start = line.find('<prob>') + len('<prob>')
                    prob_end = line.find('</prob>')
                    prob = float(line[prob_start: prob_end])

                line = reader.readline()
                if '<entity/>' in line:
                    line = reader.readline()

                if not has_prob:    # **YD-CL** allow 'prob' in the xml, but not loading it
                    assert '</annotation>' in line

                # if cur_ent_title != 'NIL' and cur_ent_title != '':
                assert cur_doc_name != ''
                ele = {
                        'start': offset,
                        'end': offset + length,
                        'mention_txt': cur_mention,
                        'entity_txt': cur_ent_title,
                    }
                if has_prob:    # **YD-CL** whether ED probability exist in the xml.
                    ele['prob'] = prob

                doc_txt = doc_name2txt[cur_doc_name]
                pos_mention = doc_txt[offset: offset + length]

                if allow_mention_shift:
                    if pos_mention != ele['mention_txt']:
                        num_shift_anno += 1
                        offset = max(0, offset - 10)
                        while pos_mention != cur_mention:
                            offset = offset + 1
                            pos_mention = doc_txt[offset: offset + length]
                            if offset > ele['start'] + 100:
                                print(
                                    'pos_mention',
                                    doc_txt[anno['offset']: anno['offset'] + length],
                                    anno['mention_txt'],
                                )
                                raise ValueError('huge error!')
                        ele['start'] = offset
                        ele['end'] = offset + length
                else:
                    if pos_mention != ele['mention_txt']:
                        print('pos_mention', pos_mention)
                        print("ele['mention_txt']", ele['mention_txt'])
                    assert pos_mention == ele['mention_txt'], 'Unmatched mention between annotation mention ' \
                                                              'and annotation position'

                # allow_mention_without_entity
                if ele['entity_txt'] == '' or ele['entity_txt'] == 'NIL':
                    ele['entity_txt'] = ''

                # Consider repeated annotations in Wikipedia
                if ele not in doc_name2anno[cur_doc_name]:
                    if ele['entity_txt'] != '':
                        doc_name2anno[cur_doc_name].append(ele)
                        num_el_anno += 1
                    else:
                        num_ner_anno += 1
                        if allow_mention_without_entity:
                            doc_name2anno[cur_doc_name].append(ele)
                else:
                    if not allow_repeat_annotation:
                        raise ValueError('find repeated annotation: ' + str(ele))

        line = reader.readline()

    print(
        'num_ner_anno', num_ner_anno,
        'num_el_anno', num_el_anno,
        'num_shift_anno', num_shift_anno,
        'num_change_length', num_change_length
    )

    # **YD** post-processing: sort the annotations by start and end.
    for doc_name in doc_name2anno:
        tmp_anno = doc_name2anno[doc_name]
        tmp_anno = sorted(tmp_anno, key=lambda x: (x['start'], x['end']))
        doc_name2anno[doc_name] = tmp_anno

    # Transform original format into the unified format.
    doc_name2instance = dict()
    for doc_name in doc_name2anno:
        assert doc_name in doc_name2txt
        txt = doc_name2txt[doc_name]
        anno = doc_name2anno[doc_name]
        starts = []
        ends = []
        entity_mentions = []
        entity_names = []

        for entity_instance in anno:
            start = entity_instance['start']
            end = entity_instance['end']
            entity_mention = entity_instance['mention_txt']
            entity_name = entity_instance['entity_txt']
            starts.append(start)
            ends.append(end)
            entity_mentions.append(entity_mention)
            entity_names.append(entity_name)
        
        if len(starts) == 0:
            continue
        doc_name2instance[doc_name] = {
            'sentence': txt,
            'entities': {
                'starts': starts,
                'ends': ends,
                'entity_mentions':entity_mentions ,
                'entity_names': entity_names,
            }
        }

    return doc_name2instance


    '''
    doc_name2instance[str(tmp_sent_index)] = {
                'sentence': sentence,
                'entities': {
                    'starts': [],
                    'ends': [],
                    'entity_mentions': [],
                    'entity_names': [],
                }
            }
    '''


def load_gendre_jsonl(file):
    with jsonlines.open(file) as reader:
        records = [record for record in reader]
    doc_name2instance = dict()
    for record in records:
        doc_name = record['id']
        sentence = record['input']
        mention = record['meta']['mention']
        entity_name = record['output'][0]['answer']
        start = sentence.index('[START_ENT] ') + len('[STRT_ENT] ') + 1
        entity_candidates = record['candidates']
        end = start + len(mention)
        if sentence[start: end] != mention:
            print(f'sentence[start: end]: {sentence[start: end]}; mention: {mention}')
        assert sentence[start: end] == mention
        doc_name2instance[doc_name] = {
            'sentence': sentence,
            'entities': {
                'starts': [start],
                'ends': [end],
                'entity_mentions': [mention],
                'entity_names': [entity_name],
                'entity_candidates': [
                    entity_candidates
                ],
            }
        }
    return doc_name2instance


def dataset_loader(file, key='', mode='tsv'): 
    '''
    file: input dataset file
    key: only used for aida, to consider train/valid/test split
    mode: options to consider different types of input file
    # mode to be expanded to multiple ED datasets
    '''
    if mode == 'tsv':
        doc_name2instance = load_tsv(file, key=key)
    elif mode == 'oke_2015':
        doc_name2instance = load_ttl_oke_2015(file)
    elif mode == 'oke_2016':
        doc_name2instance = load_ttl_oke_2016(file)
    elif mode == 'n3':
        doc_name2instance = load_ttl_n3(file)
    elif mode == 'unseen_mentions':
        doc_name2instance = load_unseen_mentions(file)
    elif mode == 'xml':
        parent_dir = os.path.dirname(os.path.dirname(file))
        dataset = os.path.basename(file).split('.')[0]
        doc_name2instance = gen_anno_from_xml(prefix=parent_dir, dataset=dataset)
    elif mode == 'gendre_jsonl':
        doc_name2instance = load_gendre_jsonl(file)
    else:
        raise ValueError('unknown mode!')
    return doc_name2instance


if __name__ == '__main__':
    # load_tsv()
    # load_ttl_oke_2015()
    # load_ttl_oke_2016()
    # load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl')
    # load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/RSS-500.ttl')

    # file = '/nfs/yding4/e2e_EL_evaluate/data/wned/xml/ori_xml2revise_xml/clueweb/clueweb.xml'
    # doc_name2instance = dataset_loader(file, mode='xml')
    # doc_name2instance = load_unseen_mentions()
    # doc_name2instance = load_ttl_oke_2016()
    # doc_name2instance = load_ttl_n3('/nfs/yding4/EL_project/dataset/n3-collection/Reuters-128.ttl')
    # num_entities = 0
    # num_mentions = 0
    # num_docs = 0
    # for doc_name, instance in doc_name2instance.items():
    #     num_docs += 1
    #     entities = instance['entities']
    #     for entity_name in entities['entity_names']:
    #         if entity_name != '':
    #             num_entities += 1
    #         num_mentions += 1
    # print(f'num_mentions: {num_mentions}, num_entities: {num_entities}, plnum_docs: {num_docs}')

    input_dir = '/nfs/yding4/In_Context_EL/data/ed/gendre'
    datasets = [
        'ace2004',
        'aida',
        'aquaint',
        'clueweb',
        'msnbc',
        'wiki',
    ]
    for dataset in datasets:
        print(f'dataset:{dataset}')
        input_file = os.path.join(input_dir, dataset + '-test-kilt.jsonl')
        load_gendre_jsonl(input_file)