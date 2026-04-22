"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Evaluation of LLMs for CyberBench
"""
import os
import re
import sys
import json
import torch
import argparse
import evaluate
import transformers
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Dict, List
from sklearn import metrics

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from langchain_openai import AzureOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.manager import get_openai_callback
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# Set pipeline parameters
chat_openai_models = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
openai_models = chat_openai_models + [
    'text-ada-001', 'text-curie-001',
    'text-davinci-002', 'text-davinci-003',
]
openai_embedding_models = ['text-embedding-ada-002-2']
selected_columns = ['instruction', 'input', 'output']
dataset2task = {
    'cyner': 'ner', 'aptner': 'ner', 'cynews': 'sum',
    'secmmlu': 'mc', 'cyquiz': 'mc',
    'mitre': 'tc', 'cve': 'tc', 'web': 'tc',
    'email': 'tc', 'http': 'tc',
}
all_dataset_names = list(dataset2task.keys())


def load_openai_model(model_name, temperature=0, max_new_tokens=512, stop_tokens=[]):
    """
    Load the OpenAI LLM
    """
    openai_model_kwargs = {"stop": stop_tokens}
    if model_name in chat_openai_models:
        llm = AzureChatOpenAI(
            deployment_name=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            model_kwargs=openai_model_kwargs,
        )
    else:
        llm = AzureOpenAI(
            deployment_name=model_name,
            max_tokens=max_new_tokens,
            temperature=temperature,
            model_kwargs=openai_model_kwargs,
        )
    return llm


def load_hf_model(
        model_name, model_type='causal', model_folder='models',
        quantization=None, lora=False, temperature=0, max_new_tokens=512, stop_tokens=[]):
    """
    Load the HuggingFace LLM in full precision or quantization with potential LoRA layers
    """
    model_path = os.path.join(model_folder, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if model_type == 'causal':
        auto_model = AutoModelForCausalLM
        peft_model = PeftModelForCausalLM
    elif model_type == 'seq2seq':
        auto_model = AutoModelForSeq2SeqLM
        peft_model = PeftModelForSeq2SeqLM
    else:
        auto_model = AutoModel
        peft_model = PeftModel

    if quantization == '4bit':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == '8bit':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        bnb_config = None

    if lora:
        lora_config = LoraConfig.from_pretrained(model_path)
        base_model = auto_model.from_pretrained(
            lora_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={"": 0},
        )
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=False)
        model = peft_model.from_pretrained(
            model=base_model,
            model_id=model_path,
        )
    else:
        model = auto_model.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map={"": 0},
        )

    eos_token_id = [tokenizer.eos_token_id] \
        + [tokenizer.encode(stop_token)[-1] for stop_token in stop_tokens]
    if temperature == 0:
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
        )
    else:
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
        )
    generator = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    llm = HuggingFacePipeline(pipeline=generator)

    return llm


def load_model(
        model_name, model_type=None, model_folder=None,
        quantization=None, lora=False, temperature=0, max_new_tokens=512, stop_tokens=[]):
    """
    Load the LLM
    """
    if model_name in openai_models:
        llm = load_openai_model(
            model_name, temperature=temperature,
            max_new_tokens=max_new_tokens, stop_tokens=stop_tokens)
    else:
        llm = load_hf_model(
            model_name, model_folder=model_folder, model_type=model_type,
            quantization=quantization, lora=lora,
            temperature=temperature, max_new_tokens=max_new_tokens, stop_tokens=stop_tokens)
    return llm


def load_embedding_model(embedding_model_name, model_folder='models'):
    """
    Load the embedding model
    """
    if embedding_model_name in openai_embedding_models:
        embedding_model = OpenAIEmbeddings(
            deployment=embedding_model_name,
            chunk_size=16,
            model_kwargs={},
        )
    else:
        embedding_model_path = os.path.join(model_folder, embedding_model_name)
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": "cuda"},
        )
    return embedding_model


class RandomExampleSelector(BaseExampleSelector):
    """
    Example selector that selects examples randomly.
    """

    def __init__(self, examples: List[Dict[str, str]], k: int):
        self.examples = examples
        self.k = k

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return np.random.choice(self.examples, size=self.k, replace=False)


def get_example_selector(df_train, embedding_model, num_shots):
    """
    Get the example selector for few shots
    """
    train_examples = df_train[selected_columns].to_dict(orient='records')

    if embedding_model is not None:
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples=train_examples,
            embeddings=embedding_model,
            vectorstore_cls=FAISS,
            k=num_shots,
        )
    else:
        example_selector = RandomExampleSelector(
            examples=train_examples,
            k=num_shots,
        )

    return example_selector


def get_prompt_template(prompt_name, num_shots, example_selector=None):
    """
    Get the prompt template
    """
    if prompt_name == 'alpaca':
        # Source: https://github.com/tatsu-lab/stanford_alpaca#data-release
        # {system_prompt}\n\n### Instruction:\n{instruction}\n\n
        # ### Input:\n{input}\n\n### Response:\n{output}\n\n
        # ### Input:\n{input}\n\n### Response:\n
        system_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
            "Write a response that appropriately completes the request."
        prompt_prefix = f"{system_prompt}\n\n### Instruction:\n{{instruction}}"
        example_prompt_template = "### Input:\n{input}\n\n### Response:\n{output}"
        example_input_variables = ["input", "output"]
        prompt_suffix = "### Input:\n{input}\n\n### Response:\n"
    else:
        prompt_prefix = "Instruction: {instruction}"
        example_prompt_template = "Input: {input}\nOutput: {output}"
        example_input_variables = ["input", "output"]
        prompt_suffix = "Input: {input}\nOutput: "

    if num_shots == 0:
        prompt_template = PromptTemplate(
            template=prompt_prefix + "\n\n" + prompt_suffix,
            input_variables=["instruction", "input"],
        )
    else:
        example_prompt = PromptTemplate(
            template=example_prompt_template,
            input_variables=example_input_variables,
        )
        prompt_template = FewShotPromptTemplate(
            example_selector=example_selector,
            example_prompt=example_prompt,
            prefix=prompt_prefix,
            suffix=prompt_suffix,
            input_variables=["instruction", "input"],
        )

    return prompt_template


def get_responses(df_test, llm, model_name, prompt_template, output_file_path):
    """
    Get LLM responses
    """
    prompts = []
    responses = []
    callbacks = []

    with get_openai_callback() as callback:
        num_tests = len(df_test)
        pbar = tqdm(enumerate(df_test.iterrows()), total=num_tests)

        for i, (index, row) in pbar:
            if 'instruction' in row:
                prompt = prompt_template.format(
                    instruction=row.instruction, input=row.input)
            else:
                prompt = prompt_template.format(input=row.input)

            try:
                response = llm.predict(prompt)
                if response.startswith(prompt):
                    response = response[len(prompt):]
            except KeyboardInterrupt:
                break
            except Exception as error:
                response = ''
                print(
                    f'Failed: model = {model_name}, dataset = {row.dataset}, index = {index}, error = "{error}"')

            prompts.append(prompt)
            responses.append(response)
            callbacks.append(str(callback))
            estimated_total_cost = callback.total_cost / (i + 1) * num_tests
            pbar.set_postfix_str(
                f"${callback.total_cost:.4f}/${estimated_total_cost:.4f}")

    df_test['prompt'] = prompts
    df_test['response'] = responses
    df_test['callback'] = callbacks
    df_test.to_csv(output_file_path, index=False)
    print(f'Saved to {output_file_path}')

    return df_test


def load_responses(output_file_path):
    """
    Load LLM responses
    """
    df_test = pd.read_csv(output_file_path)
    print(f'Loaded from {output_file_path}')
    return df_test


def evaluate_ner(df_test):
    """
    Evaluate the NER task
    """
    # Calculate F1 scores for NER
    tp_count = 0
    fp_count = 0
    fn_count = 0

    for _, row in df_test.iterrows():
        output = json.loads(row['output'].strip())

        try:
            response = json.loads(row['response'].strip())
            assert type(response) == dict \
                and all(
                    type(value) == list
                    and all(type(v) == str for v in value
                            ) for value in response.values())
        except (json.JSONDecodeError, AssertionError):
            response = {}
            print(f"Cannot parse: {row['response'].strip()}")

        for key in set(output.keys()).union(set(response.keys())):
            true_entities = set(output[key]) if key in output else set()
            predicted_entities = set(
                response[key]) if key in response else set()
            tp_count += len(true_entities.intersection(predicted_entities))
            fp_count += len(predicted_entities - true_entities)
            fn_count += len(true_entities - predicted_entities)

    precision = tp_count / max(1, tp_count + fp_count)
    recall = tp_count / max(1, tp_count + fn_count)
    f1_score = 2 * precision * recall / (precision + recall) \
        if precision != 0 and recall != 0 else 0
    print(
        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}')
    evaluation_result = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return evaluation_result


def evaluate_tc(df_test, dataset_name):
    """
    Evaluate the text classification task
    """
    # Calculate F1 scores
    positive_labels = {
        'web': 'phishing',
        'email': 'phishing',
        'http': 'anomalous',
    }
    positive_label = positive_labels[dataset_name]
    def map_label(label): return 1 if label == positive_label else 0
    y_true = df_test['output'].str.strip().apply(map_label).tolist()
    y_pred = df_test['response'].str.strip().apply(map_label).tolist()
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary')
    print(
        f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}')
    evaluation_result = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return evaluation_result


def evaluate_sum(df_test):
    """
    Evaluate the summarization task
    """
    # Calculate ROUGE scores for summarization
    rouge = evaluate.load("rouge")
    evaluation_result = rouge.compute(
        predictions=df_test['response'].tolist(),
        references=df_test['output'].tolist(),
        use_aggregator=True,
    )
    print(f"ROUGE-1: {evaluation_result['rouge1']:.4f}, "
          f"ROUGE-2: {evaluation_result['rouge2']:.4f}, "
          f"ROUGE-L: {evaluation_result['rougeL']:.4f}")
    return evaluation_result


def evaluate_responses(df_test, dataset_name, task_name=None, result_file_path=None):
    """
    Evaluate LLM responses
    """
    evaluation_result = {}

    if dataset_name in ['mitre', 'cve', 'secmmlu', 'cyquiz']:
        # Calculate accuracies
        accuracy = (
            df_test['output'].str.strip() == df_test['response'].str.strip()
        ).mean()
        print(f'Accuracy: {accuracy:.4f}')
        evaluation_result = {'accuracy': accuracy}

    elif dataset_name in ['web', 'email', 'http']:
        evaluation_result = evaluate_tc(df_test, dataset_name)

    elif dataset_name in ['cyner', 'aptner']:
        evaluation_result = evaluate_ner(df_test)

    elif dataset_name in ['cynews']:
        evaluation_result = evaluate_sum(df_test)

    # Save evaluation results
    if result_file_path is not None:
        with open(result_file_path, 'w') as file:
            json.dump(evaluation_result, file)
            print(f'Saved to {result_file_path}')

    return evaluation_result


def load_data(df, dataset_name):
    """
    Load one dataset
    """
    df_train = df[(df.dataset == dataset_name) & (df.split == 'train')].copy()
    df_test = df[(df.dataset == dataset_name) & (df.split == 'test')].copy()

    # Prepare texts for string formatting
    for column in selected_columns:
        if column in df_train:
            df_train[column] = df_train[column].str.replace(pat='{', repl='{{', regex=False)\
                .str.replace(pat='}', repl='}}', regex=False)

    return df_train, df_test


def get_output_name(
        dataset_name, model_name, embedding_model_name, prompt_name, num_shots,
        quantization=None, lora=False):
    """
    Get the output file name
    """
    dataset_base_name = os.path.splitext(os.path.basename(dataset_name))[0]
    model_option_name = ''
    if quantization is not None:
        model_option_name += f'-{quantization}'
    if lora:
        model_option_name += f'-lora'
    output_name = f'{dataset_base_name}-{model_name}{model_option_name}' \
        f'-{embedding_model_name}-{prompt_name}-{num_shots}-shot'.lower()
    return output_name


if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(
        description='Evaluate one large language model (LLM) with datasets from CyberBench.')
    parser.add_argument('--model', default='llama-2-7b-chat-hf', type=str,
                        help='LLM name in the folder of models or one of the OpenAI model (default: llama-2-7b-chat-hf)')
    parser.add_argument('--model_type', default='causal', type=str, choices=['causal', 'seq2seq'],
                        help='LLM type for HuggingFace models (default: causal)')
    parser.add_argument('--embedding', default='all-mpnet-base-v2', type=str,
                        help='embedding model name for few-shot examples or none for random selection (default: all-mpnet-base-v2)')
    parser.add_argument('--quantization', default=None, type=str, choices=['4bit', '8bit', None],
                        help='quantization for the LLM (default: None)')
    parser.add_argument('--lora', default=False, action='store_true',
                        help='load LoRA layers for the LLM? (default: False)')
    parser.add_argument('--prompt', default='alpaca', type=str, choices=['alpaca'],
                        help='prompt template name (default: alpaca)')
    parser.add_argument('--shot', default=5, type=int,
                        help='number of shots (default: 5 but 0 only for summarization tasks such as cynews)')
    parser.add_argument('--datasets', default=['cyberbench'], type=str, nargs='+',
                        help='dataset names in CyberBench (e.g. cyner) or all CyberBench datasets (cyberbench) (default: cyberbench)')
    parser.add_argument('--temperature', default=0, type=float,
                        help='temperature for text generation (default: 0)')
    parser.add_argument('--new_token', default=512, type=int,
                        help='maximum number of new tokens (default: 512)')
    parser.add_argument('--multiline', default=False, action='store_true',
                        help='keep more than the first response line? (default: False)')

    # Parse arguments
    args = parser.parse_args()
    model_name = args.model
    model_type = args.model_type
    model_folder = 'models'
    data_folder = 'data'
    embedding_model_name = args.embedding
    quantization = args.quantization
    lora = args.lora
    prompt_name = args.prompt
    default_num_shots = args.shot
    temperature = args.temperature
    max_new_tokens = args.new_token
    if args.multiline:
        stop_tokens = []
    else:
        stop_tokens = ['\n', '\n\n']

    # Get dataset names
    print(f'Loading the cyber-bench data ...')
    df_all = pd.read_csv(os.path.join(data_folder, 'cyberbench.csv'))
    available_dataset_names = df_all['dataset'].dropna().unique().tolist()

    # Get dataset names
    if 'cyberbench' in args.datasets:
        dataset_names = [d for d in all_dataset_names if d in available_dataset_names]
        missing_dataset_names = [d for d in all_dataset_names if d not in available_dataset_names]
        if missing_dataset_names:
            print(f"Skipping missing datasets not present in cyberbench.csv: {missing_dataset_names}")
    else:
        dataset_names = args.datasets

    # Set the seed
    transformers.enable_full_determinism(seed=0)
    transformers.set_seed(seed=0)

    # Load the model and data
    print(f'Loading the {model_name} model ...')
    llm = load_model(
        model_name, model_type=model_type, model_folder=model_folder,
        quantization=quantization, lora=lora,
        temperature=temperature, max_new_tokens=max_new_tokens, stop_tokens=stop_tokens)
    if embedding_model_name != 'none':
        print(f'Loading the {embedding_model_name} embedding model ...')
        embedding_model = load_embedding_model(
            embedding_model_name, model_folder=model_folder)
    else:
        print(f'Using the random selection ...')
        embedding_model = None
    os.makedirs('outputs', exist_ok=True)
    print('-' * 100)

    # Run the evaluation pipeline
    for dataset_name in dataset_names:
        if dataset_name in all_dataset_names:
            df = df_all
            task_name = dataset2task[dataset_name]
            print(f'Task: {task_name}, Dataset: {dataset_name}')
        else:
            print(f'Loading the {dataset_name} data ...')
            df = pd.read_csv(dataset_name)
            task_name = df.task.iloc[0]
            dataset_name = df.dataset.iloc[0]
            print(f'Task: {task_name}, Dataset: {dataset_name}')

        num_shots = 0 if task_name == 'sum' else default_num_shots
        output_name = get_output_name(
            dataset_name, model_name, embedding_model_name, prompt_name, num_shots,
            quantization, lora)
        output_file_path = os.path.join('outputs', f'{output_name}.csv')

        if not os.path.exists(output_file_path):
            df_train, df_test = load_data(df, dataset_name)
            if df_train.empty or df_test.empty:
                print(f"Skipping dataset '{dataset_name}' because train or test split is empty.")
                print('-' * 100)
                continue
            if num_shots != 0:
                example_selector = get_example_selector(
                    df_train, embedding_model, num_shots)
            else:
                example_selector = None

            print(
                f'Loading the {prompt_name} prompt template with {num_shots} shots ...')
            prompt_template = get_prompt_template(
                prompt_name, num_shots, example_selector)
            df_test = get_responses(
                df_test, llm, model_name, prompt_template, output_file_path)
        else:
            df_test = load_responses(output_file_path)

        result_file_path = os.path.join('outputs', f'{output_name}.json')
        evaluation_result = evaluate_responses(
            df_test, dataset_name, task_name, result_file_path)
        print('-' * 100)