import json
import pickle
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import PROJECT_ROOT

from utils import device, dtype


def process_phrase(phrase: str):
    phrase = re.sub("(\n)+", '. ', phrase)
    phrase = re.sub('\.+', '.', phrase)
    return phrase


@torch.no_grad()
def generate_phrases(args):
    """
    Generate a set of phrases for each concept in the dictionary
    """

    concept_to_phrases = {}

    # Save source code
    with open(Path(__file__).absolute()) as f:
        source_code = f.read()

    # Define experiment path
    experiment_path = PROJECT_ROOT / 'data' / 'GPT_phrases' / f"{args.exp_name}"
    if experiment_path.exists():
        print("BEWARE: experiment path already exists. Overwriting.", flush=True)

    if args.resume_experiment:
        experiment_path = PROJECT_ROOT / 'data' / 'GPT_phrases' / args.saved_exp_name
        print(f"Loading from {experiment_path}", flush=True)
        with open(experiment_path / 'concept_to_phrases.pkl', 'rb') as f:
            concept_to_phrases = pickle.load(f)

    # Load the GPT model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.gpt_model).to(device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load the dictionary of concepts
    df = pd.read_csv(PROJECT_ROOT / "data" / "oidv7-class-descriptions.csv")
    texts = df["DisplayName"].values.tolist()

    # Generate phrases
    for test_idx, concept in enumerate(tqdm(texts)):
        concept = concept.strip()

        # Skip if already generated
        if concept in concept_to_phrases:
            continue

        # Generate phrases
        prompt = f"a photo of {concept} that"
        inputs = tokenizer(prompt, return_tensors="pt")

        model_output = model.generate(input_ids=inputs.input_ids.to(device),
                                      attention_mask=inputs.attention_mask.to(device), do_sample=True,
                                      max_length=args.max_length, num_return_sequences=args.num_return_sequences,
                                      temperature=args.temperature, no_repeat_ngram_size=args.no_repeat_ngram_size)

        gen_txt = tokenizer.batch_decode(model_output, skip_special_tokens=True)

        phrases = [process_phrase(phrase) for phrase in gen_txt]
        concept_to_phrases[concept] = phrases

        if test_idx % 100 == 0 and test_idx > 0:
            save_experiment(args, concept_to_phrases, experiment_path, source_code)

    # Save the generated phrases
    save_experiment(args, concept_to_phrases, experiment_path, source_code)


def save_experiment(args, concept_to_phrases: Dict, experiment_path: Path, source_code: str) -> None:
    """
    Save the generated phrases, the hyperparameters and the source code
    """
    experiment_path.mkdir(exist_ok=True, parents=True)
    with open(experiment_path / 'hyperparameters.json', 'w+') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    with open(experiment_path / 'concept_to_phrases.pkl', 'wb') as f:
        pickle.dump(concept_to_phrases, f)
    with open(experiment_path / 'source_code.py', 'w+') as f:
        f.write(source_code)


def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, default='GPTNeo27B', help="experiment name")
    parser.add_argument("--gpt-model", type=str, default="EleutherAI/gpt-neo-2.7B", help="GPT model to use")
    parser.add_argument("--max-length", type=int, default=35, help="max length of the generated text")
    parser.add_argument("--num-return-sequences", type=int, default=256,
                        help="Number of generated sequences for each word in the dictionary")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for sampling")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=2, help=" Size of the n-gram to avoid repetitions")
    parser.add_argument("--resume-experiment", action='store_true', help="Resume the experiment if it exists",
                        default=False)

    args = parser.parse_args()
    generate_phrases(args)


if __name__ == '__main__':
    main()
