import json
import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import clip
import numpy as np
import pandas as pd
import torch
from PIL import ImageFile
from clip.model import CLIP
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import PROJECT_ROOT, targetpad_transform, collate_fn
from datasets import FashionIQDataset, CIRRDataset, CIRCODataset, ImageNetDataset
from utils import device
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from utils import get_templates

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')


def oti_inversion(args):
    """
    Perform Optimization-based Textual Inversion (OTI) using the CLIP model
    """

    # load the CLIP model
    clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device)
    clip_model: CLIP = clip_model.float()
    clip_model.visual.requires_grad_(False)
    embedding_dim = clip_model.token_embedding.embedding_dim

    # Initialize the pseudo tokens tensor
    names_list = []
    ema_global_oti_pseudo_tokens = torch.empty((0, embedding_dim))
    global_oti_pseudo_tokens = torch.empty((0, embedding_dim))

    # Resume training from a saved experiment
    if args.resume_experiment:
        print("Resuming training from a saved experiment", flush=True)

        # Load names and pseudo tokens
        with open(PROJECT_ROOT / "data" / "oti_pseudo_tokens" / args.dataset.lower() / args.split /
                  args.exp_name / f"image_names.pkl", 'rb') as f:
            names_list = pickle.load(f)

        global_oti_pseudo_tokens = torch.load(
            PROJECT_ROOT / "data" / "oti_pseudo_tokens" / args.dataset.lower() / args.split /
            args.exp_name / f'oti_pseudo_tokens.pt')
        ema_global_oti_pseudo_tokens = torch.load(
            PROJECT_ROOT / "data" / "oti_pseudo_tokens" / args.dataset.lower() / args.split /
            args.exp_name / f'ema_oti_pseudo_tokens.pt')

        # Load the saved hyperparameters
        with open(PROJECT_ROOT / "data" / "oti_pseudo_tokens" / args.dataset.lower() / args.split /
                  args.exp_name / 'hyperparameters.json') as f:
            old_hyperparamters = json.load(f)

        # Check if the hyperparameters are the same
        for k, v in old_hyperparamters.items():
            if k in args:
                if v != vars(args)[k]:
                    print(f"Warning: {k} is different from the saved experiment")
                    print(f"saved parameter: {v} \t new_parameter: {vars(args)[k]}")

    # Set the experiment path
    experiment_path = (
            PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset.lower() / args.split / args.exp_name)

    if experiment_path.exists() and not args.resume_experiment:
        print("BE CAREFUL: training path already exists, you are about to overwrite it", flush=True)

    # Load the similar concepts into a dictionary
    similar_concepts_df = pd.read_csv(PROJECT_ROOT / "data" / "similar_concepts" / args.dataset.lower() / args.split /
                                      f"{args.clip_model_name.replace('/', '')}_top_250_ensembling_image_texts.csv")
    name_to_concepts = {str(el[0]): el[1:] for el in similar_concepts_df.values.tolist()}

    # Filter out problematic concepts from the GPT phrases
    concepts_to_filter_out = set()
    with open(PROJECT_ROOT / 'data' / 'GPT_phrases' / args.gpt_exp_name / 'concept_to_phrases.pkl', 'rb') as f:
        concept_to_phrases = pickle.load(f)
    print('filtering problematic concepts')
    for k, v in tqdm(concept_to_phrases.items()):
        for phrase in v:
            p = phrase.replace(k, " $ ", 1)
            if '$' not in p:
                concepts_to_filter_out.add(k)

    # Load source code for logging
    with open(Path(__file__).absolute()) as f:
        source_code = f.read()

    # Set preprocessing function
    if args.preprocess_type == 'clip':
        preprocess = clip_preprocess
    elif args.preprocess_type == 'targetpad':
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
    else:
        raise ValueError("preprocess should be in ['targetpad', 'clip']")

    # Define the dataset
    if args.dataset.lower() == 'fashioniq':
        dataset = FashionIQDataset(args.dataset_path, args.split, ['dress', 'shirt', 'toptee'], 'relative',
                                   preprocess, no_duplicates=True)
    elif args.dataset.lower() == 'cirr':
        dataset = CIRRDataset(args.dataset_path, args.split, 'relative', preprocess, no_duplicates=True)
    elif args.dataset.lower() == 'circo':
        dataset = CIRCODataset(args.dataset_path, args.split, 'relative', preprocess)
    elif args.dataset.lower() == 'imagenet':
        dataset = ImageNetDataset(args.dataset_path, args.split, preprocess)
    else:
        raise ValueError("dataset should be in ['fashioniq', 'cirr', 'circo', 'imagenet']")

    # Define the dataloader and the criterions
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=8)
    criterion = nn.CosineEmbeddingLoss()
    criterion_target = torch.as_tensor([1], device=device)

    templates = get_templates()

    # Start the loop
    for batch_idx, batch in enumerate(tqdm(loader)):
        # Load the batch
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        # Check if the batch has already been processed
        if set(names) <= set(names_list):
            continue

        bs = len(images)

        # Randomly initialize the oti pseudo tokens
        oti_pseudo_tokens = torch.empty((bs, embedding_dim), device=device)
        nn.init.normal_(oti_pseudo_tokens, std=0.02)

        # Extract the image features
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                batch_im_features = clip_model.encode_image(images.to(device))

        # Get the GPT phrases associated with the similar concepts in the batch
        batch_gpt_phrases = []
        batch_concepts = []
        for idx, name in enumerate(names):
            l_phrases = []
            l_concept_phrases = []
            l_similar_concepts = name_to_concepts[name]
            stripped_concepts = [str(w).strip() for w in l_similar_concepts]
            l_similar_concepts = list(dict.fromkeys(stripped_concepts))[:args.top_k]

            for w in l_similar_concepts:
                if w in concepts_to_filter_out:
                    continue
                for p in concept_to_phrases[w]:
                    l_phrases.append(p)
                    l_concept_phrases.append(w)

            batch_gpt_phrases.append(l_phrases)
            batch_concepts.append(l_concept_phrases)

        # Copy the oti pseudo tokens for the EMA
        oti_pseudo_tokens = nn.Parameter(oti_pseudo_tokens)
        ema_oti_pseudo_tokens = oti_pseudo_tokens.clone().detach()

        # Define the optimizer and the scaler
        optimizer = optim.AdamW([oti_pseudo_tokens], lr=args.learning_rate, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        # Start the oti optimization loop
        for _ in range(args.oti_steps):
            optimizer.zero_grad()

            # Sample the templates sentences
            template_indexes = random.choices(range(len(templates)), k=bs)
            template_oti_texts = [templates[i].format(" $ ") for i in template_indexes]

            # Sample the GPT phrases
            gpt_phrases_indexes = [random.choice(range(len(batch_gpt_phrases[i]))) for i in
                                   range(len(batch_gpt_phrases))]
            reg_gpt_phrases = [batch_gpt_phrases[i][rand_idx] for i, rand_idx in enumerate(gpt_phrases_indexes)]
            concepts_to_replace = [batch_concepts[i][rand_idx] for i, rand_idx in enumerate(gpt_phrases_indexes)]
            # Replace the concepts in the GPT phrases with the oti pseudo tokens
            oti_gpt_phrases = [reg_text.replace(w, " $ ", 1) for w, reg_text in
                               zip(concepts_to_replace, reg_gpt_phrases)]

            # Tokenize the sentences
            tokenized_oti_gpt_phrases = clip.tokenize(oti_gpt_phrases, truncate=True).to(device)
            tokenized_reg_gpt_phrases = clip.tokenize(reg_gpt_phrases, truncate=True).to(device)
            tokenized_template_oti_texts = clip.tokenize(template_oti_texts, truncate=True).to(device)

            with torch.cuda.amp.autocast():
                # Extract the features
                template_oti_features = encode_with_pseudo_tokens(clip_model, tokenized_template_oti_texts,
                                                                  oti_pseudo_tokens)
                reg_gpt_features = clip_model.encode_text(tokenized_reg_gpt_phrases)
                reg_oti_gpt_features = encode_with_pseudo_tokens(clip_model, tokenized_oti_gpt_phrases,
                                                                 oti_pseudo_tokens)

                # Compute the loss
                cosine_loss = criterion(template_oti_features, batch_im_features, criterion_target)
                gpt_loss = criterion(reg_oti_gpt_features, reg_gpt_features, criterion_target)

                oti_loss = (args.lambda_cos * cosine_loss + args.lambda_gpt * gpt_loss) / \
                           (args.lambda_cos + args.lambda_gpt)

            # Backpropagate the loss
            scaler.scale(oti_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update the EMA
            ema_oti_pseudo_tokens = args.ema_decay * ema_oti_pseudo_tokens + (1 - args.ema_decay) * oti_pseudo_tokens

        # Update the global oti pseudo tokens and the names list
        names_list.extend(names)
        ema_global_oti_pseudo_tokens = torch.vstack(
            (ema_global_oti_pseudo_tokens, ema_oti_pseudo_tokens.detach().cpu()))
        global_oti_pseudo_tokens = torch.vstack((global_oti_pseudo_tokens, oti_pseudo_tokens.detach().cpu()))

        # Save the experiment
        if batch_idx % args.save_frequency == 0 and batch_idx > 0:
            save_experiment(args, ema_global_oti_pseudo_tokens, experiment_path, global_oti_pseudo_tokens, names_list,
                            source_code)

    # Before finishing save the experiment
    save_experiment(args, ema_global_oti_pseudo_tokens, experiment_path, global_oti_pseudo_tokens, names_list,
                    source_code)


def save_experiment(args, ema_global_oti_pseudo_tokens: torch.tensor, experiment_path: Path,
                    global_oti_pseudo_tokens: torch.tensor, names_list: List, source_code: str) -> None:
    """
    Saves the pseudo tokens, the names list, the source code and the hyperparameters of the experiment
    """

    experiment_path.mkdir(exist_ok=True, parents=True)
    with open(experiment_path / f'image_names.pkl', 'wb+') as f:
        pickle.dump(names_list, f)
    torch.save(global_oti_pseudo_tokens, experiment_path / f'oti_pseudo_tokens.pt')
    torch.save(ema_global_oti_pseudo_tokens, experiment_path / f'ema_oti_pseudo_tokens.pt')
    with open(experiment_path / 'source_code.py', 'w+') as f:
        f.write(source_code)
    with open(experiment_path / 'hyperparameters.json', 'w+') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", required=True, type=str, help="Experiment name")
    parser.add_argument("--clip-model-name", required=True, type=str,
                        help="CLIP model to use, e.g 'ViT-B/32', 'ViT-L/14'")
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'fashioniq', 'circo', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'],
                        help="Dataset split to use")

    parser.add_argument("--gpt-exp-name", type=str, default="GPTNeo27B",
                        help="Name of the GPT generation phrases experiment")
    parser.add_argument("--learning-rate", default=2e-2, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch-size", default=32, type=int, help='batch size for each optimization iteration')
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--top-k", type=int, default=15, help="Number of concepts associated to each image")
    parser.add_argument("--oti-steps", default=350, type=int, help="Number of optimization steps for each image")
    parser.add_argument("--lambda_gpt", type=float, default=0.5, help="Weight of the gpt loss")
    parser.add_argument("--lambda_cos", type=float, default=1, help="Weight of the cosine loss")
    parser.add_argument("--ema-decay", type=float, default=0.99, help="Decay for the exponential moving average")
    parser.add_argument("--save-frequency", default=5, type=int, help="Saving frequency expressed in batches")
    parser.add_argument("--resume-experiment", action='store_true', help="Resume the experiment if it exists",
                        default=False)
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    oti_inversion(args)


if __name__ == '__main__':
    main()
