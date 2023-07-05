import json
import pickle
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import PROJECT_ROOT, targetpad_transform
from datasets import CIRRDataset, CIRCODataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from phi import Phi
from utils import extract_image_features, device, collate_fn, extract_pseudo_tokens_with_phi


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str, clip_model_name: str, ref_names_list: List[str],
                                       pseudo_tokens: torch.Tensor, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval()

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names,
                                 ref_names_list, pseudo_tokens)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'cirr'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], ref_names_list: List[str], pseudo_tokens: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, reference_names, pairs_id, group_members = \
        cirr_generate_test_predictions(clip_model, relative_test_dataset, ref_names_list, pseudo_tokens)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset, ref_names_list: List[str],
                                   pseudo_tokens: torch.Tensor) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']

        group_members = np.array(group_members).T.tolist()

        input_captions = [
            f"a photo of $ that {rel_caption}" for rel_caption in relative_captions]

        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
        text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)

        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, clip_model_name: str, ref_names_list: List[str],
                                        pseudo_tokens: torch.Tensor, preprocess: callable,
                                        submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
                                                           index_names, ref_names_list, pseudo_tokens)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRCODataset, ref_names_list: List[str],
                                    pseudo_tokens: torch.Tensor) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']

        input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
        text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset,
                                                                    ref_names_list, pseudo_tokens)

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    # Compute the similarity
    similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="Filename of the generated submission file")
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'circo'], help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--eval-type", type=str, choices=['oti', 'phi', 'searle', 'searle-xl'], required=True,
                        help="If 'oti' evaluate directly using the inverted oti pseudo tokens, "
                             "if 'phi' predicts the pseudo tokens using the phi network, "
                             "if 'searle' uses the pre-trained SEARLE model to predict the pseudo tokens, "
                             "if 'searle-xl' uses the pre-trained SEARLE-XL model to predict the pseudo tokens")

    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--phi-checkpoint-name", type=str,
                        help="Phi checkpoint to use, needed when using phi, e.g. 'phi_20.pt'")

    args = parser.parse_args()

    if args.eval_type == 'oti':
        experiment_path = PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset.lower() / 'test' / args.exp_name

        with open(experiment_path / 'hyperparameters.json') as f:
            hyperparameters = json.load(f)

        pseudo_tokens = torch.load(experiment_path / 'ema_oti_pseudo_tokens.pt', map_location=device)
        with open(experiment_path / 'image_names.pkl', 'rb') as f:
            ref_names_list = pickle.load(f)

        clip_model_name = hyperparameters['clip_model_name']
        clip_model, clip_preprocess = clip.load(clip_model_name, device='cpu', jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")


    elif args.eval_type in ['phi', 'searle', 'searle-xl']:
        if args.eval_type == 'phi':
            phi_path = PROJECT_ROOT / 'data' / "phi_models" / args.exp_name
            if not phi_path.exists():
                raise ValueError(f"Experiment {args.exp_name} not found")

            hyperparameters = json.load(open(phi_path / "hyperparameters.json"))
            clip_model_name = hyperparameters['clip_model_name']
            clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

            phi = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
                      output_dim=clip_model.token_embedding.embedding_dim, dropout=hyperparameters['phi_dropout']).to(
                device)

            phi.load_state_dict(
                torch.load(phi_path / 'checkpoints' / args.phi_checkpoint_name, map_location=device)[
                    phi.__class__.__name__])
            phi = phi.eval()

        else:  # searle or searle-xl
            if args.eval_type == 'searle':
                clip_model_name = 'ViT-B/32'
            else:  # args.eval_type == 'searle-xl':
                clip_model_name = 'ViT-L/14'
            phi, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                                    backbone=clip_model_name)

            phi = phi.to(device).eval()
            clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

        if args.dataset.lower() == 'cirr':
            relative_test_dataset = CIRRDataset(args.dataset_path, 'test', 'relative', preprocess, no_duplicates=True)
        elif args.dataset.lower() == 'circo':
            relative_test_dataset = CIRCODataset(args.dataset_path, 'test', 'relative', preprocess)
        else:
            raise ValueError("Dataset not supported")

        clip_model = clip_model.float().to(device)
        pseudo_tokens, ref_names_list = extract_pseudo_tokens_with_phi(clip_model, phi, relative_test_dataset)
        pseudo_tokens = pseudo_tokens.to(device)
    else:
        raise ValueError("Eval type not supported")

    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model_name, ref_names_list, pseudo_tokens,
                                           preprocess, args.submission_name)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path, clip_model_name, ref_names_list, pseudo_tokens,
                                            preprocess, args.submission_name)

    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
