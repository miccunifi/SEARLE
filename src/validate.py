import json
import pickle
from argparse import ArgumentParser
from typing import List, Dict, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils import collate_fn, PROJECT_ROOT, targetpad_transform
from datasets import FashionIQDataset, CIRRDataset, CIRCODataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from phi import Phi
from utils import extract_image_features, device, extract_pseudo_tokens_with_phi

torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def fiq_generate_val_predictions(clip_model: CLIP, relative_val_dataset: Dataset, ref_names_list: List[str],
                                 pseudo_tokens: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
    """
    Generates features predictions for the validation set of Fashion IQ.
    """

    # Create data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []

    # Compute features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_captions']

        flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        input_captions_reversed = [
            f"{flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        input_captions = [
            f"a photo of $ that {in_cap}" for in_cap in input_captions]
        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
        text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)

        input_captions_reversed = [
            f"a photo of $ that {in_cap}" for in_cap in input_captions_reversed]
        tokenized_input_captions_reversed = clip.tokenize(input_captions_reversed, context_length=77).to(device)
        text_features_reversed = encode_with_pseudo_tokens(clip_model, tokenized_input_captions_reversed,
                                                           batch_tokens)

        predicted_features = F.normalize((F.normalize(text_features) + F.normalize(text_features_reversed)) / 2)
        # predicted_features = F.normalize((text_features + text_features_reversed) / 2)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, target_names_list


@torch.no_grad()
def fiq_compute_val_metrics(relative_val_dataset: Dataset, clip_model: CLIP, index_features: torch.Tensor,
                            index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names = fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list,
                                                                    pseudo_tokens)

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    # Compute the distances
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return {'fiq_recall_at10': recall_at10,
            'fiq_recall_at50': recall_at50}


@torch.no_grad()
def fiq_val_retrieval(dataset_path: str, dress_type: str, clip_model_name: str, ref_names_list: List[str],
                      pseudo_tokens: torch.Tensor, preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, clip_model)

    # Define the relative dataset
    relative_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess)

    return fiq_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names, ref_names_list,
                                   pseudo_tokens)


@torch.no_grad()
def cirr_generate_val_predictions(clip_model: CLIP, relative_val_dataset: Dataset, ref_names_list: List[str],
                                  pseudo_tokens: torch.Tensor) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generates features predictions for the validation set of CIRR
    """

    # Define the dataloader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn)
    predicted_features_list = []
    target_names_list = []
    group_members_list = []
    reference_names_list = []

    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
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
        target_names_list.extend(target_names)
        group_members_list.extend(group_members)
        reference_names_list.extend(reference_names)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, target_names_list, group_members_list


@torch.no_grad()
def cirr_compute_val_metrics(relative_val_dataset: Dataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, reference_names, target_names, group_members = \
        cirr_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list, pseudo_tokens)

    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    return {
        'cirr_recall_at1': recall_at1,
        'cirr_recall_at5': recall_at5,
        'cirr_recall_at10': recall_at10,
        'cirr_recall_at50': recall_at50,
        'cirr_group_recall_at1': group_recall_at1,
        'cirr_group_recall_at2': group_recall_at2,
        'cirr_group_recall_at3': group_recall_at3,
    }


@torch.no_grad()
def cirr_val_retrieval(dataset_path: str, clip_model_name: str, ref_names_list: list, pseudo_tokens: torch.Tensor,
                       preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRR validation set given the pseudo tokens and the reference names
    """

    # Load the model
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = CIRRDataset(dataset_path, 'val', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, clip_model)

    # Define the relative validation dataset
    relative_val_dataset = CIRRDataset(dataset_path, 'val', 'relative', preprocess)

    return cirr_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names,
                                    ref_names_list, pseudo_tokens)


@torch.no_grad()
def circo_generate_val_predictions(clip_model: CLIP, relative_val_dataset: Dataset, ref_names_list: List[str],
                                   pseudo_tokens: torch.Tensor) -> Tuple[
    torch.Tensor, List[str], list]:
    """
    Generates features predictions for the validation set of CIRCO
    """

    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']

        gt_img_ids = np.array(gt_img_ids).T.tolist()
        input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        batch_tokens = torch.vstack([pseudo_tokens[ref_names_list.index(ref)].unsqueeze(0) for ref in reference_names])
        tokenized_input_captions = clip.tokenize(input_captions, context_length=77).to(device)
        text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, target_names_list, gts_img_ids_list


@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset: Dataset, clip_model: CLIP, index_features: torch.Tensor,
                              index_names: List[str], ref_names_list: List[str], pseudo_tokens: torch.Tensor) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, relative_val_dataset,
                                                                                   ref_names_list, pseudo_tokens)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float())

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


@torch.no_grad()
def circo_val_retrieval(dataset_path: str, clip_model_name: str, ref_names_list: List[str], pseudo_tokens: torch.Tensor,
                        preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = CIRCODataset(dataset_path, 'val', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, clip_model)

    # Define the relative validation dataset
    relative_val_dataset = CIRCODataset(dataset_path, 'val', 'relative', preprocess)

    return circo_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names, ref_names_list,
                                     pseudo_tokens)


def main():
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, help="Experiment to evaluate")
    parser.add_argument("--eval-type", type=str, choices=['oti', 'phi', 'searle', 'searle-xl'], required=True,
                        help="If 'oti' evaluate directly using the inverted oti pseudo tokens, "
                             "if 'phi' predicts the pseudo tokens using the phi network, "
                             "if 'searle' uses the pre-trained SEARLE model to predict the pseudo tokens, "
                             "if 'searle-xl' uses the pre-trained SEARLE-XL model to predict the pseudo tokens"
                        )
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'fashioniq', 'circo'],
                        help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)

    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--phi-checkpoint-name", type=str,
                        help="Phi checkpoint to use, needed when using phi, e.g. 'phi_20.pt'")

    args = parser.parse_args()

    if args.eval_type in ['phi', 'oti'] and args.exp_name is None:
        raise ValueError("Experiment name is required when using phi or oti evaluation type")
    if args.eval_type == 'phi' and args.phi_checkpoint_name is None:
        raise ValueError("Phi checkpoint name is required when using phi evaluation type")

    if args.eval_type == 'oti':
        experiment_path = PROJECT_ROOT / 'data' / "oti_pseudo_tokens" / args.dataset.lower() / 'val' / args.exp_name
        if not experiment_path.exists():
            raise ValueError(f"Experiment {args.exp_name} not found")

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
            phi = phi.eval()

            phi.load_state_dict(
                torch.load(phi_path / 'checkpoints' / args.phi_checkpoint_name, map_location=device)[
                    phi.__class__.__name__])

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

        if args.dataset.lower() == 'fashioniq':
            relative_val_dataset = FashionIQDataset(args.dataset_path, 'val', ['dress', 'toptee', 'shirt'],
                                                    'relative', preprocess, no_duplicates=True)
        elif args.dataset.lower() == 'cirr':
            relative_val_dataset = CIRRDataset(args.dataset_path, 'val', 'relative', preprocess,
                                               no_duplicates=True)
        elif args.dataset.lower() == 'circo':
            relative_val_dataset = CIRCODataset(args.dataset_path, 'val', 'relative', preprocess)
        else:
            raise ValueError("Dataset not supported")

        clip_model = clip_model.float().to(device)
        pseudo_tokens, ref_names_list = extract_pseudo_tokens_with_phi(clip_model, phi, relative_val_dataset)
        pseudo_tokens = pseudo_tokens.to(device)
    else:
        raise ValueError("Eval type not supported")

    print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")
    if args.dataset.lower() == 'fashioniq':
        recalls_at10 = []
        recalls_at50 = []
        for dress_type in ['shirt', 'dress', 'toptee']:
            fiq_metrics = fiq_val_retrieval(args.dataset_path, dress_type, clip_model_name, ref_names_list,
                                            pseudo_tokens, preprocess)
            recalls_at10.append(fiq_metrics['fiq_recall_at10'])
            recalls_at50.append(fiq_metrics['fiq_recall_at50'])

            for k, v in fiq_metrics.items():
                print(f"{dress_type}_{k} = {v:.2f}")
            print("\n")

        print(f"average_fiq_recall_at10 = {np.mean(recalls_at10):.2f}")
        print(f"average_fiq_recall_at50 = {np.mean(recalls_at50):.2f}")

    elif args.dataset.lower() == 'cirr':
        cirr_metrics = cirr_val_retrieval(args.dataset_path, clip_model_name, ref_names_list, pseudo_tokens,
                                          preprocess)

        for k, v in cirr_metrics.items():
            print(f"{k} = {v:.2f}")

    elif args.dataset.lower() == 'circo':
        circo_metrics = circo_val_retrieval(args.dataset_path, clip_model_name, ref_names_list, pseudo_tokens,
                                            preprocess)

        for k, v in circo_metrics.items():
            print(f"{k} = {v:.2f}")


if __name__ == '__main__':
    main()
