import json
import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Tuple, Dict, List, Set

from comet_ml import Experiment
import clip
import numpy as np
import pandas as pd
import torch
from clip.model import CLIP
from torch import optim, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import targetpad_transform, PROJECT_ROOT
from datasets import CIRRDataset, ImageNetDataset
from encode_with_pseudo_tokens import encode_with_pseudo_tokens
from phi import Phi
from utils import collate_fn, extract_image_features, device, CustomTensorDataset, contrastive_loss, \
    extract_pseudo_tokens_with_phi
from validate import cirr_compute_val_metrics

torch.multiprocessing.set_sharing_strategy('file_system')


def update_train_running_results(train_running_results, total_loss: torch.Tensor, gpt_loss: torch.Tensor,
                                 distillation_loss: torch.Tensor, images_in_batch: int):
    """
    Update the running results of the training
    :return:
    """
    train_running_results['accumulated_total_loss'] += total_loss.cpu().detach().item() * images_in_batch
    train_running_results['accumulated_gpt_loss'] += gpt_loss.cpu().detach().item() * images_in_batch
    train_running_results['accumulated_distillation_loss'] += distillation_loss.cpu().detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Set the description of the tqdm training bar
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"total_loss: {train_running_results['accumulated_total_loss'] / train_running_results['images_in_epoch']:.3f} "
             f"gpt_loss: {train_running_results['accumulated_gpt_loss'] / train_running_results['images_in_epoch']:.3f} "
             f"distil_loss: {train_running_results['accumulated_distillation_loss'] / train_running_results['images_in_epoch']:.3f}"
    )


def save_phi(name: str, cur_epoch: int, model_to_save: Phi, training_path: Path) -> None:
    """
    Save the weights of Phi during training
    """
    models_path = training_path / "checkpoints"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))


def save_checkpoint(name: str, epoch: int, model_to_save: Phi, optimizer: torch.optim.Optimizer, scaler: GradScaler,
                    training_path: Path) -> None:
    """
    Save model weights, optimizer and scaler
    """
    models_path = training_path / "checkpoints"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': epoch,
        model_name: model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'training_path': training_path,
    }, str(models_path / f'{name}.pt'))


def get_model_oti_tokens_preprocess(args, clip_model_name: str, exp_name: str, gpt_exp_name: str, oti_exp_name: str,
                                    preprocess_type: Literal['clip', 'targetpad']) -> \
        Tuple[CLIP, callable, Path, Dict[str, List[str]], Set[str], List[str], torch.Tensor]:
    """
    Get the CLIP model, the oti tokens and the preprocess function
    """

    training_path: Path = Path(PROJECT_ROOT / 'data' / "phi_models" / exp_name)

    if training_path.exists():
        print("BE CAREFUL: training path already exists, you are about to overwrite it", flush=True)

    # Save all the hyperparameters on a file
    training_path.mkdir(exist_ok=True, parents=True)
    with open(training_path / "hyperparameters.json", 'w+') as file:
        json.dump(vars(args), file, sort_keys=True, indent=4)

    # Load the CLIP model
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model.eval().float()

    # Define the preprocess function
    if preprocess_type == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used for training')
    elif preprocess_type == "targetpad":
        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
        print(f'Target pad  preprocess pipeline is used for training')
    else:
        raise ValueError(f"preprocess_type should be either clip or targetpad, got {preprocess_type}")

    # Load GPT phrases for regularization
    concepts_to_filter_out = set()
    with open(PROJECT_ROOT / 'data' / 'GPT_phrases' / gpt_exp_name / 'concept_to_phrases.pkl', 'rb') as f:
        concept_to_phrases = pickle.load(f)
    print('filtering problematic concepts')
    for k, v in tqdm(concept_to_phrases.items()):
        for phrase in v:
            p = phrase.replace(k, " $ ", 1)
            if '$' not in p:
                concepts_to_filter_out.add(k)

    # Load OTI tokens
    with open(PROJECT_ROOT / 'data' / 'oti_pseudo_tokens' / 'imagenet' / 'test' / oti_exp_name / f"image_names.pkl",
              'rb') as f:
        names_list = pickle.load(f)

    oti_tokens = torch.load(
        PROJECT_ROOT / 'data' / 'oti_pseudo_tokens' / 'imagenet' / 'test' / oti_exp_name / f'ema_oti_pseudo_tokens.pt',
        map_location=device)

    return clip_model, preprocess, training_path, concept_to_phrases, concepts_to_filter_out, names_list, oti_tokens


def train_one_epoch(train_loader: DataLoader, clip_model: CLIP, epoch: int, phi: Phi, lambda_distil: float,
                    lambda_gpt: float, name_to_concepts: Dict[str, List[str]], concept_to_phrases: Dict[str, List[str]],
                    num_epochs: int, optimizer: optim.Optimizer, scaler: GradScaler, temperature: float,
                    top_k_concepts: int, training_log_frame: pd.DataFrame, training_path: Path,
                    concepts_to_filter_out: Set[str], names_list: List[str], oti_pseudo_tokens: torch.Tensor) -> None:
    with experiment.train():
        phi.train()

        distillation_criterion = contrastive_loss
        gpt_criterion = nn.CosineEmbeddingLoss()
        gpt_criterion_target = torch.as_tensor([1], device=device)

        train_running_results = {'images_in_epoch': 0, 'accumulated_total_loss': 0, 'accumulated_gpt_loss': 0,
                                 'accumulated_distillation_loss': 0}

        train_bar = tqdm(train_loader, ncols=150)
        for idx, batch in enumerate(train_bar):
            images_features = batch['image'].to(device)
            names = batch['image_name']

            images_in_batch = images_features.size(0)
            step = len(train_bar) * (epoch - 1) + idx
            optimizer.zero_grad()

            # Get the concepts for the current batch
            batch_concepts = [random.choice(name_to_concepts[name][:top_k_concepts]) for name in names]
            while sum([concept in concepts_to_filter_out for concept in batch_concepts]) != 0:
                batch_concepts = [random.choice(name_to_concepts[name][:top_k_concepts]) for name in names]

            with torch.cuda.amp.autocast():
                # Get the estimated tokens using Phi
                estimated_tokens = phi(images_features)

                # Compute the distillation loss
                if lambda_distil > 0:
                    target_oti_tokens = torch.vstack(
                        [oti_pseudo_tokens[names_list.index(ref)].unsqueeze(0) for ref in names])
                    distillation_loss = distillation_criterion(target_oti_tokens, estimated_tokens, temperature)

                # Compute the regularization loss
                if lambda_gpt > 0:
                    reg_gpt_phrases = [random.choice(concept_to_phrases[concept]) for concept in batch_concepts]  #
                    token_gpt_phrases = [gpt_phrase.replace(w, " $ ") for w, gpt_phrase in
                                         zip(batch_concepts, reg_gpt_phrases)]

                    tokenized_reg_gpt_phrases = clip.tokenize(reg_gpt_phrases, truncate=True).to(device)
                    tokenized_tokens_gpt_phrases = clip.tokenize(token_gpt_phrases, truncate=True).to(device)

                    gpt_phrases_embeddings = clip_model.encode_text(tokenized_reg_gpt_phrases)
                    gpt_pseudo_tokens_phrases_embeddings = encode_with_pseudo_tokens(
                        clip_model, tokenized_tokens_gpt_phrases, estimated_tokens)
                    gpt_loss = gpt_criterion(gpt_phrases_embeddings, gpt_pseudo_tokens_phrases_embeddings,
                                             gpt_criterion_target)
                else:
                    gpt_loss = torch.nn.MSELoss()(torch.zeros(1).to(device), torch.zeros(1).to(device))

                total_loss = lambda_distil * distillation_loss + lambda_gpt * gpt_loss

            # Backpropagate and update the weights
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update running results and tqdm bar
            experiment.log_metric('step_total_loss', total_loss.detach().cpu().item(), step=step)
            experiment.log_metric('step_gpt_loss', gpt_loss.detach().cpu().item(), step=step)
            experiment.log_metric('step_distillation_loss', distillation_loss.detach().cpu().item(), step=step)

            update_train_running_results(train_running_results, total_loss, gpt_loss, distillation_loss,
                                         images_in_batch)
            set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

        # Epoch logging
        epoch_total_loss = float(
            train_running_results['accumulated_total_loss'] / train_running_results['images_in_epoch'])
        epoch_gpt_loss = float(
            train_running_results['accumulated_gpt_loss'] / train_running_results['images_in_epoch'])
        epoch_distillation_loss = float(
            train_running_results['accumulated_distillation_loss'] / train_running_results['images_in_epoch'])

        experiment.log_metric('epoch_total_loss', epoch_total_loss, epoch=epoch)
        experiment.log_metric('epoch_gpt_loss', epoch_gpt_loss, epoch=epoch)
        experiment.log_metric('epoch_distillation_loss', epoch_distillation_loss, epoch=epoch)

        # Training CSV logging
        training_log_frame = pd.concat(
            [training_log_frame,
             pd.DataFrame(data={'epoch': epoch, 'epoch_total_loss': epoch_total_loss,
                                'epoch_gpt_loss': epoch_gpt_loss, 'epoch_distillation_loss': epoch_distillation_loss},
                          index=[0])])
        training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)


def train_phi(args):
    clip_model, preprocess, training_path, concept_to_phrases, concepts_to_filter_out, names_list, oti_tokens = \
        get_model_oti_tokens_preprocess(args, args.clip_model_name, args.exp_name, args.gpt_exp_name, args.oti_exp_name,
                                        args.preprocess_type)

    # Define the phi model
    embedding_dim = clip_model.token_embedding.embedding_dim
    phi = Phi(input_dim=clip_model.visual.output_dim, hidden_dim=clip_model.visual.output_dim * 4,
              output_dim=embedding_dim, dropout=args.phi_dropout)
    phi.to(device)

    # Define the train datasets
    train_dataset = ImageNetDataset(args.imagenet_dataset_path, 'test', preprocess)

    # Extract the image features of the ImageNet test set
    training_image_features, training_image_names = extract_image_features(train_dataset, clip_model)
    training_image_features = training_image_features.cpu()

    # Create a dataloader for the training set for directly yielding the features along with the image names
    tensor_classic_train_dataset = CustomTensorDataset(training_image_features, training_image_names)
    classic_train_loader = DataLoader(dataset=tensor_classic_train_dataset, batch_size=args.batch_size,
                                      num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
                                      shuffle=True, drop_last=True)

    # Load the concepts associated with each image
    similar_concepts_df = pd.read_csv(PROJECT_ROOT / 'data' / "similar_concepts" / 'imagenet' / 'test' /
                                      f"{args.clip_model_name.replace('/', '')}_top_250_ensembling_image_texts.csv")
    name_to_concepts = {str(el[0]): el[1:] for el in similar_concepts_df.values.tolist()}

    # Define CIRR validation set
    cirr_relative_val_dataset = CIRRDataset(args.cirr_dataset_path, 'val', 'relative', preprocess)
    cirr_classic_val_dataset = CIRRDataset(args.cirr_dataset_path, 'val', 'classic', preprocess)

    # Extract the features for the CIRR validation set
    cirr_val_index_features, cirr_val_index_names = extract_image_features(cirr_classic_val_dataset, clip_model)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.AdamW(phi.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(1, args.num_epochs + 1):
        train_one_epoch(classic_train_loader, clip_model, epoch, phi, args.lambda_distil, args.lambda_gpt,
                        name_to_concepts, concept_to_phrases, args.num_epochs, optimizer,
                        scaler, args.temperature, args.top_k_concepts, training_log_frame, training_path,
                        concepts_to_filter_out, names_list, oti_tokens)

        # Validation
        if epoch % args.validation_frequency == 0:
            with experiment.validate():
                phi.eval()

                # Extract the pseudo tokens for the CIRR validation set using Phi
                cirr_val_pseudo_tokens, cirr_val_ref_names_list = extract_pseudo_tokens_with_phi(clip_model, phi,
                                                                                                 cirr_relative_val_dataset)
                cirr_val_pseudo_tokens = cirr_val_pseudo_tokens.to(device)

                # Compute the CIRR validation metrics
                cirr_results_dict = cirr_compute_val_metrics(cirr_relative_val_dataset, clip_model,
                                                             cirr_val_index_features, cirr_val_index_names,
                                                             cirr_val_ref_names_list, cirr_val_pseudo_tokens)

                print(json.dumps(cirr_results_dict, indent=4))

                experiment.log_metrics(
                    cirr_results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                val_log_dict = {'epoch': epoch}
                val_log_dict.update(cirr_results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=val_log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

        if args.save_training:
            if epoch > 0 and epoch % args.save_frequency == 0:
                save_phi(f'phi_{epoch}', epoch, phi, training_path)
            save_phi(f'phi_last', epoch, phi, training_path)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--exp-name", required=True, type=str, help="Experiment name")
    parser.add_argument("--clip-model-name", required=True, type=str,
                        help="CLIP model to use, e.g 'ViT-B/32', 'ViT-L/14'")
    parser.add_argument("--imagenet-dataset-path", type=str, help="Path to ImageNet dataset", required=True)
    parser.add_argument("--cirr-dataset-path", type=str, help="Path to Cirr dataset", required=True)
    parser.add_argument("--oti-exp-name", type=str, required=True,
                        help="Name of the ImageNet OTI tokens experiment")

    parser.add_argument("--gpt-exp-name", type=str, default="GPTNeo27B",
                        help="Name of the GPT generation phrases experiment")
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    parser.add_argument("--phi-dropout", default=0.5, type=float, help="Dropout probability for the phi network")
    parser.add_argument("--batch-size", default=256, type=int, help="Phi training batch size")
    parser.add_argument("--num-workers", default=10, type=int, help="Number of workers")
    parser.add_argument("--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-epochs", default=100, type=int, help="Number training epochs")
    parser.add_argument("--lambda-distil", type=float, default=1, help="Distillation loss weight")
    parser.add_argument("--lambda-gpt", type=float, default=0.75, help="GPT loss weight")
    parser.add_argument("--temperature", default=0.25, type=float, help="Distillation loss temperature")
    parser.add_argument("--validation-frequency", default=1, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--save-frequency", default=5, type=int, help="Saving frequency expressed in epochs")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the model checkpoints or not")
    parser.add_argument("--top-k-concepts", type=int, default=150, help="Number of concepts associated to each image")

    parser.add_argument("--api-key", type=str, help="Api for Comet logging")
    parser.add_argument("--workspace", type=str, help="Workspace of Comet logging")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"phi training",
            workspace=args.workspace,
            disabled=False
        )
    else:
        print("Comet logging DISABLED, to enable it please provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name=f"",
            workspace="",
            disabled=True)

    experiment.set_name(args.exp_name)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    experiment.log_code(folder=str(PROJECT_ROOT / 'src'))
    experiment.log_parameters(args)

    train_phi(args)
