from argparse import ArgumentParser

import clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import ImageFile
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import PROJECT_ROOT, targetpad_transform
from datasets import FashionIQDataset, CIRRDataset, CIRCODataset, ImageNetDataset
from utils import get_templates, device

ImageFile.LOAD_TRUNCATED_IMAGES = True


@torch.no_grad()
def associate_image_concepts(args):
    """
    Associate a set of concepts to each image in the dataset
    """

    # load the CLIP models
    model, clip_preprocess = clip.load(args.clip_model_name, device=device)

    model: CLIP = model.eval().requires_grad_(False)
    feature_dim = model.visual.output_dim

    # Number of concepts per image
    concepts_per_image = 250

    # Define the preprocess pipeline
    if args.preprocess_type == 'targetpad':
        preprocess = targetpad_transform(1.25, model.visual.input_resolution)
    elif args.preprocess_type == 'clip':
        preprocess = clip_preprocess
    else:
        raise ValueError("transform_type should be in ['clip', 'targetpad']")

    # Get the dataset
    if args.dataset.lower() == 'fashioniq':
        dataset = FashionIQDataset(args.dataset_path, args.split, ['dress', 'shirt', 'toptee'], args.dataset_mode,
                                   preprocess=preprocess, no_duplicates=True)
    elif args.dataset.lower() == 'cirr':
        dataset = CIRRDataset(args.dataset_path, args.split, args.dataset_mode, preprocess=preprocess,
                              no_duplicates=True)
    elif args.dataset.lower() == 'circo':
        dataset = CIRCODataset(args.dataset_path, args.split, args.dataset_mode, preprocess=preprocess)
    elif args.dataset.lower() == 'imagenet':
        dataset = ImageNetDataset(args.dataset_path, args.split, preprocess=preprocess)
    else:
        raise ValueError("dataset should be in ['fashioniq', 'cirr', 'circo', 'imagenet']")

    templates = get_templates()

    # Load the dictionary of concepts
    df = pd.read_csv(PROJECT_ROOT / "data" / "oidv7-class-descriptions.csv")
    texts = df["DisplayName"].values.tolist()

    # Create the dataloader
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    # Compute the embeddings for the dictionary
    bs = args.batch_size
    dictionary_embeddings = torch.zeros((0, feature_dim), device=device)
    for i in tqdm(range(0, len(texts), bs)):
        if i + bs > len(texts) - 1:
            bs = len(texts) - i

        for k in range(i, i + bs):
            prompts = [f"{template.format(f' {texts[k]} ')}" for template in templates]
            tokens = clip.tokenize(prompts).to(device)
            feat = model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
            feat = feat.mean(dim=0)
            feat /= feat.norm()
            dictionary_embeddings = torch.vstack((dictionary_embeddings, feat))

    # Normalize the embeddings
    dictionary_embeddings = F.normalize(dictionary_embeddings, dim=-1)

    output_path = PROJECT_ROOT / 'data' / "similar_concepts" / args.dataset.lower() / args.split
    output_path.mkdir(parents=True, exist_ok=True)

    texts_output_csv_column_names = ["Image Name"] + [str(i) for i in range(1, concepts_per_image + 1)]
    texts_output_csv = [[]]
    texts_output_csv_name = f"{args.clip_model_name.replace('/', '')}_top_{concepts_per_image}_ensembling_image_texts.csv"
    texts_output_csv_path = output_path / texts_output_csv_name

    # Extract the concepts for each image
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        image_feat = F.normalize(model.encode_image(images), dim=-1).float()
        similarity = image_feat @ dictionary_embeddings.T

        images_knn_indexes = torch.topk(similarity, concepts_per_image, largest=True).indices.cpu()
        for name, image_knn_indexes in zip(names, images_knn_indexes):
            decoded = []
            for idx in image_knn_indexes:
                decoded.append(texts[idx])
            texts_output_csv.append([name] + decoded)

    # Save the dataframe
    texts_output_csv = texts_output_csv[1:]  # Remove empty first item
    df = pd.DataFrame(texts_output_csv)
    df.columns = texts_output_csv_column_names
    df.to_csv(texts_output_csv_path, index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--clip-model-name", required=True, type=str,
                        help="CLIP model to use, e.g 'ViT-B/32', 'ViT-L/14'")
    parser.add_argument("--dataset", type=str, required=True, choices=['cirr', 'fashioniq', 'circo', 'imagenet'],
                        help="Dataset to use")
    parser.add_argument("--dataset-path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--split", type=str, required=True, choices=['train', 'val', 'test'],
                        help="Dataset split to use")

    parser.add_argument("--dataset-mode", type=str, default='classic', choices=['relative', 'classic'],
                        help="Dataset mode (does not influence ImageNet dataset)")
    parser.add_argument("--batch-size", default=32, type=int, help='Batch size')
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--preprocess-type", default="targetpad", type=str, choices=['clip', 'targetpad'],
                        help="Preprocess pipeline to use")
    args = parser.parse_args()

    associate_image_concepts(args)


if __name__ == '__main__':
    main()
