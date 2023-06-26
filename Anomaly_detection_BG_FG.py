import os
import time
from argparse import ArgumentParser
import sklearn
from mmdet.models.utils import unpack_gt_instances
import numpy as np
from matplotlib import pyplot as plt
from mmengine.runner import Runner
import torch
import copy
from mmdet.utils import register_all_modules as register_all_modules_mmdet
from mmrotate.utils import register_all_modules
from mmengine.config import Config
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay
from torchvision import models
from mmdet.registry import TASK_UTILS
from mmengine.structures import InstanceData
from mmdet.structures.bbox import HorizontalBoxes
from tqdm import tqdm
import skimage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

images_to_use = ['P0000__1024__0___2472', 'P0002__1024__824___824', 'P0021__1024__824___1648',
                 'P0022__1024__824___4120',
                 'P0036__1024__2169___1648', 'P0063__1024__824___1648', 'P0082__1024__824___3296',
                 'P0129__1024__384___824',
                 'P0136__1024__1648___1554', 'P0158__1024__1648___824', 'P0315__1024__0___0', 'P0260__1024__0___0',
                 'P0504__1024__0___0', 'P0653__1024__0___172', 'P0871__1024__1367___0', 'P0873__1024__610___824',
                 'P0913__1024__1648___0', 'P0911__1024__11___973', 'P1055__1024__3832___3418',
                 'P1374__1024__1648___824']


def create_dataloader(cfg, mode="train"):
    dataloader_mode = mode + "_dataloader"
    test_dataloader = cfg.get(dataloader_mode)
    dataloader_cfg = copy.deepcopy(test_dataloader)
    dataloader_cfg["dataset"]["_scope_"] = "mmrotate"
    data_loader = Runner.build_dataloader(dataloader_cfg, seed=123456)
    return data_loader


def show_img(img: torch.tensor):
    image_tensor = img.cpu()
    # Convert the tensor to a NumPy array
    image_np = image_tensor.numpy()
    # Transpose the dimensions if needed
    # (PyTorch tensors are typically in the channel-first format, while NumPy arrays are in channel-last format)
    image_np = np.transpose(image_np, (1, 2, 0)).astype(int)
    # Display the image using matplotlib
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()


# calculate bboxes indices for all pyramid levels
def calc_patches_indices(pyramid_levels: int, patch_size: int, scale_factor: float, image_size):
    patches_indices_tensor = []
    curr_image_size = image_size
    for i in range(pyramid_levels):
        scale = scale_factor ** i
        curr_scale_patches_indices = compute_patches_indices_per_scale(curr_image_size,
                                                                       (patch_size, patch_size),
                                                                       scale_factor=scale)
        curr_image_size = (int(scale_factor * curr_image_size[0]), int(scale_factor * curr_image_size[1]))
        patch_ind = curr_scale_patches_indices.view(curr_scale_patches_indices.shape[0], 4)
        patches_indices_tensor.append(patch_ind)
    patches_indices_tensor = torch.cat(patches_indices_tensor, dim=0)
    patches_horizontal_boxes = HorizontalBoxes(patches_indices_tensor)
    formatted_patches_indices = InstanceData(priors=patches_horizontal_boxes)
    return formatted_patches_indices


def compute_patches_indices_per_scale(image_size, patch_size, scale_factor=1):
    patch_H, patch_W = patch_size
    H, W = image_size
    patches_W, _ = divmod(W, patch_W)
    patches_H, _ = divmod(H, patch_H)
    indices = torch.tensor([(i, j) for i in range(patches_H) for j in range(patches_W)])
    top_left = indices * torch.tensor(patch_size)
    bottom_right = top_left + torch.tensor(patch_size)
    corners = torch.stack([top_left, bottom_right], dim=1).float()
    corners *= 1 / scale_factor
    return corners.int()


def split_images_into_patches(images, patch_size):
    # dimensions of the input batch (B: batch size, C: channels, H: height, W: width)
    B, C, H, W = images.shape
    # shape of the puzzle piece
    piece_H, piece_W = patch_size
    # calculate the maximum height and width that fits the piece shape
    max_H = (H // piece_H) * piece_H
    max_W = (W // piece_W) * piece_W
    # truncate the images to the maximum size that fits the piece shape
    images = images[:, :, :max_H, :max_W]
    # number of pieces along each dimension
    num_pieces_H = max_H // piece_H
    num_pieces_W = max_W // piece_W
    # split the images
    splits = images.view(B, C, num_pieces_H, piece_H, num_pieces_W, piece_W)
    splits = splits.permute(0, 2, 4, 1, 3, 5).contiguous()
    splits = splits.view(B, num_pieces_H * num_pieces_W, C, piece_H, piece_W)
    return splits


def preprocess_patches(patches):
    resized_patches = F.interpolate(patches, size=(224, 224), mode='bilinear', align_corners=False)
    resized_patches = resized_patches / 255.0
    normalized_patches = (resized_patches - mean) / std
    return normalized_patches


def create_gaussian_pyramid_old(data, pyramid_levels: int = 5, scale_factor: float = 0.6,
                            kernel_size: int = 3, kernel_sigma: int = 1):
    current_level = torch.stack(data).float()
    pyramid_batch = [current_level]
    # show_img(current_level[0])
    # create gaussian pyramid
    for level in range(pyramid_levels - 1):
        blurred = TF.gaussian_blur(current_level,
                                   kernel_size=kernel_size, sigma=kernel_sigma)
        downsampled = F.interpolate(blurred, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        pyramid_batch.append(downsampled)
        # show_img(downsampled[0])
        current_level = downsampled
    return pyramid_batch

def create_gaussian_pyramid(data: list, pyramid_levels, scale_factor):
    all_pyramids = []
    for im_ind in range(len(data)):
        curr_pyr = skimage.transform.pyramid_gaussian(data[im_ind].permute(1, 2, 0).numpy(), max_layer=pyramid_levels,
                                                      downscale=1 / scale_factor)
        all_pyramids.append(curr_pyr)


def calculate_scores_and_labels(test_features, labels, train_features, k):
    # set 1 for foreground and 0 for background
    labels[torch.nonzero(labels >= 0)] = 1
    labels[torch.nonzero(labels == -1)] = 0
    scores = calculate_scores(test_features.cpu(), train_features.cpu(), k=k)
    return scores, labels


def calculate_scores(test_features, train_features, k=3):
    diff = torch.cdist(test_features, train_features)
    # Sort the differences along the second dimension
    sorted_diff, indices = torch.sort(diff, dim=1)
    # Select the k closest values
    closest_values = sorted_diff[:, :k]
    scores = torch.mean(closest_values.float(), dim=1)
    return scores


def convert_oriented_representation_to_bounding_square(oriented_representation):
    h = oriented_representation.heights
    w = oriented_representation.widths
    max_dim_length = torch.cat([h.unsqueeze(0), w.unsqueeze(0)]).max(dim=0).values.unsqueeze(-1)
    left_corner = oriented_representation.centers - max_dim_length / 2.0
    right_corner = oriented_representation.centers + max_dim_length / 2.0
    return left_corner.int(), right_corner.int()


def slice_patches(image, all_labels, left_corners, right_corners):
    # Slice the patches from the image using tensor indexing
    sliced_patches = []
    filtered_labels = []
    for ind, (curr_left_corner, curr_right_corner, label) in enumerate(zip(left_corners, right_corners, all_labels)):
        x_start, y_start = curr_left_corner[0], curr_left_corner[1]
        x_end, y_end = curr_right_corner[0], curr_right_corner[1]
        sliced_patch = image[:, y_start:y_end, x_start:x_end]
        if sliced_patch.shape[1] != 0 and sliced_patch.shape[2] != 0:
            sliced_patches.append(sliced_patch)
            filtered_labels.append(label)
    # print("Num of boxes is " + str(len(filtered_labels)))
    return sliced_patches, filtered_labels


def get_gt_bboxes_and_labels(gt_instances, image):
    bounding_square_left_corner, bounding_square_right_corner = convert_oriented_representation_to_bounding_square(gt_instances.bboxes)
    bboxes, labels = slice_patches(image, gt_instances.labels,
                                   bounding_square_left_corner, bounding_square_right_corner)

    return bboxes, torch.tensor(labels)


def calculate_batch_features_and_labels(data_batch, pyramid_levels: int, patch_size: int, scale_factor: float,
                                        features_model, is_supervised=False):
    pyramid_batch = create_gaussian_pyramid_old(data_batch['inputs'], pyramid_levels=pyramid_levels,
                                            scale_factor=scale_factor,)
    # pyramid_batch = create_gaussian_pyramid(data_batch['inputs'], pyramid_levels=pyramid_levels,
    #                                          scale_factor=scale_factor,)
    # extract patches
    patches_accord_level = []
    for pyramid_level_data in pyramid_batch:
        patches = split_images_into_patches(pyramid_level_data,
                                            (patch_size, patch_size))
        patches_accord_level.append(patches)
    images_patches = torch.cat(patches_accord_level, dim=1)
    # extract ground truth labels per patch
    gt_instances = unpack_gt_instances(data_batch['data_samples'])
    batch_gt_instances, batch_gt_instances_ignore, _ = gt_instances

    images_patches_features = []
    patches_labels = []
    for image_idx in range(images_patches.shape[0]):
        # extract gt label for each patch
        assign_result = bbox_assigner.assign(
            formatted_patches_indices, batch_gt_instances[image_idx],
            batch_gt_instances_ignore[image_idx])
        # remove permanently labels and features with iou larger than 0.3, i.e. keep only background
        # bounding boxes with gt_inds 0 is background. the threshold is in the config file.
        indices_of_bounding_boxes_with_iou_bellow_threshold = torch.squeeze(
            torch.nonzero(assign_result.gt_inds == 0))
        # background_labels is class label for classes and -1 for background.
        background_labels = assign_result.labels[indices_of_bounding_boxes_with_iou_bellow_threshold]
        # extract the image pixels for each background patch:
        background_patches = images_patches[image_idx][indices_of_bounding_boxes_with_iou_bellow_threshold]
        # preprocess patches for ResNet:
        prep_patches_per_image = preprocess_patches(background_patches)
        if is_supervised:
            foreground_labels = torch.tensor([]).cuda()
            foreground_patches = torch.tensor([]).cuda()
        else:
            # extract foreground patches using blocked squared bounding box.
            foreground_patches, foreground_labels = get_gt_bboxes_and_labels(
                batch_gt_instances[image_idx], data_batch['inputs'][image_idx])

        # merge labels for later evaluation
        all_labels = torch.concat((background_labels, foreground_labels), dim=0)
        # extract features for each patch
        preprocesses_foreground_patches = [preprocess_patches(torch.unsqueeze(gt_bbox, dim=0).float())
                                           for gt_bbox in foreground_patches]
        preprocesses_foreground_patches = torch.concat(preprocesses_foreground_patches, dim=0) \
            if len(preprocesses_foreground_patches) > 0 else torch.tensor([])

        background_and_foreground_patches = torch.concat(
            [prep_patches_per_image, preprocesses_foreground_patches],
            dim=0).to(device)

        with torch.no_grad():
            output = features_model(background_and_foreground_patches)
            output = torch.flatten(output, 1)

        patches_labels.append(all_labels.cpu().detach())
        images_patches_features.append(output.cpu().detach())
    images_patches_features = torch.concat(images_patches_features, dim=0)
    patches_labels = torch.concat(patches_labels, dim=0)
    return images_patches_features, patches_labels


def printing_data_statistics(labels, is_train):
    fg_number = torch.nonzero(labels != 0).size()[0]
    bg_number = torch.nonzero(labels == 0).size()[0]
    title = "train" if is_train else "test"
    print(title + " data contain " + str(fg_number) + " fg samples and " + str(bg_number) + " bg samples\n")


def cache_features_dictionary(dataset_cfg, pyramid_levels: int, patch_size: int, scale_factor:float, dataset_type: str,
                              target_dictionary_path: str):
    data_loader = create_dataloader(dataset_cfg, mode=dataset_type)
    features = []
    labels = []
    for idx, data_batch in tqdm(enumerate(data_loader)):
        images_patches_features, patches_labels = calculate_batch_features_and_labels(data_batch, pyramid_levels,
                                                                                      patch_size, scale_factor,
                                                                                      features_model, is_supervised=False)
        features.append(images_patches_features.cpu().detach())
        labels.append(patches_labels)
        # code to remove:
        for sub_idx in range(len(data_batch['data_samples'])):
            if not data_batch['data_samples'][sub_idx].img_id in images_to_use:
                raise ValueError(f'Unexpected image: {data_batch["data_samples"][0].img_id}')

    features = torch.concat(features, dim=0)
    labels = torch.concat(labels, dim=0)
    labels[torch.nonzero(labels >= 0)] = 1
    labels[torch.nonzero(labels == -1)] = 0
    data = {'features': features, 'labels': labels}
    torch.save(data, target_dictionary_path)


def calculate_scores_and_labels_for_test_dataset(pyramid_levels, patch_size, scale_factor, dictionary, dataloader,
                                                 features_model, k):
    scores = []
    labels = []
    for idx, data_batch in tqdm(enumerate(dataloader)):
        if idx == 5:
            break
        images_patches_features, patches_labels = calculate_batch_features_and_labels(data_batch, pyramid_levels,
                                                                                      patch_size, scale_factor,
                                                                                      features_model, is_supervised=False)
        batch_scores, batch_labels = calculate_scores_and_labels(images_patches_features, patches_labels,
                                                                 dictionary, k)
        scores.append(batch_scores)
        labels.append(batch_labels)
    scores = torch.concat(scores, axis=0)
    labels = torch.concat(labels, dim=0)
    return scores, labels


if __name__ == '__main__':
    t = time.time()
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-o", "--output_dir", help="The saved model path")
    parser.add_argument("-l", "--pyramid_levels", default=5, help="Number of pyramid levels")
    parser.add_argument("-sf", "--scale_factor", default=0.6, help="Scale factor between pyramid levels")
    parser.add_argument("-ps", "--patch_size", default=50, help="The patch size")
    parser.add_argument("-k", "--k_value", default=3, help="Nearest Neighbours count.")
    parser.add_argument("-use-cached", "--use_cached", action='store_true',
                        help="If flagged, use the cached feature dict. "
                             "Otherwise, recalculate it every time you run the script.")

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    # mmrotate initialization stuff.
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    image_size = cfg["train_pipeline"][3]['scale']
    formatted_patches_indices = calc_patches_indices(args.pyramid_levels, args.patch_size, args.scale_factor,
                                                     image_size)
    bbox_assigner = TASK_UTILS.build(cfg["patches_assigner"])
    model = models.resnet50(pretrained=True)
    model.eval()
    model = model.to(device)
    features_model = torch.nn.Sequential(*list(model.children())[:-1]).eval()

    # Create features for the training stage in a case it does not exist
    dictionary_path = os.path.join(args.output_dir, 'features_dict_new.pt')
    if not args.use_cached:
        print('Recaching features dictionary...')
        cache_features_dictionary(cfg, args.pyramid_levels, args.patch_size, args.scale_factor,
                                  dataset_type='subtrain', target_dictionary_path=dictionary_path)
    # Test stage
    data = torch.load(dictionary_path)
    printing_data_statistics(data['labels'], is_train=True)

    val_data_loader = create_dataloader(cfg, mode='val')
    k_value = args.k_value
    k_to_auc = {}

    for k_value in [1, 3, 5, 7, 11, 13]:
        print(f"Evaluating scores and labels for k={k_value}...")
        scores, labels = calculate_scores_and_labels_for_test_dataset(args.pyramid_levels, args.patch_size,
                                                                      args.scale_factor, dictionary=data['features'],
                                                                      dataloader=val_data_loader,
                                                                      features_model=features_model,
                                                                      k=int(k_value))

        printing_data_statistics(labels, is_train=False)
        # 1 to ood 0 to objects
        print(f"Calculating AuC for k={k_value}...")
        fpr, tpr, thresholds = metrics.roc_curve(labels.tolist(), scores.tolist())
        print(fpr)
        print(tpr)
        auc = metrics.auc(fpr, tpr)
        print("auc val is " + str(auc))
        k_to_auc[k_value] = auc
        RocCurveDisplay.from_predictions(
            labels.tolist(),
            scores.tolist(),
            name=f"ood vs id",
            color="darkorange",
        )
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("k = " + str(k_value))
        plt.show()
        # plt.savefig(args.output_dir + "/statistics/ROC_CURVE_OOD_VS_ID")
    import json
    with open(os.path.join(args.output_dir, 'k_to_auc_supervised.json'), 'w') as f:
        json.dump(k_to_auc, f, indent=4)


