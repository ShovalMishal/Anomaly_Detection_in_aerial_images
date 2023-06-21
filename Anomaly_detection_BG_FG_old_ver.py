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
def calc_patches_indices(args, image_size):
    patches_indices = []
    curr_image_size = image_size
    for i in range(args.pyramid_levels):
        scale = args.scale_factor ** i
        curr_scale_patches_indices = compute_patches_indices_per_scale(curr_image_size,
                                                                       (args.patch_size, args.patch_size),
                                                                       scale_factor=scale)
        curr_image_size = (int(args.scale_factor * curr_image_size[0]), int(args.scale_factor * curr_image_size[1]))
        patches_indices.append(curr_scale_patches_indices)
    patches_indices_tensor = []
    for patch_ind in patches_indices:
        patch_ind = patch_ind.view(patch_ind.shape[0], 4)
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


def create_gaussian_pyramid(data):
    current_level = torch.stack(data).float()
    pyramid_batch = [current_level]
    # show_img(current_level[0])
    # create gaussian pyramid
    for level in range(args.pyramid_levels - 1):
        blurred = TF.gaussian_blur(current_level, kernel_size=3, sigma=1)
        downsampled = F.interpolate(blurred, scale_factor=args.scale_factor, mode='bilinear', align_corners=False)
        pyramid_batch.append(downsampled)
        # show_img(downsampled[0])
        current_level = downsampled
    return pyramid_batch


def calculate_scores_and_labels(test_features, labels, train_features, k):
    # set 1 for foreground and 0 for background
    labels[torch.nonzero(labels >= 0)] = 1
    labels[torch.nonzero(labels == -1)] = 0
    # scores = sklearn.metrics.pairwise_distances_argmin_min(test_features.cpu().numpy(), train_features.cpu().numpy())
    # scores = scores[1]
    scores = calculate_scores(test_features, train_features, k=k)
    return scores, labels


def calculate_scores(test_features, train_features, k=3):
    diff = torch.cdist(test_features, train_features)
    # Sort the differences along the second dimension
    sorted_diff, indices = torch.sort(diff, dim=1)
    # Select the k closest values
    closest_values = sorted_diff[:, :k]
    scores = torch.mean(closest_values.float(), dim=1)
    return scores


def get_gt_bboxes_and_labels(gt_instances, image):
    bboxes = []
    labels = []
    for bbox, label in zip(gt_instances.bboxes, gt_instances.labels):
        bbox = torch.squeeze(bbox.tensor)
        min_x = max(0, bbox[0] - bbox[2]/2)
        max_x = min(image.shape[1], bbox[0] + bbox[2] / 2)
        min_y = max(0, bbox[1] - bbox[3] / 2)
        max_y = min(image.shape[2], bbox[1] + bbox[3] / 2)
        square_length_y = abs(max_x - min_x) - abs(max_y - min_y) if abs(max_x - min_x) > abs(max_y - min_y) else 0
        square_length_x = abs(max_y - min_y) - abs(max_x - min_x) if abs(max_x - min_x) < abs(max_y - min_y) else 0
        cropped_img = image[:, int(min_y - square_length_y / 2):int(max_y + square_length_y / 2),
                      int(min_x - square_length_x / 2):int(max_x + square_length_x / 2)]
        if cropped_img.shape[1]!=0 and cropped_img.shape[2]!=0:
            bboxes.append(cropped_img)
            labels.append(label)
    return bboxes, torch.tensor(labels)


def calculate_batch_features_and_labels(data_batch):
    pyramid_batch = create_gaussian_pyramid(data_batch['inputs'])
    # create patches
    patches_accord_level = []
    for level in range(args.pyramid_levels):
        patches = split_images_into_patches(pyramid_batch[level], (args.patch_size, args.patch_size))
        patches_accord_level.append(patches)
    images_patches = torch.cat(patches_accord_level, dim=1)
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
        indices_to_keep = torch.squeeze(torch.nonzero(assign_result.gt_inds == 0))
        filtered_labels = assign_result.labels[indices_to_keep]
        filtered_image_patches = images_patches[image_idx][indices_to_keep]
        prep_patches_per_image = preprocess_patches(filtered_image_patches)
        gt_bboxes, gt_labels = get_gt_bboxes_and_labels(batch_gt_instances[image_idx], data_batch['inputs'][image_idx])
        filtered_labels = torch.concat((filtered_labels, gt_labels), dim=0)
        # extract features for each patch
        prep_gt_bboxes = [preprocess_patches(torch.unsqueeze(gt_bbox, dim=0).float()) for gt_bbox in gt_bboxes]
        prep_gt_bboxes = torch.concat(prep_gt_bboxes, dim=0) if len(prep_gt_bboxes) > 0 else torch.tensor([])

        prep_all_patches = torch.concat((prep_patches_per_image, prep_gt_bboxes), dim=0).to(device)
        with torch.no_grad():
            output = features_model(prep_all_patches)
            output = torch.flatten(output, 1)

        patches_labels.append(filtered_labels)
        images_patches_features.append(output)
    images_patches_features = torch.concat(images_patches_features, dim=0)
    patches_labels = torch.concat(patches_labels, dim=0)
    return images_patches_features, patches_labels


def printing_data_statistics(labels, is_train):
    fg_number = torch.nonzero(labels != 0).size()[0]
    bg_number = torch.nonzero(labels == 0).size()[0]
    title = "train" if is_train else "test"
    print(title + " data contain " + str(fg_number) + " fg samples and " + str(bg_number) + " bg samples\n")


if __name__ == '__main__':
    t = time.time()
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="The relative path to the cfg file")
    parser.add_argument("-o", "--output_dir", help="The saved model path")
    parser.add_argument("-l", "--pyramid_levels", default=5, help="Number of pyramid levels")
    parser.add_argument("-sf", "--scale_factor", default=0.6, help="Scale factor between pyramid levels")
    parser.add_argument("-ps", "--patch_size", default=50, help="The patch size")
    parser.add_argument("-k", "--k_value", default=1, help="The patch size")
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    image_size = cfg["train_pipeline"][3]['scale']
    formatted_patches_indices = calc_patches_indices(args, image_size)
    bbox_assigner = TASK_UTILS.build(cfg["patches_assigner"])
    model = models.resnet50(pretrained=True)
    model.eval()
    model = model.to(device)
<<<<<<< HEAD
    features_model = torch.nn.Sequential(*list(model.children())[:-1])
=======
    features_model = torch.nn.Sequential(*list(model.children())[:-1]).eval()
>>>>>>> 91a906d (anomaly expirement and some initial arragments)

    # Create features for the training stage in a case it does not exist
    if not os.path.exists(args.output_dir + '/features_dict_old.pt'):
        data_loader = create_dataloader(cfg)
        features = []
        labels = []
        for idx, data_batch in enumerate(data_loader):
            if data_batch['data_samples'][0].img_id in images_to_use:
                images_to_use.remove(data_batch['data_samples'][0].img_id)
                images_patches_features, patches_labels = calculate_batch_features_and_labels(data_batch)
                features.append(images_patches_features)
                labels.append(patches_labels)
            elif len(images_to_use) == 0:
                break
        features = torch.concat(features, dim=0)
        labels = torch.concat(labels, dim=0)
        labels[torch.nonzero(labels >= 0)] = 1
        labels[torch.nonzero(labels == -1)] = 0
        data = {'features': features, 'labels': labels}
        torch.save(data, args.output_dir + '/features_dict_old.pt')

    # Test stage
    data = torch.load(args.output_dir + '/features_dict_old.pt')
    printing_data_statistics(data['labels'], is_train=True)
    val_data_loader = create_dataloader(cfg, mode='val')
    scores = []
    labels = []
    for idx, data_batch in enumerate(val_data_loader):
        if idx == 1000:
            break
        images_patches_features, patches_labels = calculate_batch_features_and_labels(data_batch)
        batch_scores, batch_labels = calculate_scores_and_labels(images_patches_features, patches_labels,
                                                                 data["features"], int(args.k_value))
        scores.append(batch_scores)
        labels.append(batch_labels)
    scores = torch.concat(scores, axis=0)
    labels = torch.concat(labels, dim=0)
    printing_data_statistics(labels, is_train=False)
    # 1 to ood 0 to objects
    fpr, tpr, thresholds = metrics.roc_curve(labels.tolist(), scores.tolist())
    print(fpr)
    print(tpr)
    print("auc val is " + str(metrics.auc(fpr, tpr)))
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
    plt.title("k = "+str(args.k_value))
    plt.show()
    plt.savefig(args.output_dir + "/statistics/ROC_CURVE_OOD_VS_ID")
