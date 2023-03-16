import torch
import sys
sys.path.insert(1, '/home/alex/facade/SSIW/Test_Minist/tools/')
from sklearn.metrics import confusion_matrix
import test as t
import numpy as np
from utils.transforms_utils import get_imagenet_mean_std, normalize_img, pad_to_crop_sz, resize_by_scaled_short_side
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from utils.color_seg import color_seg

class PredictionArgs():
    def __init__(self, test_w, test_h):
        self.test_w = test_w
        self.test_h = test_h

def plot_predictions(idx, x, y, class_labels, model, gt_embs_list, single_args):
    # cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, 12)))
    cmap = plt.cm.get_cmap('viridis', 11)
    handles = []
    for i, name in enumerate(class_labels):
        colour = cmap(i)
        label = class_labels[i]
        patch = mpatches.Patch(color=colour, label=label)
        handles.append(patch)

    prediction = predict(x, model, gt_embs_list, single_args)

    fig, axs = plt.subplots(1, 5, figsize=(12, 2.3))
    axs = axs.ravel()
    for ax in axs:
        ax.set_axis_off()
    axs[0].imshow(x / 255)
    axs[0].text(0, 1, f"idx={idx}", fontsize=8, ha="center", va="bottom")
    axs[1].imshow(prediction + 1, cmap=cmap)
    pred_color = color_seg(prediction + 1)
    vis_seg = t.visual_segments(pred_color, x.numpy().astype(np.uint8))
    axs[4].imshow(vis_seg)
    axs[3].legend(handles=handles, loc="center left", ncol=1, fontsize=8)
    axs[2].imshow(y, cmap=cmap)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def plot_cm(cm, class_labels):
    plt.figure(figsize=(5, 5))
    total_samples = np.sum(cm)
    sns.heatmap(cm/np.sum(cm, axis=1)[:, np.newaxis],
                annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False,annot_kws={"fontsize":7})
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.xlabel('Pred', fontsize=10)
    plt.ylabel('GT', fontsize=10)
    plt.show()
    return

def compute_iou(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(num_classes) + 1)
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = np.zeros(num_classes)
    for c in range(num_classes):
        if union[c] == 0:
            iou[c] = 0
        else:
            iou[c] = intersection[c] / union[c]
    return cm, iou

def compute_pixel_accuracy(y_true, y_pred):
    num_correct_pixels = np.sum(y_true == y_pred)
    num_total_pixels = y_true.size
    pixel_accuracy = num_correct_pixels / num_total_pixels
    return pixel_accuracy

def predict(x, model, gt_embs_list, single_args):
    h, w, _ = x.shape
    model.eval()
    with torch.no_grad():
        x_resized = resize_by_scaled_short_side(x.numpy(), 720, 1)
        out_logit = t.single_scale_single_crop_cuda(model, x_resized, h, w, gt_embs_list=gt_embs_list, args=single_args)
        prediction = out_logit.argmax(axis=-1).squeeze()
        return prediction


def get_test_set_metrics(dataset, test_split, model, gt_embs_list, single_args):
    img_metrics = {}
    class_iou_sum = {c: 0 for c in range(12)}
    class_iou_count = {c: 0 for c in range(12)}
    num_images = 0
    cm_total = np.zeros((12, 12))
    iou_total = 0

    idx_start = int(len(dataset)*(1-test_split))
    for idx in range(idx_start, len(dataset)):
        x = dataset[idx][0]
        y = dataset[idx][1]

        prediction = predict(x, model, gt_embs_list, single_args)
        cm, ious = compute_iou(y.numpy(), prediction + 1, 12)
        acc = compute_pixel_accuracy(y.numpy(), prediction+1)
        img_metrics[idx] = {'ious': ious, 'cm': cm, 'acc': acc}

        cm_total += cm
        iou_total += np.mean(ious)
        num_images += 1

        for c in range(12):
            if np.sum(cm[c, :]) > 0:
                class_iou_sum[c] += ious[c]
                class_iou_count[c] += 1

    mean_iou = iou_total / num_images
    class_iou_avg = {c: class_iou_sum[c] / class_iou_count[c] for c in range(12)}

    return cm_total, mean_iou, class_iou_avg, img_metrics

