import torch
import torch.nn as nn
import os, sys
import cv2
import numpy as np
import tqdm
from PIL import Image
from pathlib import Path
from time import gmtime, strftime
# Import test.py from repo
sys.path.insert(1, '/home/alex/facade/SSIW/Test_Minist/tools/')
import test as t
from utils.get_class_emb import create_embs_from_names
from utils.segformer import get_configured_segformer

from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

import torch.optim as optim

class CMPDataset(Dataset):
    def __init__(self, root_dir, transform=None, resize=True):
        self.root_dir = root_dir
        self.transform = transform
        dataset_files = os.listdir(root_dir)
        self.file_names = sorted(set([Path(file).stem for file in dataset_files]))
        self.resize = resize
        self.original_sizes = {}

        # From imagenet, as in paper
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.file_names[index] + ".jpg")
        img = cv2.imread(img_path, -1)[:, :, ::-1]

        label_path = os.path.join(self.root_dir, self.file_names[index] + ".png")
        label = Image.open(label_path)
        label = np.array(label)

        h, w = img.shape[:2]
        self.original_sizes[index] = (h, w)

        if self.resize:
            img = cv2.resize(img, (520, 520), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (520, 520), interpolation=cv2.INTER_NEAREST)

        # Apply transformations
        if self.transform is not None:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']

        img = self.normalize(img)

        # Convert label to tensor and return image and label
        label = torch.from_numpy(label).long()
        img = torch.from_numpy(np.ascontiguousarray(img)).to(torch.float32)
        return img, label

    def get_batch(self, indices):
        # Load images and labels for the batch
        batch_images = []
        batch_labels = []
        for index in indices:
            img, label = self.__getitem__(index)
            print(f'img: {img.shape}')
            print(f'label: {label.shape}')
            # print(f'manual transpose of img: {np.transpose(img, (2,0,1)).shape}')
            # img = np.transpose(img, (2, 0, 1))
            # print('transpose done')
            print(f'new img shape {img.shape}')
            batch_images.append(img)
            batch_labels.append(label)

        print(batch_images)
        batch_images = torch.stack(batch_images, dim=0)
        print(f'b_images shape: {batch_images.shape}')
        batch_labels = torch.stack(batch_labels, dim=0)

        return batch_images, batch_labels

    def get_original_size(self, index):
        return self.original_sizes[index]


def get_prediction_tau(embs, gt_embs_list, tau):
    prediction = []
    logits = []
    B, _, _, _ = embs.shape
    for b in range(B):
        score = embs[b, ...]
        score = score.unsqueeze(0)
        emb = gt_embs_list
        emb = emb / emb.norm(dim=1, keepdim=True)
        score = score / score.norm(dim=1, keepdim=True)
        score = score.permute(0, 2, 3, 1) @ emb.t()
        # [N, H, W, num_cls] You maybe need to remove the .t() based on the shape of your saved .npy
        score = score / tau
        score = score.permute(0, 3, 1, 2)  # [N, num_cls, H, W]
        prediction.append(score.max(1)[1])
        logits.append(score)
    if len(prediction) == 1:
        prediction = prediction[0]
        logit = logits[0]
    else:
        prediction = torch.cat(prediction, dim=0)
        logit = torch.cat(logits, dim=0)
    return logit

class PredictionCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=255)
        # self.temperature = nn.Parameter(torch.tensor([1.0]), requires_grad=True).to(device)

    def forward(self, emb, gt_embs_list, labels):
        logits = t.get_prediction(emb, gt_embs_list)
        loss = self.cross_entropy(logits, labels - 1)
        return loss

def evaluate(model, criterion, test_loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.permute(0, -1, 1, 2).to(torch.float32)
            images = images.to(device)
            labels = labels.to(device)

            emb, _, _ = model(inputs=images, label_space=['universal'])
            total_loss += criterion(emb, gt_embs_list, labels).item()

        total_loss /= len(test_loader)
        print('Test set: Average loss: {:.4f}'.format(total_loss))
        return total_loss

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set up data
    dataset_dir = '/home/alex/facade/CMP_facade'
    with open(os.path.join(dataset_dir, 'label_names.txt')) as file:
        class_dict = {line.split()[0]: line.split()[1] for line in file}
    class_labels = list(class_dict.values())
    dataset = CMPDataset(os.path.join(dataset_dir, 'base'), resize=True)

    # Define sentences:
    sentences = {
        'background': 'the part of a picture, scene, or design that forms a setting for the main figures or objects, or appears furthest from the viewer.',
        'facade': 'the principal front of a building, that faces on to a street or open space.',
        'window': 'an opening in the wall or roof of a building or vehicle, fitted with glass in a frame to admit light or air and allow people to see out.',
        'door': 'a hinged, sliding, or revolving barrier at the entrance to a building, room, or vehicle, or in the framework of a cupboard.',
        'cornice': 'which is generally any horizontal decorative moulding that crowns a building or furniture element for example, the cornice over a door or window',
        'sill': 'a decorative (raised) panel or stripe under a window',
        'balcony': 'a platform projecting from the wall of a building, supported by columns or console brackets, and enclosed with a balustrade, usually above the ground floor.',
        'blind': 'a type of functional window covering used to obstruct light, made of hard or soft material; i.e. shutters, roller shades, wood blinds, standard vertical, and horizontal blinds',
        'deco': 'a piece of original art, paintings, reliefs, statues.',
        'molding': 'a horizontal decorative stripe across the facade, possibly with a repetitive texture pattern, used to cover transitions between surfaces or for decoration',
        'pillar': 'a vertical decorative stripe across the facade which is made of stone, or appearing to be so.',
        'shop': 'a building or part of a building where goods or services are sold or advertised.'}

    # For consistency with paper include this preamble
    def add_preamble(x): return f'This is a picture of a {x}, '
    sentences = {k: add_preamble(k)+v for (k,v) in sentences.items()}

    gt_embs_list = create_embs_from_names(class_labels, other_descriptions=sentences).float()
    gt_embs_list = gt_embs_list.to(device)
    id_to_label = {i: v for i, v in enumerate(class_labels)}
    id_to_label[255] = 'unlabel'

    num_model_classes = 512 # to initially load in the weights
    model = get_configured_segformer(num_model_classes, criterion=None, load_imagenet_model=False)
    model = torch.nn.DataParallel(model)
    ckpt_path = '/home/alex/facade/SSIW/Test_Minist/models/segformer_7data.pth'
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    model.load_state_dict(ckpt_filter, strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = True
        #if 'auxi_net' not in name and 'head' not in name:
        #    param.requires_grad = False

    model = model.to(device)

    batch_size = 8
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    criterion = PredictionCriterion()
    scheduler = StepLR(optimizer, step_size=20, gamma=0.01)
    accumulation_steps = 3

    min_loss = float('inf')
    losses = []

    train_losses = []
    batch_losses = []
    eval_losses = []

    model.train()

    scaler = GradScaler()
    save_itr = 5
    save_dir = strftime("%Y%m%d%H%M", gmtime())

    for epoch in range(100):

        for batch, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):

            # N, C, , ,

            mean, std = get_imagenet_mean_std()

            images = images.permute(0, -1, 1, 2).to(torch.float32)
            images = images.half().to(device)
            labels = labels.to(device)

            # optimizer.zero_grad()

            with autocast():
                emb, _, _ = model(inputs=images, label_space=['universal'])
                loss = criterion(emb, gt_embs_list, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch + 1) % accumulation_steps == 0:
                print('Update loss')
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                batch_losses.append(loss.item())


            print(f" Epoch :{epoch} Images {batch*batch_size}/{train_size}. Loss: {loss.item()} Lr: {scheduler.get_last_lr()}")

        scheduler.step()
        train_losses.append(loss.item())
        eval_loss = evaluate(model, criterion, test_loader, device)
        eval_losses.append(eval_loss)

        if epoch + 1 % save_itr == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            print(f'Save to {save_dir}')
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_FT_320_weights_epoch{epoch + 1}.pt"))

            import csv
            with open(os.path.join(save_dir, 'epoch_losses.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['train_loss', 'eval_loss'])
                for train_loss, eval_loss in zip(train_losses, eval_losses):
                    writer.writerow([train_loss, eval_loss])

            with open(os.path.join(save_dir, 'batch_loss.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['train_loss', 'eval_loss'])
                for batch_losses, eval_loss in zip(batch_losses, eval_losses):
                    writer.writerow([batch_losses, eval_loss])