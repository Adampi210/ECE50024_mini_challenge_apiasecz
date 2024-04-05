from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import os

# Partially based on the tutorial on finetuning from https://github.com/timesler/facenet-pytorch

dev_used = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = '../../../data/datasets/mini_challenge_train_crop/'
BATCH_SIZE = 100
num_epochs = 8
LR = 1e-3
frac_train = 0.95

# Data transform
transform_data = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

if __name__ == '__main__':
    # Get the training and validation data
    data_set = datasets.ImageFolder(root = data_dir, transform = transform_data)
    img_idx = np.arange(len(data_set))
    np.random.shuffle(img_idx)
    train_idx = img_idx[:int(frac_train * len(img_idx))]
    val_idx = img_idx[int((1 - frac_train) * len(img_idx)):]

    # Dataloaders
    train_loader = DataLoader(
        data_set,
        batch_size = BATCH_SIZE,
        sampler = SubsetRandomSampler(train_idx)
    )
    valid_loader = DataLoader(
        data_set,
        batch_size = BATCH_SIZE,
        sampler = SubsetRandomSampler(val_idx)
    )

    # Model (also load from memory)
    celeb_class_model = InceptionResnetV1(
        classify = True, 
        pretrained = 'vggface2',
        num_classes = len(data_set.class_to_idx)
    ).to(dev_used)
    # torch.save(celeb_class_model.state_dict(), '../../../data/model_params/mini_challenge_models/new_facenet_params_in_train.pth')
    celeb_class_model.load_state_dict(torch.load('../../../data/model_params/mini_challenge_models/new_facenet_params_after_train.pth'))

    # Hyperparams
    optimizer = optim.Adam(celeb_class_model.parameters(), lr = LR)
    scheduler = MultiStepLR(optimizer, [5, 10])
    loss_fn = torch.nn.CrossEntropyLoss()
    batch_metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }
    
    # Train loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')

        # Train
        celeb_class_model.train()
        training.pass_epoch(
            celeb_class_model, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics = batch_metrics, show_running = True, device = dev_used
        )
        # Save after each train round
        torch.save(celeb_class_model.state_dict(), '../../../data/model_params/mini_challenge_models/new_facenet_params_in_train.pth')

        # Validate
        celeb_class_model.eval()
        training.pass_epoch(
            celeb_class_model, loss_fn, valid_loader,
            batch_metrics = batch_metrics, show_running = True, device = dev_used
        )
    # Save final moodel
    torch.save(celeb_class_model.state_dict(), '../../../data/model_params/mini_challenge_models/new_facenet_params_after_train.pth')