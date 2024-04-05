from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import csv
import os
from PIL import Image
import numpy as np

# Partially based on the tutorial on finetuning from https://github.com/timesler/facenet-pytorch

dev_used = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_data_dir = '../../../data/datasets/mini_challenge_test_crop/'

# Data transform (same as training)
transform_data = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# Create a class for the testing data
class TestDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        # Need to get them in original order
        self.image_files = sorted(os.listdir(root_dir), key = lambda x: int(os.path.splitext(x)[0]))

    # To get the img
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # In case, convert to RGB (should be RGB by default)
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image

if __name__ == '__main__':
    # Testing data
    test_dataset = TestDataset(root_dir = test_data_dir, transform = transform_data)

    # Model (same as train, load after training)
    celeb_class_model = InceptionResnetV1(
        classify = True,
        pretrained = 'vggface2',
        num_classes = 100  # Set the number of classes to the total number of celebrities in your training data
    ).to(dev_used)

    celeb_class_model.load_state_dict(torch.load('../../../data/model_params/mini_challenge_models/new_facenet_params_after_train.pth'))

    # Testing
    celeb_class_model.eval()
    predictions = []

    # Dataloader for test dataset
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    i = 0
    # Predict
    with torch.no_grad():
        for idx, face in enumerate(test_loader):
            print(f'Face {i}')
            i += 1
            # Only run through the model if could find the face
            if face is not None:
                face = face.to(dev_used)
                output = celeb_class_model(face)
                _, predicted = torch.max(output.data, 1)
                predicted_class = predicted.item()
                predictions.append([idx, predicted_class])
            else:
                predictions.append([idx, -1])  # Append -1 if no face is detected
                
    # Load the class mapping from the training data
    train_data_dir = '../../../data/datasets/mini_challenge_train_crop/'
    train_dataset = datasets.ImageFolder(root = train_data_dir, transform = transform_data)
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    # Convert predicted class indices to class names
    predictions = [[idx, idx_to_class[class_idx]] if class_idx != -1 else [idx, "no class"] for idx, class_idx in predictions]

    # Save predictions
    pred_file = './data/predicted_labels.csv'
    with open(pred_file, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id', 'Category'])
        writer.writerows(predictions)