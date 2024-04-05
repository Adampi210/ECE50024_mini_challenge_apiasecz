import os
import csv
import shutil
from facenet_pytorch import MTCNN
import torch
from PIL import Image

dev_used = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Move the files into a correct directory structure (same as VGG)
def move_files_to_correct_structure(img_data_dir, img_label_file, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    # Read the data file paths and labels
    data_fpaths = []
    with open(img_label_file, 'r') as img_label_f:
        reader = csv.reader(img_label_f)
        next(reader)  # Skip the header
        for row in reader:
            img_fname = row[1]
            img_path = os.path.join(img_data_dir, img_fname)
            img_label = row[2]
            data_fpaths.append((img_path, img_label))

    # Move files to the new directory structure
    i = 0
    for img_path, img_label in data_fpaths:
        i += 1
        # Create the class directory if it doesn't exist
        class_dir = os.path.join(output_dir, img_label)
        os.makedirs(class_dir, exist_ok = True)

        # Move the file to the corresponding class directory
        filename = os.path.basename(img_path)
        destination_path = os.path.join(class_dir, filename)
        shutil.copy(img_path, destination_path)

    print("Files moved successfully to the new structure.")
    print(f"Moved {i} files")

# Get cropped files for training
def get_cropped_files(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    # Use MTCNN for face detection
    mtcnn = MTCNN(device = dev_used, select_largest = False)
    i = 0
    couldnt_save_files = 0
    # Go over all labels
    for label_dir in os.listdir(data_dir):
        img_data_dir = os.path.join(data_dir, label_dir)
        new_data_dir = os.path.join(output_dir, label_dir)
        os.makedirs(new_data_dir, exist_ok = True)
        # Go over images
        for img_file in os.listdir(img_data_dir):
            # Get cropped image each
            img_path = os.path.join(data_dir, img_file)
            new_path = os.path.join(output_dir, img_file)
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img_final = img.convert('RGB')
            else:
                img_final = img
            if i == 0:
                init_img = img_final
            x = mtcnn(img_final, save_path = new_path)
            
            # If initially couldn't get the face, try with smaller face size
            min_face_size = 20
            while x is None:
                if min_face_size < 3:
                    break
                min_face_size -= 1
                print('Couldn\'t save image')
                mtcnn_new = MTCNN(min_face_size = min_face_size, select_largest = False, device = dev_used)
                x = mtcnn_new(img_final, save_path = new_path)
                
            # Then try with selecting largest face
            min_face_size = 20
            while x is None:
                if min_face_size < 3:
                    break
                min_face_size -= 1
                print('Couldn\'t save image')
                mtcnn_new = MTCNN(min_face_size = min_face_size, select_largest = True, device = dev_used)
                x = mtcnn_new(img_final, save_path = new_path)
            
            # Finally, if all else failed, just save the initial image
            if x is None:
                x = mtcnn_new(init_img, save_path = new_path)
                print(f'Couldn\'t save {img_file}, saving init image')
                couldnt_save_files += 1
            else:
                print(f'Saved {img_file}')
            i += 1
        print(f'Couldn\'t save {couldnt_save_files} files out of {i} files')
        print(f'Cropped {i} files')

# Get cropped files for testing
def get_test_cropped_files(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    mtcnn = MTCNN(device = dev_used, select_largest = False)
    i = 0
    max_retries = 10
    sorted_files = sorted(os.listdir(data_dir), key = lambda x: int(os.path.splitext(x)[0]))
    # Go over all labels
    couldnt_save_files = 0
    for img_file in sorted_files:
        # Get cropped image each
        img_path = os.path.join(data_dir, img_file)
        new_path = os.path.join(output_dir, img_file)
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img_final = img.convert('RGB')
        else:
            img_final = img
        if i == 0:
            init_img = img_final
        x = mtcnn(img_final, save_path = new_path)
        
        # If initially couldn't get the face, try with smaller face size
        min_face_size = 20
        while x is None:
            if min_face_size < 3:
                break
            min_face_size -= 1
            print('Couldn\'t save image')
            mtcnn_new = MTCNN(min_face_size = min_face_size, select_largest = False, device = dev_used)
            x = mtcnn_new(img_final, save_path = new_path)
            
        # Then try with selecting largest face
        min_face_size = 20
        while x is None:
            if min_face_size < 3:
                break
            min_face_size -= 1
            print('Couldn\'t save image')
            mtcnn_new = MTCNN(min_face_size = min_face_size, select_largest = True, device = dev_used)
            x = mtcnn_new(img_final, save_path = new_path)
        
        # Finally, if all else failed, just save the initial image
        if x is None:
            x = mtcnn_new(init_img, save_path = new_path)
            print(f'Couldn\'t save {img_file}, saving init image')
            couldnt_save_files += 1
        else:           
            print(f'Saved {img_file}')
        i += 1
    print(f'Cropped {i} files')
    print(f'Couldn\'t save {couldnt_save_files} files out of {i} files')
 
if __name__ == '__main__':
    print(f'Running on device: {dev_used}')
    # Move files to new directory structure
    # move_files_to_correct_structure('../../../data/datasets/mini_challenge_large/train/', './data/train.csv', '../../../data/datasets/mini_challenge_train_orig/')
    
    # Get cropped files training
    # get_cropped_files('../../../data/datasets/mini_challenge_train_orig/', '../../../data/datasets/mini_challenge_train_crop/')
    
    # Get cropped files testing
    # get_test_cropped_files('../../../data/datasets/mini_challenge_test/test/', '../../../data/datasets/mini_challenge_test_crop/')
    
    pass