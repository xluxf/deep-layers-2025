import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from matplotlib import pyplot as plt
import mlflow
from tqdm import tqdm


from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image
import glob

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.has_masks = mask_dir is not None and os.path.exists(mask_dir)
        
        self.images = self._load_images(self.image_dir)
        self.masks = self._load_images(self.mask_dir) if self.has_masks else []

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self.images)} images")
            
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        
        # Resize to target size
        resize_transform = transforms.Resize((1000, 1000))
        image = resize_transform(image)
        
        # Convert to tensor
        image = transforms.ToTensor()(image)

        # Load mask if available
        if self.has_masks:
            mask_path = self.masks[idx]
            mask = Image.open(mask_path).convert('L')
            mask = resize_transform(mask)
            mask = transforms.ToTensor()(mask)
    
        # transform both image and mask if transform is provided
        if self.transform:
            # Apply same transform to both image and mask
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)

            if self.has_masks:
                torch.manual_seed(seed)
                mask = self.transform(mask)

        # return image and mask if available
        if self.has_masks:
            return image, mask
        else:
            return image
        
    def _load_images(self, directory):
        # load all image file paths from the directory

        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
        image_paths.sort()  # ensure consistent order
        return image_paths
    

def train_model(model, 
                training_data,
                validation_data,
                name='my_model.pth',
                stopping_condition='val_accuracy',
                steps_per_epoch=150,
                verbose=False,
                epochs=16):
    ' procedure to train model using fixed setting and callbacks '

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 6
    
    mlflow.start_run()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(training_data):
            if batch_idx >= steps_per_epoch:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.long().squeeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.numel()
            train_correct += (predicted == targets).sum().item()
        
        train_accuracy = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in validation_data:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.long().squeeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.numel()
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = val_correct / val_total
        
        # Log metrics
        mlflow.log_metric("train_loss", train_loss / steps_per_epoch, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss / len(validation_data), step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss/steps_per_epoch:.4f}, '
                  f'Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss/len(validation_data):.4f}, '
                  f'Val Acc: {val_accuracy:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), name)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    mlflow.end_run()
    
    # Load best model
    model.load_state_dict(torch.load(name))
    return model


from skimage import io, morphology, transform, segmentation, measure, color
from skimage.feature import peak_local_max
from matplotlib import cm

def get_labels(pred, shuffle=False):
    
    # find local maxima
    im = morphology.erosion(pred, morphology.disk(3))
    peak_idx = peak_local_max(im, min_distance=5, threshold_rel=.2)
    markers = np.zeros_like(pred, dtype=bool)
    markers[tuple(peak_idx.T)] = True

    # segment
    labels = segmentation.watershed(-pred, measure.label(markers), mask=pred>.1)
        
    return labels



def save_results(res, store_dir='results', img_dir='BF-C2DL-HSC/test'):

    # get input images
    img_names = os.listdir(img_dir)
    img_names.sort()

    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)

    for i in tqdm(range(len(res))):
        
        # read image
        img_path = os.path.join(img_dir, img_names[i] )
        img = io.imread(img_path)
        img = transform.resize(img, (1000, 1000))
        
        # get prediction
        pred = res[i, :, :, 1]

        # get labels
        labels = get_labels(pred)
        store_path = os.path.join(store_dir, f'res{i+1700:04d}.jpg')

        # visialize
        cmap = cm.get_cmap('tab20b').copy()
        cmap.set_bad(color='black')

        color_mask = color.label2rgb(labels,
                               colors=cmap.colors,
                               image=img,
                               bg_label=0,
                               alpha=.4)

        # save result
        io.imsave(store_path, (color_mask*255).astype(np.uint8), check_contrast=False)
        


def show_predictions(model,
                     dataset,
                     idx=0):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # get index of the batch and index of the image in the batch
    batch_size = next(iter(dataset))[0].shape[0]
    img_idx, batch_idx = idx % batch_size, idx // batch_size
    
    # take input batch
    for i, (im, gt) in enumerate(dataset):
        if i == batch_idx:
            break
    
    # predict batch
    with torch.no_grad():
        im = im.to(device)
        res = model(im)
        res = torch.softmax(res, dim=1)
        res = res.cpu().numpy()

    # get input and result
    labels = ['input image', 'ground truth', 'prediction']   
    images = [inverse_preprocess_input(im[img_idx, :, :].cpu().numpy()),
              gt[img_idx, :, :].numpy(),
              np.argmax(res[img_idx, :, :], axis=0)]

    # plot result
    plot_in_grid(images, labels, n_cols=3, title=f' Image index: {str(idx)}')
    
    
def show_augmentation(dataset, idx, n_samples=8):
    
    batch_size = next(iter(dataset))[0].shape[0]
    img_idx, batch_idx = idx % batch_size, idx // batch_size
    
    images, gts = [], []

    for i in range(n_samples):
        for j, (im, gt) in enumerate(dataset):
            if j == batch_idx:
                break

        # de-preprocess image
        img = im[img_idx, :, :].numpy()
        img = inverse_preprocess_input(img)

        images.append(img)
        images.append(gt[img_idx, :, :])

    plot_in_grid(images, [])


def set_device():
    print("=== PyTorch GPU Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory
        
    else:
        print("CUDA not available. Reasons could be:")
        print("1. No NVIDIA GPU installed")
        print("2. CUDA drivers not installed")
        print("3. PyTorch not compiled with CUDA support")
        print("4. Incompatible CUDA/PyTorch versions")
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    print("================================")
    return device


def test_gpu_functionality():
    """Test basic GPU operations to ensure everything works"""
    print("\n=== Testing GPU Functionality ===")
    
    if not torch.cuda.is_available():
        print("GPU not available - skipping GPU tests")
        return
    
    try:
        # Test basic tensor operations on GPU
        device = torch.device('cuda')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        z = torch.mm(x, y)
        
        print("✓ Basic GPU tensor operations: SUCCESS")
        
        # Test memory allocation
        current_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"✓ Current GPU memory usage: {current_memory:.1f} MB")
        
        # Clear memory
        del x, y, z
        torch.cuda.empty_cache()
        print("✓ GPU memory cleanup: SUCCESS")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
    
    print("===============================\n")


def inverse_preprocess_input(img):
    # rescale
    img = img + np.array([[[ 103.939, 116.779, 123.68 ]]])
           
    # cut off overflowing values
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    
    # BGR to RGB
    r, g, b = np.split(img, 3, axis=-1)
    img = np.concatenate([b, g, r], axis=2)
    
    return img


def ishow(img,
          cmap='viridis',
          title='',
          fig_size=(8,6),
          colorbar=True,
          interpolation='none'):
    ' Function `ishow` displays an image in a new window. '
    
    extent = (0, img.shape[1], img.shape[0], 0)
    fig, ax = plt.subplots(figsize=fig_size)
    pcm = ax.imshow(img,
              extent=extent,
              cmap=cmap,
              interpolation=interpolation)
    
    ax.set_frame_on(False)
    plt.title(title)
    plt.tight_layout()
    if colorbar:
        
        fig.colorbar(pcm, orientation='vertical')
    plt.show()
        
    
# old
        
        
def plot_training_history(train_log):
    ' plot training history '
    names = train_log.history.keys()

    for name in names:
        plt.plot(train_log.history[name])

    plt.title("training history")
    plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.legend(names)
    plt.show()
        
        
def plot_in_grid(images,
                 names,
                 n_cols=4,
                 title='',
                 cmap='viridis',
                 fixed_range=False,
                 colorbar=False,
                 pad_images=False):
    '''
    Plots grid of images
    
    : param images : list of numpy arrays
    : param names : list of strings, names of figures
    : param n_cols : n cols in a grid
    : param title : string of the whle figure name
    : param fixed_range : False or a tuple (min_value, max_value)
    '''
        
    if len(images) != len(names):
        names = [''] * len(images)
    
    n_samples = len(images)
    n_rows = int(np.ceil(n_samples / 4)) 
    
    fig, axes = plt.subplots(figsize=(15,n_rows*4),
                         nrows=n_rows,
                         ncols=n_cols)
    fig.suptitle(title, fontsize=12)

    ax = axes.ravel()
    for i, (img, title) in enumerate(zip(images, names)):
        if fixed_range:
            vmin, vmax = fixed_range
        else:
            vmin, vmax = np.min(img), np.max(img)
        if pad_images:
            img = np.pad(img, ((1, 1), (1, 1)))
        extent = (0, img.shape[1], img.shape[0], 0)
        pcm = ax[i].imshow(img,
                           cmap=cmap,
                           interpolation="none",
                           vmin=vmin,
                           vmax=vmax,
                           extent=extent)
        ax[i].set_title(title, fontsize=10)
        
        if colorbar:
            fig.colorbar(pcm, ax=ax[i], shrink=0.75, location='bottom')

    plt.tight_layout()
    plt.show()
    


def plot_experiment(x,
                    y,
                    legend,
                    x_label='kernel radius (px)',
                    y_label='Average execution time (s)',
                    title='Execution time / Kernel radius'):
    ''' visualize execution time 
    : param x : list of parameters
    : param y : list measurements
    : param legend : names of measurements
    '''
    fig, ax = plt.subplots(figsize=(8,6))
    for sample in y:
        plt.plot(x, sample)
        plt.scatter(x, sample)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(legend)
    plt.grid()
    plt.show()
    
    
def plot_batch(batch, labels, n_rows=1):
    '''
    plots several samples fro mthe batch
    : param : image data, batch of N samples
    : label : np array of size N
    : n_rows : number of ploted rows of lenght 4
    '''
    fig, axes = plt.subplots(figsize=(15,4*n_rows),
                         nrows=n_rows,
                         ncols=4)
    ax = axes.ravel()
    
    for i in range(n_rows*4):
        img = batch[i, :, :]
        lbl = labels[i]
        ax[i].imshow(img, 'gray', interpolation="none")
        ax[i].set_title(f'label: {lbl}\nintensity range:  {np.min(img)} - {np.max(img)}', fontsize=12)

    plt.tight_layout()
    plt.show()
    

def compare_images(img, ref):
    ' compares intensities of two images at pixel level '
    diff = np.abs(img-ref)
    if np.max(diff) < np.e ** -10:
        print('COMPARE: Images are the same.')
    else:
        print(f'COMPARE: Images differ. ')
        plt.imshow(diff, interpolation='none')
        plt.colorbar()
        plt.show()

