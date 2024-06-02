import argparse  # Library for parsing command-line arguments
import logging  # Library for logging messages
import os  # Library for interacting with the operating system
import random  # Library for generating random numbers
import sys  # Library for interacting with the system
import torch  # Library for deep learning
print(f'Torch version is {torch.__version__}')
import torch.nn as nn  # Module for creating neural networks
import torch.nn.functional as F  # Module for functional operations in PyTorch
import torchvision.transforms as transforms  # Module for image transformations
import torchvision.transforms.functional as TF  # Module for functional image transformations
from pathlib import Path  # Class for working with file paths
from torch import optim  # Module for optimization algorithms
from torch.utils.data import DataLoader, random_split  # Classes for loading and splitting data
from tqdm import tqdm  # Library for creating progress bars

# import wandb  # Library for experiment tracking and visualization
from evaluate import evaluate  # Custom module for model evaluation
from unet import UNet  # Custom module for the U-Net model
from utils.data_loading import BasicDataset, CarvanaDataset  # Custom modules for loading datasets
from utils.dice_score import dice_loss  # Custom module for calculating dice loss

# Define the paths to the image, mask, and checkpoint directories
dir_img = Path(os.environ['SM_CHANNEL_TRAINING']) / 'imgs'
dir_mask = Path(os.environ['SM_CHANNEL_TRAINING']) / 'masks'
dir_checkpoint = Path(os.environ['SM_CHANNEL_TRAINING']) / 'checkpoints'


def train_model(
    model,
    device,
    epochs: int = 1,
    batch_size: int = 10,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)  # Create a dataset using CarvanaDataset class
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)  # Create a dataset using BasicDataset class

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)  # Calculate the number of validation samples
    n_train = len(dataset) - n_val  # Calculate the number of training samples
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )  # Split the dataset into training and validation sets

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )  # Arguments for the data loaders
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)  # Create a data loader for training set
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)  # Create a data loader for validation set

    # (Initialize logging)
    # experiment = wandb.init(project="U-Net", resume="allow", anonymous="must")  # Initialize the experiment for logging
    # experiment.config.update(
    #     dict(
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         learning_rate=learning_rate,
    #         val_percent=val_percent,
    #         save_checkpoint=save_checkpoint,
    #         img_scale=img_scale,
    #         amp=amp,
    #     )
    # )  # Update the experiment configuration

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    """
    )  # Log the training configuration

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )  # Create an optimizer for training the model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=5
    )  # Create a learning rate scheduler
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)  # Create a gradient scaler for mixed precision training
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()  # Create a loss function
    global_step = 0  # Initialize the global step counter

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()  # Set the model to training mode
        epoch_loss = 0  # Initialize the epoch loss
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]  # Get the images and masks from the batch

                assert images.shape[1] == model.n_channels, (
                    f"Network has been defined with {model.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )  # Check if the number of input channels matches the network configuration

                images = images.to(
                    device=device,
                    dtype=torch.float32,
                    memory_format=torch.channels_last,
                )  # Move the images to the device
                true_masks = true_masks.to(device=device, dtype=torch.long)  # Move the masks to the device

                with torch.autocast(
                    device.type if device.type != "mps" else "cpu", enabled=amp
                ):  # Enable automatic mixed precision (AMP) for faster and memory efficient training
                    masks_pred = model(images)  # Forward pass: get the predicted masks from the model
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())  # Calculate the loss for binary classification
                        loss += dice_loss(
                            F.sigmoid(masks_pred.squeeze(1)),
                            true_masks.float(),
                            multiclass=False,
                        )  # Calculate the dice loss for binary classification
                    else:
                        loss = criterion(masks_pred, true_masks)  # Calculate the loss for multi-class classification
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes)
                            .permute(0, 3, 1, 2)
                            .float(),
                            multiclass=True,
                        )  # Calculate the dice loss for multi-class classification

                optimizer.zero_grad(set_to_none=True)  # Zero the gradients
                grad_scaler.scale(loss).backward()  # Backward pass: compute gradients
                grad_scaler.unscale_(optimizer)  # Unscales the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # Clip the gradients to prevent exploding gradients
                grad_scaler.step(optimizer)  # Update the model parameters
                grad_scaler.update()  # Update the gradient scaler for AMP

                pbar.update(images.shape[0])  # Update the progress bar
                global_step += 1  # Increment the global step counter
                epoch_loss += loss.item()  # Accumulate the loss for the epoch
                # experiment.log(
                #     {"train loss": loss.item(), "step": global_step, "epoch": epoch}
                # )  # Log the training loss
                pbar.set_postfix(**{"loss (batch)": loss.item()})  # Update the progress bar with the current loss

                # Evaluation round
                division_step = n_train // (5 * batch_size)  # Calculate the division step for evaluation
                if division_step > 0:
                    # if global_step % division_step == 0:
                    #     histograms = {}
                    #     for tag, value in model.named_parameters():
                    #         tag = tag.replace("/", ".")
                    #         if not (torch.isinf(value) | torch.isnan(value)).any():
                    #             histograms["Weights/" + tag] = wandb.Histogram(
                    #                 value.data.cpu()
                    #             )
                    #         if not (
                    #             torch.isinf(value.grad) | torch.isnan(value.grad)
                    #         ).any():
                    #             histograms["Gradients/" + tag] = wandb.Histogram(
                    #                 value.grad.data.cpu()
                    #             )

                        val_score = evaluate(model, val_loader, device, amp)  # Evaluate the model on the validation set
                        scheduler.step(val_score)  # Adjust the learning rate based on the validation score

                        logging.info("Validation Dice score: {}".format(val_score))  # Log the validation score
                        # try:
                        #     experiment.log(
                        #         {
                        #             "learning rate": optimizer.param_groups[0]["lr"],
                        #             "validation Dice": val_score,
                        #             "images": wandb.Image(images[0].cpu()),
                        #             "masks": {
                        #                 "true": wandb.Image(
                        #                     true_masks[0].float().cpu()
                        #                 ),
                        #                 "pred": wandb.Image(
                        #                     masks_pred.argmax(dim=1)[0].float().cpu()
                        #                 ),
                        #             },
                        #             "step": global_step,
                        #             "epoch": epoch,
                        #             **histograms,
                        #         }
                        #     )  # Log the images, masks, and histograms
                        # except:
                        #     pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)  # Create the checkpoint directory if it doesn't exist
            state_dict = model.state_dict()  # Get the model state dictionary
            state_dict["mask_values"] = dataset.mask_values  # Add the mask values to the state dictionary
            torch.save(
                state_dict, str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch))
            )  # Save the model checkpoint
            logging.info(f"Checkpoint {epoch} saved!")  # Log the checkpoint saving


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )  # Create an argument parser
    parser.add_argument(
        "--epochs", "-e", metavar="E", type=int, default=5, help="Number of epochs"
    )  # Add an argument for the number of epochs
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        metavar="B",
        type=int,
        default=1,
        help="Batch size",
    )  # Add an argument for the batch size
    parser.add_argument(
        "--learning-rate",
        "-l",
        metavar="LR",
        type=float,
        default=1e-5,
        help="Learning rate",
        dest="lr",
    )  # Add an argument for the learning rate
    parser.add_argument(
        "--load", "-f", type=str, default=False, help="Load model from a .pth file"
    )  # Add an argument for loading a pre-trained model
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=0.5,
        help="Downscaling factor of the images",
    )  # Add an argument for the image scaling factor
    parser.add_argument(
        "--validation",
        "-v",
        dest="val",
        type=float,
        default=10.0,
        help="Percent of the data that is used as validation (0-100)",
    )  # Add an argument for the validation percentage
    parser.add_argument(
        "--amp", action="store_true", default=False, help="Use mixed precision"
    )  # Add an argument for using mixed precision training
    parser.add_argument(
        "--bilinear", action="store_true", default=False, help="Use bilinear upsampling"
    )  # Add an argument for using bilinear upsampling
    parser.add_argument(
        "--classes", "-c", type=int, default=2, help="Number of classes"
    )  # Add an argument for the number of classes

    return parser.parse_args()  # Parse the command-line arguments


if __name__ == "__main__":
    args = get_args()  # Get the command-line arguments

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")  # Configure the logging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the device for training
    logging.info(f"Using device {device}")  # Log the device being used

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)  # Create an instance of the U-Net model
    model = model.to(memory_format=torch.channels_last)  # Move the model to the device

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )  # Log the network configuration

    if args.load:
        state_dict = torch.load(args.load, map_location=device)  # Load the pre-trained model state dictionary
        del state_dict["mask_values"]  # Remove the mask values from the state dictionary
        model.load_state_dict(state_dict)  # Load the model state dictionary
        logging.info(f"Model loaded from {args.load}")  # Log the model loading

    model.to(device=device)  # Move the model to the device
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )  # Train the model
    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! "
            "Enabling checkpointing to reduce memory usage, but this slows down training. "
            "Consider enabling AMP (--amp) for fast and memory efficient training"
        )  # Log the OutOfMemoryError
        torch.cuda.empty_cache()  # Empty the GPU cache
        model.use_checkpointing()  # Enable checkpointing to reduce memory usage
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
        )  # Train the model with checkpointing enabled
