import os
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import DriveDataset
from loss import DiceLoss, DiceBCELoss, FocalLoss, BCELoss
from utils import seeding, create_dir, epoch_time
from tensorboardX import SummaryWriter
import albumentations as A
from models.mpresnet import MPResNet


NET_NAME = 'MPResNet34_adam_bce'
DATA_NAME = 'Germany_Africa_field'
ROOT_DIR = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc"
working_path = os.path.dirname(os.path.abspath(__file__))

args = {
    "batch_size": 16,
    "lr": 1e-4,
    "device": 2,
    "epochs": 200,
    "classes": 1,
    "channels": 3,
    "weight": os.path.join(working_path, "weight", DATA_NAME, NET_NAME),
    "logs": os.path.join(working_path, "logs", DATA_NAME, NET_NAME)
}

writer = SummaryWriter(args['logs'])

def main():
    seeding(42)
    create_dir(args["weight"])
    create_dir(args["logs"])
    
    train_x = sorted(glob(ROOT_DIR + "/train/images/*"))
    train_y = sorted(glob(ROOT_DIR + "/train/masks/*"))

    valid_x = sorted(glob(ROOT_DIR + "/val/images/*"))
    valid_y = sorted(glob(ROOT_DIR + "/val/masks/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    transform = A.Compose([A.HorizontalFlip(p=0.5),
                                     A.VerticalFlip(p=0.5),
                                     A.Rotate(p=0.5),
                                     A.Rotate(p=0.5, limit=15),
                                     A.OneOf([A.RandomSizedCrop(min_max_height=(128, 256), height=256, width=256, p=0.5),
                                              A.PadIfNeeded(min_height=256, min_width=256, p=0.5)], p=1)])

    train_dataset = DriveDataset(train_x, train_y, transform=transform)
    valid_dataset = DriveDataset(valid_x, valid_y, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=4
    )

    device = torch.cuda.set_device(args["device"])
    device = torch.device(f"cuda:{args['device']}")
    
    model = MPResNet(args["channels"], num_classes=args["classes"])
    model = model.to(device)    

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    loss_fn = BCELoss()

    best_valid_loss = float("inf")

    for epoch in range(args["epochs"]+1):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device, epoch)
        valid_loss = evaluate(model, valid_loader, loss_fn, device, epoch)

        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.5f} to {valid_loss:2.5f}. Saving checkpoint."
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(args["weight"], NET_NAME + "_best.pth"))

        if epoch % 50 == 0 and epoch != 0:
            print(f"Saving epochs: {epoch}")
            torch.save(model.state_dict(), os.path.join(args["weight"], NET_NAME + f"_{epoch}.pth"))

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.5f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.5f}\n'
        print(data_str)
    
    writer.close()
    print('Training finished')


def train(model, loader, optimizer, loss_fn, device, curr_ep):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred, aux = model(x)
        loss_y_pred = loss_fn(y_pred, y)
        loss_aux = loss_fn(aux, y)
        loss = (loss_y_pred * 0.7) + (loss_aux * 0.3)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    writer.add_scalar('Train_loss', epoch_loss, curr_ep)

    return epoch_loss


def evaluate(model, loader, loss_fn, device, curr_ep):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred, _ = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
        writer.add_scalar('Valid_loss', epoch_loss, curr_ep)

    return epoch_loss


if __name__ == "__main__":
    main()
