from tqdm import tqdm

import torch

import config as CFG
from dataset import CLIPDatasetSixFeatures, transforms
from torch.utils.data import DataLoader
from CLIP import CLIPModel
from utils import AvgMeter, get_lr, seed_everything
# initalize tensorboard summary writer
from torch.utils.tensorboard import SummaryWriter



def train_epoch(model, train_loader, optimizer, lr_scheduler, step, summary_writer, epoch):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    counter = 0
    for batch in tqdm_object:
        image, features = batch
        batch ={'image': image.to(CFG.device), 'text_features': features.to(CFG.device).float()}
        #batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        summary_writer.add_scalar('train_loss_sample', loss_meter.avg, epoch*len(train_loader) + counter)
        counter += 1
    summary_writer.add_scalar('train_loss_epoch', loss_meter.avg, epoch)
    return loss_meter


def valid_epoch(model, valid_loader,  summary_writer, epoch):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    counter = 0
    for batch in tqdm_object:
        image, features = batch
        batch ={'image': image.to(CFG.device), 'text_features': features.to(CFG.device).float()}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        summary_writer.add_scalar('valid_loss_sample', loss_meter.avg, epoch*len(valid_loader) + counter)
        counter += 1
    summary_writer.add_scalar('valid_loss_epoch', loss_meter.avg, epoch)
    return loss_meter


def main():
    batch_size = CFG.batch_size
    dataset = CLIPDatasetSixFeatures("malignancy_2.csv", "C:\\Users\\fu057938\\GeoCLIP_Project\\Nodules\\Nodules", transform=transforms)  # replace with your file paths
    seed_everything(CFG.seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size//2)

    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"
    summary_writer = SummaryWriter('tb_logs')

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, summary_writer, epoch)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, summary_writer, epoch)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), CFG.model_path)
            print("Saved Best Model!")


if __name__ == "__main__":
    main()