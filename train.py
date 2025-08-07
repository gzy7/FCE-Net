import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure
from model import FCE
from losses import CombinedLoss
from dataloader import create_dataloaders

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            ssim = calculate_ssim(output, high)
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim

def main():
    train_low = ''
    train_high = ''
    test_low = ''
    test_high = ''
    learning_rate = 2e-4
    num_epochs = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256,
                                                   batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    model = FCE().to(device)

    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_psnr = 0
    print('Training started.')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim = validate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Saving model with PSNR: {best_psnr:.6f}')

if __name__ == '__main__':
    main()