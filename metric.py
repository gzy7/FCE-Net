import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
import lpips

def load_image(path, target_size=(600, 400)):
    image = Image.open(path).convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(dim=1)
        img2_gray = img2.mean(dim=1)
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
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def calculate_lpips(img1, img2, lpips_model):
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1
    lpips_val = lpips_model(img1, img2)
    return lpips_val.item()

def evaluate_folder(pred_dir, gt_dir):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='alex').to(device)

    total_psnr, total_ssim, total_lpips = 0, 0, 0
    num = len(pred_files)

    for i in range(num):
        pred_path = os.path.join(pred_dir, pred_files[i])
        gt_path = os.path.join(gt_dir, gt_files[i])

        pred_img = load_image(pred_path).to(device)
        gt_img = load_image(gt_path).to(device)

        psnr = calculate_psnr(pred_img, gt_img)
        ssim = calculate_ssim(pred_img, gt_img)
        lpips_score = calculate_lpips(pred_img, gt_img, lpips_model)

        total_psnr += psnr
        total_ssim += ssim
        total_lpips += lpips_score

        print(f"[{i + 1}/{num}] {pred_files[i]} - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips_score:.4f}")

    avg_psnr = total_psnr / num
    avg_ssim = total_ssim / num
    avg_lpips = total_lpips / num
    print(f"\nAverage PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}, Average LPIPS: {avg_lpips:.4f}")

if __name__ == '__main__':
    pred_dir = ''
    gt_dir = ''
    evaluate_folder(pred_dir, gt_dir)