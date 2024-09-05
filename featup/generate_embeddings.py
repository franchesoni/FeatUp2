from pathlib import Path
import numpy as np
import torch
import tqdm
from PIL import Image
import torchvision.transforms.v2 as T

from datasets.COCO import Coco

# identity_fn = lambda x: x
minmaxnorm = lambda x: (x - x.min()) / (x.max() - x.min())

def generate_embeddings(dstdir, do_val=False):
    dstdir = Path(dstdir)
    dstdir.mkdir(exist_ok=True, parents=True)

    # load dinov2 model
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    # resize to 224 transform
    img_transform = torch.nn.Sequential(
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    )
    # label_transform = torch.nn.Sequential(T.ToImage(), T.ToDtype(torch.int), T.Resize((224, 224), interpolation=Image.NEAREST))
    label_transform = torch.nn.Sequential(T.ToImage(), T.ToDtype(torch.int), T.Resize((16, 16), interpolation=Image.NEAREST))
    if do_val:
        ds = Coco('/export/home/data/featupdata/', 'val', transform=img_transform, target_transform=label_transform, include_labels=True)
    else:
        ds = Coco('/export/home/data/featupdata/', 'train', transform=img_transform, target_transform=label_transform, include_labels=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=32, shuffle=False)
    # ds2 = Coco('/export/home/data/featupdata/', 'train', transform=img_transform, target_transform=label_transform2, include_labels=True)
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    i = 0
    results = {"feats": np.empty((len(ds), 384, 16, 16), dtype=np.float32), "labels": np.empty((len(ds), 16, 16), dtype=np.int32)}
    
    for batch_ind, batch in tqdm.tqdm(enumerate(dl)):
        img, imgpath, label = batch['img'], batch['img_path'], batch['label']
        # Image.fromarray(((label+1)/label.max()*255).to(torch.uint8).numpy()).save('label.png')
        # Image.fromarray(((dslabel+1)/dslabel.max()*255).to(torch.uint8).numpy()).save('labelds.png')
        # # img.save('img.png')
        # Image.fromarray( (img.permute(1, 2, 0).numpy() * 255).astype('uint8')).save('img.png')
        img = (img - mean) / std
        with torch.no_grad():
            feats = dino.forward_features(img)["x_norm_patchtokens"]
        P = 14
        B, C, H, W = img.shape
        Ph, Pw = H // P, W // P
        B, PhPw, F = feats.shape
        feats = feats.reshape(B, Ph, Pw, F).permute(0, 3, 1, 2).cpu().numpy()
        results["feats"][i:i+B] = feats
        results["labels"][i:i+B] = label.numpy()

        # Image.fromarray((minmaxnorm(feats[:3])*255).astype('uint8').transpose(1, 2, 0)).save('ifeats.png')
        # i += 1
        # breakpoint()
    print('saving final embeddings...')
    np.savez(dstdir / f"embeddings{'_val' if do_val else ''}.npz", **results)
    print('done')



if __name__ == '__main__':
    generate_embeddings('/export/home/data/featupdata/featup_embeddings', do_val=True)