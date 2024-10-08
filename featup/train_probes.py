from os.path import join

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, JaccardIndex

from PIL import Image
import torchvision.transforms.v2 as T

from featup.datasets.COCO import Coco
from featup.datasets.EmbeddingFile import EmbeddingFile, EmbeddingAndImage
from featup.losses import ScaleAndShiftInvariantLoss
from featup.util import pca, remove_axes, norm


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


class LitPrototypeEvaluator(pl.LightningModule):
    def __init__(self, task, n_dim):
        super().__init__()
        self.task = task
        self.n_dim = n_dim

        if self.task == 'seg':
            n_classes = 27
        elif self.task == 'depth':
            n_classes = 1

            self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').cuda()
            self.midas.eval()
            self.midas_loss = ScaleAndShiftInvariantLoss()

            self.mse = 0
            self.ssil = 0
            self.steps = 0

        self.prototypes_buff = self.register_buffer("prototypes", torch.zeros(n_classes, n_dim))
        self.classifier = torch.nn.Conv2d(n_dim, n_classes, 1)

        self.prot_acc_metric = Accuracy(num_classes=n_classes, task="multiclass")
        self.prot_acc_buff = self.register_buffer("prot_acc", torch.tensor(0.0))
        self.prot_iou_metric = JaccardIndex(num_classes=n_classes, task="multiclass")
        self.prot_iou_buff = self.register_buffer("prot_iou", torch.tensor(0.0))

        self.linear_acc_metric = Accuracy(num_classes=n_classes, task="multiclass")
        self.linear_acc_buff = self.register_buffer("linear_acc", torch.tensor(0.0))
        self.linear_iou_metric = JaccardIndex(num_classes=n_classes, task="multiclass")
        self.linear_iou_buff = self.register_buffer("linear_iou", torch.tensor(0.0))

        self.n_classes = n_classes  
        self.ce = torch.nn.CrossEntropyLoss()

        # set attribute like this to avoid buffer registration
        object.__setattr__(self, 'featup', torch.hub.load('mhamilton723/FeatUp', 'dinov2', use_norm=False))

    def get_prototypes(self, feats):
        b, c, h, w = feats.shape
        k = self.prototypes.shape[0]
        matches = torch.einsum("kc,bchw->kbhw", F.normalize(self.prototypes, dim=1), F.normalize(feats, dim=1)) \
            .reshape(k, -1).argmax(0)
        return self.prototypes[matches].reshape(b, h, w, c).permute(0, 3, 1, 2)

    def training_step(self, batch, batch_idx):
        feats, label = batch
        b, c, h, w = feats.shape

        small_labels = F.interpolate(
            label.unsqueeze(1).to(torch.float32),
            size=(feats.shape[2], feats.shape[3])).to(torch.int64)

        linear_preds = self.classifier(feats)

        if self.task == 'seg':
            flat_labels = small_labels.permute(0, 2, 3, 1).reshape(b * h * w)
            flat_linear_preds = linear_preds.permute(0, 2, 3, 1).reshape(b * h * w, -1)

            selected = flat_labels > -1
            linear_loss = self.ce(
                flat_linear_preds[selected],
                flat_labels[selected])
            loss = linear_loss
            self.log("linear_loss", linear_loss)
            self.log("loss", loss)

            for l in range(self.n_classes):
                self.prototypes[l] += feats.permute(0, 2, 3, 1).reshape(b * h * w, -1)[flat_labels == l].sum(dim=0)

            if self.global_step % 10 == 1 and self.trainer.is_global_zero:
                with torch.no_grad():
                    prots = self.get_prototypes(feats)
                    prot_loss = -(F.normalize(feats, dim=1) * F.normalize(prots, dim=1)).sum(1).mean()
                self.logger.experiment.add_scalar("prot_loss", prot_loss, self.global_step)

        elif self.task == 'depth':
            loss = self.midas_loss(linear_preds.squeeze(), small_labels.squeeze(),
                                   torch.ones_like(linear_preds.squeeze()))
            self.log('loss', loss)

        if self.global_step % 200 == 0 and self.trainer.is_global_zero:
            n_images = 5
            fig, axes = plt.subplots(4, n_images, figsize=(4 * n_images, 5 * 5))

            prot_preds = torch.einsum("bchw,kc->bkhw",
                                      F.normalize(feats, dim=1),
                                      F.normalize(self.prototypes, dim=1, eps=1e-10))

            colorize = Coco.colorize_label if self.task == 'seg' else lambda x: x.detach().cpu()
            for i in range(n_images):
                feats_pca = pca([feats])[0][0][i]
                axes[0, i].imshow(feats_pca.permute(1,2,0).cpu())
                axes[1, i].imshow(colorize(label[i]))
                if self.task == 'depth':
                    axes[2, i].imshow(colorize(linear_preds[i][0]))
                    axes[3, i].imshow(colorize(prot_preds[i][0]))
                elif self.task == 'seg':
                    axes[2, i].imshow(colorize(linear_preds.argmax(1)[i]))
                    axes[3, i].imshow(colorize(prot_preds.argmax(1)[i]))

            plt.tight_layout()
            remove_axes(axes)
            self.logger.experiment.add_figure('predictions', fig, self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            feats, label, img = batch

            if self.task == 'seg':
                label = F.interpolate(
                    label.to(torch.float32).unsqueeze(1), size=(224, 224)).to(torch.int64).squeeze(1)
                b, h, w = label.shape

                # we can upsample here if we want, this is commented/uncommented for each run I need to do (sorry)
                if UPSAMPLER_TYPE == "none":
                    pass
                elif UPSAMPLER_TYPE == "bilinear":
                    feats = F.interpolate(feats, (h, w), mode='bilinear')
                elif UPSAMPLER_TYPE == "featup":
                    self.featup = self.featup.to(feats.device)
                    feats = self.featup.upsampler(feats, norm(img))

                # we added the nearest interpolation below, as it wasn't present
                prot_preds = torch.einsum(
                    "bchw,kc->bkhw",
                    F.normalize(feats, dim=1),
                    F.normalize(self.prototypes, dim=1, eps=1e-10)).argmax(1, keepdim=True)
                linear_preds = self.classifier(feats).argmax(1, keepdim=True)

                flat_labels = label.flatten()
                selected = flat_labels > -1
                flat_labels = flat_labels[selected]

                flat_prot_preds = F.interpolate(
                    prot_preds.to(torch.float32), (h, w), mode='nearest').to(torch.int64).flatten()[selected]
                self.prot_acc_metric.update(flat_prot_preds, flat_labels)
                self.prot_iou_metric.update(flat_prot_preds, flat_labels)

                flat_linear_preds = F.interpolate(
                    linear_preds.to(torch.float32), (h, w), mode='nearest').to(torch.int64).flatten()[selected]
                self.linear_acc_metric.update(flat_linear_preds, flat_labels)
                self.linear_iou_metric.update(flat_linear_preds, flat_labels)

            elif self.task == 'depth':
                linear_preds = self.classifier(feats)
                small_labels = F.interpolate(
                    label.unsqueeze(1).to(torch.float32),
                    size=(feats.shape[2], feats.shape[3])).to(torch.int64)
                mse = (small_labels - linear_preds).pow(2).mean()
                midas_l = self.midas_loss(linear_preds.squeeze(), small_labels.squeeze(),
                                          torch.ones_like(linear_preds.squeeze()))
                self.mse += mse.item()
                self.ssil += midas_l.item()

                self.steps += 1

        return None

    def on_validation_epoch_end(self):
        self.prot_acc = self.prot_acc_metric.compute()
        self.prot_iou = self.prot_iou_metric.compute()
        self.linear_acc = self.linear_acc_metric.compute()
        self.linear_iou = self.linear_iou_metric.compute()
        compute_result(self)
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(), lr=5e-3)

def compute_result(evaluator):
    result = {
        "Prototype Accuracy": float(evaluator.prot_acc),
        "Prototype mIoU": float(evaluator.prot_iou),
        "Linear Accuracy": float(evaluator.linear_acc),
        "Linear mIoU": float(evaluator.linear_iou),
    }
    print(result)
    return result

validate = True
UPSAMPLER_TYPE = "featup"

@hydra.main(config_path="configs", config_name="train_probe.yaml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.output_root)
    seed_everything(seed=0, workers=True)

    log_dir = f"runs/probes/{cfg.task}-probe"
    chkpt_dir = f"runs/probes/{cfg.task}-probe-{cfg.model_type}.ckpt"

    # emb_root = join(cfg.pytorch_data_dir, "cocostuff", "embedding", cfg.model_type)
    emb_root = "/export/home/data/featupdata/featup_embeddings"

    if validate and (not UPSAMPLER_TYPE in ['none', 'bilinear']):  # reduce batch size to avoid memory overflow
        cfg.batch_size = 8


    # resize to 224 transform
    img_transform = torch.nn.Sequential(
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    )
    label_transform = torch.nn.Sequential(T.ToImage(), T.ToDtype(torch.int), T.Resize((16, 16), interpolation=Image.NEAREST))
    coco_val_ds = Coco('/export/home/data/featupdata/', 'val', transform=img_transform, target_transform=label_transform, include_labels=True)

    train_dataset = EmbeddingFile(join(emb_root, "embeddings.npz"))
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    val_dataset = EmbeddingAndImage(join(emb_root, f"embeddings_val.npz"), coco_val_ds)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    evaluator = LitPrototypeEvaluator(cfg.task, train_dataset.dim())
    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False)

    trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=cfg.epochs,
        logger=tb_logger,
        log_every_n_steps=100,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=10,
    )

    if validate:
        ckpt_path = "/home/fmarchesoni/segmentation_features/evaluations/FeatUp2/epoch=49-step=19100.ckpt"
        trainer.validate(evaluator, val_loader, ckpt_path=ckpt_path)
    else:
        trainer.fit(evaluator, train_loader, val_loader)

        trainer.save_checkpoint(chkpt_dir)
    results = compute_result(evaluator)


if __name__ == "__main__":
    my_app()
