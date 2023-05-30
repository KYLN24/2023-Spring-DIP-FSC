import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import MultiheadAttention
from torchvision.models import ResNet18_Weights, resnet18

from utils import DownMSELoss


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super(ImageEncoder, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(CrossAttentionBlock, self).__init__()
        self.attention = MultiheadAttention(
            embed_dim, num_heads, dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.fc_1 = nn.Linear(embed_dim, 1024)
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(1024, embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x, y):
        x = self.ln_1(self.attention(x, y, y)[0] + x)
        x = self.ln_2(self.fc_2(self.dropout(F.gelu(self.fc_1(x)))) + x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, n_blocks=3):
        super(CrossAttention, self).__init__()
        self.cross_attention_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x, y):
        for block in self.cross_attention_blocks:
            x = block(x, y)
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, dropout=0.5):
        super(SelfAttentionBlock, self).__init__()
        self.attention = MultiheadAttention(
            embed_dim, num_heads, dropout, batch_first=True
        )
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.fc_1 = nn.Linear(embed_dim, 1024)
        self.dropout = nn.Dropout(dropout)
        self.fc_2 = nn.Linear(1024, embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.ln_1(self.attention(x, x, x)[0] + x)
        x = self.ln_2(self.fc_2(self.dropout(F.gelu(self.fc_1(x)))) + x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, n_blocks=3):
        super(SelfAttention, self).__init__()
        self.self_attention = nn.Sequential(
            *[
                SelfAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x):
        return self.self_attention(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
        )
        self.shortcut = (
            nn.Conv2d(in_channel, out_channel, 1)
            if in_channel != out_channel
            else nn.Identity()
        )

    def forward(self, x):
        return self.net(x) + self.shortcut(x)


class ExEncoder(nn.Module):
    def __init__(self) -> None:
        super(ExEncoder, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder.avgpool = nn.Identity()
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], 512, -1)
        return x


class DMapDecoder(nn.Module):
    def __init__(self, in_channel):
        super(DMapDecoder, self).__init__()
        self.in_channel = in_channel
        self.upsample = nn.Sequential(
            ResBlock(self.in_channel, 512),
            ResBlock(512, 256),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResBlock(256, 256),
            ResBlock(256, 128),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ResBlock(128, 128),
            ResBlock(128, 64),
            ResBlock(64, 64),
            ResBlock(64, 1),
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class LitModule(pl.LightningModule):
    def __init__(
        self,
        optimizer="adamw",
        scheduler="",
        learning_rate=1e-3,
        train_batch_size=2,
        val_batch_size=2,
        dmap_scale=1,
        dmap_resize=8,
    ):
        super(LitModule, self).__init__()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = learning_rate
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.dmap_scale = dmap_scale
        self.dmap_resize = dmap_resize

        self.criterion = DownMSELoss(dmap_resize)
        self.save_hyperparameters()

        self.image_encoder = ImageEncoder()
        self.self_attention_1 = SelfAttention(512, 4, 0.5, n_blocks=2)

        self.ex_encoder = ExEncoder()
        self.ex_cross_attention = CrossAttention(512, 4, 0.5, n_blocks=2)

        self.self_attention_2 = SelfAttention(512, 4, 0.5, n_blocks=2)
        self.dmap_decoder = DMapDecoder(512)

    def forward(self, raw_img, examplar_1, examplar_2, examplar_3, cls):
        code = self.image_encoder(raw_img)

        hc = code.shape[2]
        wc = code.shape[3]

        code = code.view(code.shape[0], code.shape[1], -1).permute((0, 2, 1))
        code = code + self.self_attention_1(code)

        ex_code_1 = self.ex_encoder(examplar_1)
        ex_code_2 = self.ex_encoder(examplar_2)
        ex_code_3 = self.ex_encoder(examplar_3)
        ex_code = torch.concat([ex_code_1, ex_code_2, ex_code_3], dim=2)
        ex_code = torch.permute(ex_code, (0, 2, 1))
        code = code + self.ex_cross_attention(code, ex_code)
        code = code + self.self_attention_2(code)

        code = code.permute((0, 2, 1))
        code = code.view(code.shape[0], code.shape[1], hc, wc)
        dmap = self.dmap_decoder(code)
        return dmap

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
            return {"optimizer": optimizer}
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4
            )
        else:
            raise NotImplementedError
        if self.scheduler:
            if self.scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=250, eta_min=1e-8
                )
            elif self.scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=50, gamma=0.5
                )
            else:
                raise NotImplementedError
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        raw_img, dmap, examplar_1, examplar_2, examplar_3, cls = batch

        pred = self(raw_img, examplar_1, examplar_2, examplar_3, cls)
        loss = self.criterion(pred, dmap * self.dmap_scale)

        gt_count = torch.sum(dmap, dim=[1, 2, 3])
        pred_count = torch.sum(pred / self.dmap_scale, dim=[1, 2, 3])

        mae = torch.mean(torch.abs(pred_count - gt_count))

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.train_batch_size,
            sync_dist=True,
        )

        self.log(
            "train/mae",
            mae,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.train_batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        raw_img, dmap, examplar_1, examplar_2, examplar_3, cls = batch
        pred = self(raw_img, examplar_1, examplar_2, examplar_3, cls)
        gt_count = torch.sum(dmap, dim=[1, 2, 3])
        pred_count = torch.sum(pred / self.dmap_scale, dim=[1, 2, 3])
        mae = torch.mean(torch.abs(pred_count - gt_count))

        self.log(
            "val/mae",
            mae,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.val_batch_size,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        raw_img, dmap, examplar_1, examplar_2, examplar_3, cls = batch
        pred = self(raw_img, examplar_1, examplar_2, examplar_3, cls)
        gt_count = torch.sum(dmap, dim=[1, 2, 3])
        pred_count = torch.sum(pred / self.dmap_scale, dim=[1, 2, 3])
        mae = torch.mean(torch.abs(pred_count - gt_count))

        self.log(
            "test/mae",
            mae,
            on_step=True,
            prog_bar=True,
            batch_size=self.val_batch_size,
            sync_dist=True,
        )
