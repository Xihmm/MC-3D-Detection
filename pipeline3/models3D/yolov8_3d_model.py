import torch
import torch.nn as nn


class Conv3D(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv3d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck3D(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv3D(c, c, 3, 1)
        self.cv2 = Conv3D(c, c, 3, 1)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f3D(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        hidden = c2 // 2
        self.cv1 = Conv3D(c1, hidden * 2, 1, 1)
        self.blocks = nn.ModuleList([Bottleneck3D(hidden, shortcut) for _ in range(n)])
        self.cv2 = Conv3D(hidden * (2 + n), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF3D(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        hidden = c1 // 2
        self.cv1 = Conv3D(c1, hidden, 1, 1)
        self.pool = nn.MaxPool3d(k, stride=1, padding=k // 2)
        self.cv2 = Conv3D(hidden * 4, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DecoupledHead3D(nn.Module):
    """
    YOLOv8-style 3D anchor-free decoupled head.

    Each scale predicts:
      box: [B, 6, Dz, Dy, Dx]
      obj: [B, 1, Dz, Dy, Dx]
      cls: [B, nc, Dz, Dy, Dx]
    """
    def __init__(self, channels, nc=1):
        super().__init__()
        self.nc = nc

        self.box_heads = nn.ModuleList()
        self.obj_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()

        for c in channels:
            self.box_heads.append(
                nn.Sequential(
                    Conv3D(c, c, 3, 1),
                    Conv3D(c, c, 3, 1),
                    nn.Conv3d(c, 6, 1)
                )
            )

            self.obj_heads.append(
                nn.Sequential(
                    Conv3D(c, c, 3, 1),
                    Conv3D(c, c, 3, 1),
                    nn.Conv3d(c, 1, 1)
                )
            )

            self.cls_heads.append(
                nn.Sequential(
                    Conv3D(c, c, 3, 1),
                    Conv3D(c, c, 3, 1),
                    nn.Conv3d(c, nc, 1)
                )
            )

    def forward(self, feats):
        outputs = []
        for i, f in enumerate(feats):
            outputs.append({
                "box": self.box_heads[i](f),
                "obj": self.obj_heads[i](f),
                "cls": self.cls_heads[i](f),
            })
        return outputs


class YOLOv8_3D(nn.Module):
    def __init__(self, in_channels=1, nc=1):
        super().__init__()

        self.stem = Conv3D(in_channels, 32, 3, 1)

        self.stage1 = nn.Sequential(
            Conv3D(32, 64, 3, 2),
            C2f3D(64, 64, n=1)
        )

        self.stage2 = nn.Sequential(
            Conv3D(64, 128, 3, 2),
            C2f3D(128, 128, n=2)
        )

        self.stage3 = nn.Sequential(
            Conv3D(128, 256, 3, 2),
            C2f3D(256, 256, n=2),
            SPPF3D(256, 256)
        )

        self.head = DecoupledHead3D(
            channels=[64, 128, 256],
            nc=nc
        )

    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        return self.head([p2, p3, p4])


def build_yolov8_3d(nc=1):
    return YOLOv8_3D(in_channels=1, nc=nc)


if __name__ == "__main__":
    model = build_yolov8_3d(nc=1)
    x = torch.zeros(1, 1, 33, 320, 320)

    with torch.no_grad():
        outputs = model(x)

    for i, out in enumerate(outputs):
        print(f"Scale {i}")
        print("box:", out["box"].shape)
        print("cls:", out["cls"].shape)