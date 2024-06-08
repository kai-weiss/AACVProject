import torch.nn as nn
from yolo_model import YOLOv8
from CBAM import CBAM

class YOLOv8_CBAM(YOLOv8):
    def __init__(self, cbam_channels):
        super(YOLOv8_CBAM, self).__init__()
        self.cbam1 = CBAM(cbam_channels[0])
        self.cbam2 = CBAM(cbam_channels[1])

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam1(x)
        x = self.neck(x)
        x = self.cbam2(x)
        x = self.head(x)
        return x

# Example usage
# model = YOLOv8_CBAM(cbam_channels=[128, 256])
