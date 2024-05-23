import torch
import torch.nn as nn
from erfpan import ERF_PAN
from erfnet import ERF_Backbone
from erfhead import ERF_Head
from matplotlib import pyplot as plt


class YOLO_ERF(nn.Module):
    def __init__(self, init_biases=False):
        super().__init__()
        self.backbone = ERF_Backbone()
        self.neck = ERF_PAN()
        self.head = ERF_Head(128*3, init_biases=init_biases)

    def forward(self, x, targets=None):
        c3, c4, c5 = self.backbone(x)
        out = self.neck([c3, c4, c5])
        out = self.head(out)
        return out


##model = YOLO_ERF(640)
##model.eval()
##model.cuda()
##
##x = torch.rand((1, 3, 640, 640), dtype=torch.float32).cuda()
##
###torch.onnx.export(model, x, "yolo-erf.onnx")
##
##print("Input shape:", x.shape)
##print()
##    
##for i in range(1):
##    out = model(x)
##    print("Out shape:", out.shape)
##    
##print()
##
##pytorch_total_params = sum(p.numel() for p in model.parameters())
##print("Total parameters:", pytorch_total_params)


from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    model = YOLO_ERF()
    macs, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
