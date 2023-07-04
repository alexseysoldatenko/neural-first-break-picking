import torch


class Similar_Unet(torch.nn.Module):
    """
        Base implemintation of Unet for segmentation. Input.shape == output.shape
        params:
            in_channel - number of input image channels
            out_classes - number of output classes

    
    """
    def __init__(self, in_channel: int, out_classes: int):
        super().__init__()
        self.first_downsampling_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, 64, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d((2,2))
        )

        self.second_downsampling_block = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d((2,2))
        )

        self.third_downsampling_block = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(256, 256, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d((2,2))
        )

        self.fourth_downsampling_block = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(512, 512, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.MaxPool2d((2,2))
        )

        self.first_upconv_block = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(1024, 1024, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(1024,512,(2,2))
        )

        self.second_upconv_block = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 512, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(512, 512, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(512,256,(2,2))
        )

        self.third_upconv_block = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(256, 256, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(256,128,(2,2))
        )

        self.fourth_upconv_block = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose2d(128,64,(2,2))
        )

        self.final_block = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, (3,3), padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, out_classes, (1,1)),
            torch.nn.Softmax()
        )
    
    def forward(self,x :torch.Tensor):
        x1 = self.first_downsampling_block(x)
        x2 = self.second_downsampling_block(x1)
        x3 = self.third_downsampling_block(x2)
        x4 = self.fourth_downsampling_block(x3)
        x  = self.first_upconv_block(x4)
        x  = self.first_upconv_block(torch.cat((x,x4),1))
        x  = self.first_upconv_block(torch.cat((x,x3),1))
        x  = self.first_upconv_block(torch.cat((x,x2),1))
        x  = self.final_block(torch.cat((x,x1),1))
        return x










