import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Concat(nn.Module):
    # for PRM-C
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, input):
        return torch.cat(input, 1)


class DownSample(nn.Module):
    def __init__(self, scale):
        super(DownSample, self).__init__()
        self.scale = scale

    def forward(self, x):
        sample = F.interpolate(x, scale_factor=self.scale)
        return sample


class BnResidualConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BnResidualConv1, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.bn(x)
        return self.conv(F.relu(x))


class BnResidualConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BnResidualConv3, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.bn(x)
        return self.conv(F.relu(x))


class PRM_A(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PRM_A, self).__init__()
        self.skip=BnResidualConv1(in_channels,out_channels)

        self.line = nn.Sequential(
            BnResidualConv1(in_channels, int(out_channels / 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BnResidualConv3(int(out_channels / 2), int(out_channels / 2)),
            nn.Upsample(scale_factor=2),
            BnResidualConv1(int(out_channels / 2), out_channels),

        )

        self.conv1=BnResidualConv1(in_channels,int(out_channels/2))
        self.conv2 = BnResidualConv1(in_channels, int(out_channels / 2))
        self.conv3 = BnResidualConv1(in_channels, int(out_channels / 2))
        self.conv4 = BnResidualConv1(in_channels, int(out_channels / 2))
        self.down1=DownSample(scale=pow(2,-1))
        self.down2 = DownSample(scale=pow(2, -0.75))
        self.down3 = DownSample(scale=pow(2, -0.5))
        self.down4 = DownSample(scale=pow(2, -0.25))

        self.f1=BnResidualConv3(int(out_channels/2),out_channels)
        self.f2 = BnResidualConv3(int(out_channels / 2), out_channels)
        self.f3 = BnResidualConv3(int(out_channels / 2), out_channels)
        self.f4 = BnResidualConv3(int(out_channels / 2), out_channels)

        self.conv_g=BnResidualConv1(out_channels,out_channels)

    def forward(self, x):
        size=(x.size()[-2],x.size()[-1])

        res=self.skip(x)

        x0=self.line(x)

        x1=self.conv1(x)
        x1=self.down1(x1)
        x1=self.f1(x1)
        x1=F.interpolate(x1,size=size)

        x2=self.conv2(x)
        x2=self.down2(x2)
        x2=self.f2(x2)
        x2=F.interpolate(x2,size=size)

        x3=self.conv3(x)
        x3=self.down3(x3)
        x3=self.f3(x3)
        x3=F.interpolate(x3,size=size)

        x4=self.conv4(x)
        x4=self.down4(x4)
        x4=self.f4(x4)
        x4=F.interpolate(x4,size=size)

        x=x1+x2+x3+x4

        x=self.conv_g(x)

        out=res+x0+x

        return out





class PRM_B(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PRM_B, self).__init__()
        self.skip=BnResidualConv1(in_channels,out_channels)

        self.line = nn.Sequential(
            BnResidualConv1(in_channels, int(out_channels / 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BnResidualConv3(int(out_channels / 2), int(out_channels / 2)),
            nn.Upsample(scale_factor=2),
            BnResidualConv1(int(out_channels / 2), out_channels),

        )

        self.conv_share=BnResidualConv1(in_channels,int(out_channels/2))
        self.down1=DownSample(scale=pow(2,-1))
        self.down2 = DownSample(scale=pow(2, -0.75))
        self.down3 = DownSample(scale=pow(2, -0.5))
        self.down4 = DownSample(scale=pow(2, -0.25))

        self.conv1=BnResidualConv3(int(out_channels/2),out_channels)
        self.conv2 = BnResidualConv3(int(out_channels / 2), out_channels)
        self.conv3 = BnResidualConv3(int(out_channels / 2), out_channels)
        self.conv4 = BnResidualConv3(int(out_channels / 2), out_channels)

        self.conv_g=BnResidualConv1(out_channels,out_channels)

    def forward(self, x):
        size=(x.size()[-2],x.size()[-1])

        res=self.skip(x)

        x0=self.line(x)

        x=self.conv_share(x)

        x1=self.down1(x)
        x1=self.conv1(x1)
        x1=F.interpolate(x1,size=size)

        x2=self.down2(x)
        x2=self.conv2(x2)
        x2=F.interpolate(x2,size=size)

        x3=self.down3(x)
        x3=self.conv3(x3)
        x3=F.interpolate(x3,size=size)

        x4=self.down4(x)
        x4=self.conv4(x4)
        x4=F.interpolate(x4,size=size)

        x=x1+x2+x3+x4

        x=self.conv_g(x)

        out=res+x0+x

        return out

class PRM_C(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PRM_C, self).__init__()
        self.skip=BnResidualConv1(in_channels,out_channels)
        self.line = nn.Sequential(
            BnResidualConv1(in_channels, int(out_channels / 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BnResidualConv3(int(out_channels / 2), int(out_channels / 2)),
            nn.Upsample(scale_factor=2),
            BnResidualConv1(int(out_channels / 2), out_channels),

        )
        self.conv_share=BnResidualConv1(in_channels,int(out_channels/2))
        self.down1=DownSample(scale=pow(2,-1))
        self.down2 = DownSample(scale=pow(2, -0.75))
        self.down3 = DownSample(scale=pow(2, -0.5))
        self.down4 = DownSample(scale=pow(2, -0.25))

        self.conv1=BnResidualConv3(int(out_channels/2),out_channels)
        self.conv2 = BnResidualConv3(int(out_channels / 2), out_channels)
        self.conv3 = BnResidualConv3(int(out_channels / 2), out_channels)
        self.conv4 = BnResidualConv3(int(out_channels / 2), out_channels)

        self.conv_g=BnResidualConv1(out_channels*4,out_channels)

    def forward(self, x):
        size=(x.size()[-2],x.size()[-1])

        res=self.skip(x)

        x0=self.line(x)

        x=self.conv_share(x)

        x1=self.down1(x)
        x1=self.conv1(x1)
        x1=F.interpolate(x1,size=size)

        x2=self.down2(x)
        x2=self.conv2(x2)
        x2=F.interpolate(x2,size=size)

        x3=self.down3(x)
        x3=self.conv3(x3)
        x3=F.interpolate(x3,size=size)

        x4=self.down4(x)
        x4=self.conv4(x4)
        x4=F.interpolate(x4,size=size)

        x=torch.cat((x1,x2,x3,x4),1)

        x=self.conv_g(x)

        out=res+x0+x

        return out


class PRM_D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PRM_D, self).__init__()

        self.skip=BnResidualConv1(in_channels,out_channels)

        self.line=nn.Sequential(
            BnResidualConv1(in_channels,int(out_channels/2)),
            nn.MaxPool2d(kernel_size=2,stride=2),
            BnResidualConv3(int(out_channels/2),int(out_channels/2)),
            nn.Upsample(scale_factor=2),
            BnResidualConv1(int(out_channels/2),out_channels),

        )


        self.conv_share=BnResidualConv1(in_channels,int(out_channels/2))

        self.dilated_conv1=nn.Sequential(
            nn.BatchNorm2d(int(out_channels/2)),
            nn.ReLU(),
            nn.Conv2d(int(out_channels/2),int(out_channels/2),kernel_size=3,dilation=1,padding=1)
        )
        self.dilated_conv2 = nn.Sequential(
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(out_channels / 2), int(out_channels / 2), kernel_size=3, dilation=2, padding=2)
        )
        self.dilated_conv3 = nn.Sequential(
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(out_channels / 2), int(out_channels / 2), kernel_size=3, dilation=3, padding=3)
        )
        self.dilated_conv4 = nn.Sequential(
            nn.BatchNorm2d(int(out_channels / 2)),
            nn.ReLU(),
            nn.Conv2d(int(out_channels / 2), int(out_channels / 2), kernel_size=3, dilation=4, padding=4)
        )

        self.conv_g=BnResidualConv1(int(out_channels/2),out_channels)

    def forward(self, x):
        size=(x.size()[-2],x.size()[-1])

        res=self.skip(x)

        x0=self.line(x)

        x=self.conv_share(x)

        x1=self.dilated_conv1(x)
        x2 = self.dilated_conv2(x)
        x3 = self.dilated_conv3(x)
        x4 = self.dilated_conv4(x)

        x=x1+x2+x3+x4

        x=self.conv_g(x)

        out=res+x0+x

        return out














if __name__=="__main__":
    input=torch.ones((1,3,224,224))

    # conv1=torch.nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=2,dilation=2)

    net=PRM_A(3,64)
    output=net(input)
    print(output.size())
