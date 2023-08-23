from pre_import import *
from openstl.models import simvp_model
class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.conv_gx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv_gx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.conv_gy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv_gy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        self.eps=10**-3
        # self.eps=0.

    def forward(self, im):
        B,T,C,H,W = im.shape
        im = im.view(B*T, C, H, W)

        p2d = (1,1,1,1)
        if im.size(1) == 1:
            g_x = self.conv_gx(im)
            g_y = self.conv_gy(im)
            g = torch.sqrt(torch.pow(0.5 * g_x, 2) + torch.pow(0.5 * g_y, 2) + self.eps)
            g = torch.nn.functional.pad(g, p2d, "constant", 0)
        else:

            for kk in range(0, im.size(1)):
                g_x = self.conv_gx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                g_y = self.conv_gy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                ng = torch.sqrt(torch.pow(0.5 * g_x, 2) + torch.pow(0.5 * g_y, 2)+ self.eps)
                print(ng.shape)
                ng = torch.nn.functional.pad(ng, p2d, "constant", 0)
                print(ng.shape)
                if kk == 0:
                    g = ng
                else:
                    g = torch.cat((g, ng), dim=1)
        g = g.view(B,T,C,H,W)
        return g
    

class GSimVP(torch.nn.Module):
    def __init__(self, in_shape, hid_T=128):
        super(GSimVP, self).__init__()
        self.simvp = simvp_model.SimVP_Model(in_shape=in_shape, hid_T=hid_T)
        self.g_img = Gradient_img()
        

    def forward(self, img):
        output = self.simvp(img)
        edge = self.g_img(output)
        return output, edge
    