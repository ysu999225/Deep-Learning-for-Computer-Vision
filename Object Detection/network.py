import torch
import math
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = -1 // divisor
    num_groups = 32 // divisor
    eps = 1e-5  # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine
    )


class Anchors(nn.Module):
    def __init__(
        self,
        stride,
        sizes=[4, 4 * math.pow(2, 1 / 3), 4 * math.pow(2, 2 / 3)],
        aspect_ratios=[0.5, 1, 2],
    ):
        """
        Args:
            stride: stride of the feature map relative to the original image
            sizes: list of sizes (sqrt of area) of anchors in units of stride
            aspect_ratios: list of aspect ratios (h/w) of anchors
        __init__ function does the necessary precomputations.
        forward function computes anchors for a given feature map

        Ex. if you are trying the compute the anchor boxes for the above image,
        you should store a 9*4 tensor in self.anchor_offsets(because we have 9 types of anchor boxes
        and each anchor box has 4 offsets from the location of the center of the box).
        HINT: Try using a double for loop to loop over every possible
        combination of sizes and aspect ratios to find out the coordinates of the anchor box for that particular size and aspect ratio.

        When calculating the width and height of the anchor box for a particular size and aspect ratio, remember that if the anchor box were a square
        then the length of each side would just be size*stride; however, we want the height and width of our anchor box to have a different
        width and height depending on the aspect ratio, so think about how you would use the aspect ratio to appropriatly scale the width and height such that w*h = (size*stride)^2 and
        aspect_ratio = h/w
        """
        super(Anchors, self).__init__()
        self.stride = stride
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        #initialize the empty list of anchors
        self.anchor_offsets = []
        # HINT: Try using a double for loop to loop over every possible 
        # combination of sizes and aspect ratios 
        for size in self.sizes:
            for aspect_ratio in self.aspect_ratios:
                h = size * self.stride * math.sqrt(aspect_ratio)
                w = size * self.stride / math.sqrt(aspect_ratio)
                # convert the list to pytorch
                coordinates = torch.tensor([-w / 2, -h / 2, w / 2, h / 2])
                self.anchor_offsets.append(coordinates)
        self.anchor_offsets = torch.stack(self.anchor_offsets,dim=0)
    

        

    def forward(self, x):
        """
        Args:
            x: feature map of shape (B, C, H, W)
        Returns:
            anchors: list of anchors in the format (x1, y1, x2, y2)(in other words (xmin, ymin, xmax, ymax)), giving the shape
            of (B, A*4, H, W) where A is the number of types of anchor boxes we have.
            Hint: We want to loop over every pixel of the input and use that as the center of our bounding boxes. Then we can apply the offsets for each bounding box that we
            found earlier to get all the bounding boxes for that pixel. However, remember that this feature map is scaled down from the original image, so
            when finding the base y, x values(the location of the center of the anchor box with respect to the original image), remember that you need to multiply the current position
            in our feature map (i,j) by the stride to get what the position of the center would be in the base image.
            Hint2: the anchor boxes will be identical for all elements of the batch so try just calculating the anchor boxes for one element of the batch and
                then using torch.repeat to duplicate the tensor B times
            Hint3, remember to transfer your anchors to the same device that the input x is on before returning(you can access this with x.device). This is so we can use a gpu when training.

            MAKING CODE EFFICIENT: We recommend first just using for loops and to make sure that your logic is correct. However, this will be very slow. Therefore,
            we recommend using torch.mesh grid to create the grid of all possible y and x values and then adding them to your anchor offsets tensor that you stored from before
            to get tensors for x1, x2, y1, and y2 for all the anchor boxes. Then you should simply stack those tensors together, reshape them to match the expected output
            size and use torch.repeat to repeat that tensor B times across the batch dimension.
            Your final code should be fully verterized and not have any for loops. 
            Also make sure that when you create a tensor you put it on the same device that x is on.
        """
        B, C, H, W = x.size()
        device = x.device
        anchors = torch.zeros(B, 36, H, W).to(device)
        #Making code efficient
        y_range = torch.arange(0, H).to(device) * self.stride
        x_range = torch.arange(0, W).to(device) * self.stride
        # ycenter and xcenter for each point in the feature map
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='xy')  
        
        # reshapes this tensor to have dimensions [1,9,4]
        anchor_offsets = self.anchor_offsets.view(1, 9, 4).to(device)
        # set a tensor of shape [H, W, 4] and for each grid [i,j,i,j]
        coordinate = torch.stack([x_grid, y_grid, x_grid, y_grid], dim=-1).view(H, W, 1, 4)
        # [H,W,9,4]
        anchors = anchor_offsets + coordinate
        # flatten the dimension to [H,W,36] and add a dimension [1,H,W,36]
        anchors = anchors.view(H, W, -1).unsqueeze(0)
        #[1,36,H,W]
        anchors = anchors.permute(0, 3, 1, 2)
        #[B,36,H,W] repeat in the batch
        anchors = anchors.repeat(B, 1, 1, 1)
        

        return anchors


class RetinaNet(nn.Module):
    def __init__(self, p67=False, fpn=False):
        super(RetinaNet, self).__init__()
        self.resnet = [
            create_feature_extractor(
                resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
                return_nodes={
                    "layer2.3.relu_2": "conv3",
                    "layer3.5.relu_2": "conv4",
                    "layer4.2.relu_2": "conv5",
                },
            )
        ]
        self.resnet[0].eval()
        self.cls_head, self.bbox_head = self.get_heads(10, 9)

        self.p67 = p67
        self.fpn = fpn

        anchors = nn.ModuleList()

        self.p5 = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0),
            group_norm(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
        )
        self._init(self.p5)
        anchors.append(Anchors(stride=32))

        if self.p67:
            self.p6 = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p6)
            self.p7 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                group_norm(256),
            )
            self._init(self.p7)
            anchors.append(Anchors(stride=64))
            anchors.append(Anchors(stride=128))

        if self.fpn:
            self.p4_lateral = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                group_norm(256),
            )
            self.p4 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p4)
            self._init(self.p4_lateral)
            anchors.append(Anchors(stride=16))

            self.p3_lateral = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0), group_norm(256)
            )
            self.p3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), group_norm(256)
            )
            self._init(self.p3)
            self._init(self.p3_lateral)
            anchors.append(Anchors(stride=8))

        self.anchors = anchors

    def _init(self, modules):
        for layer in modules.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def to(self, device):
        super(RetinaNet, self).to(device)
        self.anchors.to(device)
        self.resnet[0].to(device)
        return self

    def get_heads(self, num_classes, num_anchors, prior_prob=0.01):
        cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(
                256, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
            ),
        )
        bbox_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            group_norm(256),
            nn.ReLU(),
            nn.Conv2d(256, num_anchors * 4, kernel_size=3, stride=1, padding=1),
        )

        # Initialization
        for modules in [cls_head, bbox_head]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(cls_head[-1].bias, bias_value)

        return cls_head, bbox_head

    def get_ps(self, feats):
        conv3, conv4, conv5 = feats["conv3"], feats["conv4"], feats["conv5"]
        p5 = self.p5(conv5)
        outs = [p5]

        if self.p67:
            p6 = self.p6(conv5)
            outs.append(p6)

            p7 = self.p7(p6)
            outs.append(p7)

        if self.fpn:
            p4 = self.p4(
                self.p4_lateral(conv4)
                + nn.Upsample(size=conv4.shape[-2:], mode="nearest")(p5)
            )
            outs.append(p4)

            p3 = self.p3(
                self.p3_lateral(conv3)
                + nn.Upsample(size=conv3.shape[-2:], mode="nearest")(p4)
            )
            outs.append(p3)
        # outs = [outs[:]]
        return outs

    def forward(self, x):
        with torch.no_grad():
            feats = self.resnet[0](x)

        feats = self.get_ps(feats)

        # apply the class head and box head on top of layers
        outs = []
        for f, a in zip(feats, self.anchors):
            cls = self.cls_head(f)
            bbox = self.bbox_head(f)
            outs.append((cls, bbox, a(f)))
        return outs
