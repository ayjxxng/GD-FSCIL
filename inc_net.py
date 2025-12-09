import copy
import logging
import math
import torch
from torch import nn
import timm
from torch.nn import functional as F


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()
        self.use_RP = False

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(10)

    def forward(self, input):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if self.W_rand is not None:
                a = input @ self.W_rand
                inn = torch.nn.functional.relu(a)

            else:
                inn = input
            out = F.linear(inn, self.weight)

        if self.sigma is not None:
            out = self.sigma * out

        return out


def get_convnet(args, pretrained=False):
    name = args["convnet_type"].lower()
    # Resnet
    if name == "pretrained_resnet50":
        from resnet import resnet50

        model = resnet50(pretrained=True, args=args)
        return model.eval()
    elif name == "pretrained_resnet152":
        from resnet import resnet152

        model = resnet152(pretrained=True, args=args)
        return model.eval()
    elif name == "vit_base_patch32_224_clip_laion2b":
        # note: even though this is "B/32" it has nearly the same num params as the standard ViT-B/16
        model = timm.create_model(
            "vit_base_patch32_224_clip_laion2b", pretrained=True, num_classes=0
        )
        model.out_dim = 768
        return model.eval()

    # NCM or NCM w/ Finetune
    elif name == "pretrained_vit_b16_224" or name == "vit_base_patch16_224":
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()
    elif name == "pretrained_vit_b16_224_in21k" or name == "vit_base_patch16_224_in21k":
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        model.out_dim = 768
        return model.eval()

    # SSF
    elif "_ssf" in name:
        from petl import vision_transformer_ssf  # registers vit_base_patch16_224_ssf

        if name == "pretrained_vit_b16_224_ssf":
            model = timm.create_model("vit_base_patch16_224_ssf", pretrained=True, num_classes=0)
            model.out_dim = 768
        elif name == "pretrained_vit_b16_224_in21k_ssf":
            model = timm.create_model(
                "vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0
            )
            model.out_dim = 768
        return model.eval()

    # VPT
    elif "_vpt" in name:
        from petl.vpt import build_promptmodel

        if name == "pretrained_vit_b16_224_vpt":
            basicmodelname = "vit_base_patch16_224"
        elif name == "pretrained_vit_b16_224_in21k_vpt":
            basicmodelname = "vit_base_patch16_224_in21k"

        VPT_type = "Deep"

        Prompt_Token_num = 5  # args["prompt_token_num"]

        model = build_promptmodel(
            modelname=basicmodelname, Prompt_Token_num=Prompt_Token_num, VPT_type=VPT_type
        )
        prompt_state_dict = model.obtain_prompt()
        model.load_prompt(prompt_state_dict)
        model.out_dim = 768
        return model.eval()

    elif "_adapter" in name:
        ffn_num = 64  # args["ffn_num"]
        if "adapter" in args["model_name"]:
            from petl import vision_transformer_adapter
            from easydict import EasyDict

            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
            )
            if name == "pretrained_vit_b16_224_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_adapter(
                    num_classes=0,
                    global_pool=False,
                    drop_path_rate=0.0,
                    tuning_config=tuning_config,
                )
                model.out_dim = 768
            elif name == "pretrained_vit_b16_224_in21k_adapter":
                model = vision_transformer_adapter.vit_base_patch16_224_in21k_adapter(
                    num_classes=0,
                    global_pool=False,
                    drop_path_rate=0.0,
                    tuning_config=tuning_config,
                )
                model.out_dim = 768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    else:
        raise NotImplementedError("Unknown type {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass


class ResNetCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat(
                [weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()]
            )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc


class StyleAug(nn.Module):
    def __init__(self, p=0.5, mode="mix", alpha=0.3, tau=0.1):
        super().__init__()
        self.p = p
        self.mode = mode
        self.alpha = alpha
        self.tau = tau

    def forward(self, z):
        # z: (B, N, D)  # ViT patch tokens
        if not self.training or torch.rand(1).item() > self.p:
            return z

        B, N, D = z.shape
        mu = z.mean(dim=1, keepdim=True)      # (B, 1, D)
        std = z.std(dim=1, keepdim=True) + 1e-6

        z_norm = (z - mu) / std

        if self.mode == "mix":
            # batch shuffle
            perm = torch.randperm(B, device=z.device)
            mu2, std2 = mu[perm], std[perm]
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,1,1)).to(z.device)
            mu_t = lam * mu + (1-lam) * mu2
            std_t = lam * std + (1-lam) * std2
        else:  # "gauss"
            noise_mu = torch.randn_like(mu) * self.tau
            noise_std = torch.randn_like(std) * self.tau
            mu_t = mu + noise_mu
            std_t = std + noise_std

        z_aug = std_t * z_norm + mu_t
        return z_aug


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.style_aug = StyleAug(p=0.9, mode="mix")

    def update_fc(self, nb_classes):
        fc = CosineLinear(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            weight = torch.cat(
                [weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()]
            )
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def forward(self, x, use_style_aug=False, return_features: bool = False):
        if use_style_aug:
            B = x.shape[0]
            x = self.convnet.patch_embed(x)
            x = self.style_aug(x)
            cls_tokens = self.convnet.cls_token.expand(
                B, -1, -1
            )  
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.convnet.pos_embed
            x = self.convnet.pos_drop(x)

            for idx, blk in enumerate(self.convnet.blocks):
                x = blk(x)

            x = self.convnet.norm(x)
            x = x[:, 0]
            x = self.convnet.head(x)
        else:
            x = self.convnet(x)
        out = self.fc(x)
        if return_features:
            return {"logits": out, "features": x}
        else:
            return {"logits": out}
