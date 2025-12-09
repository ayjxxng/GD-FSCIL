import logging
import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inc_net import SimpleVitNet
from utils.toolkit import tensor2numpy, accuracy
import wandb
from torch.func import functional_call
from collections import OrderedDict


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self._network = None
        self.num_workers = args.get("num_workers", 0)

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

        self.use_wandb = args.get("use_wandb", False)
        self.wandb_project = args.get("wandb_project", "SAFE")

    def eval_task(self):
        y_pred, y_true, pred = self._eval_cnn(self.test_loader)
        acc_total, grouped = self._evaluate(y_pred, y_true)
        grouped_list = [grouped]

        return acc_total, grouped_list, y_pred[:, 0], y_true

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        pred = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            pred.append(outputs.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(pred)

    def _evaluate(self, y_pred, y_true):
        acc_total, grouped = accuracy(
            y_pred.T[0], y_true, self._known_classes, self.class_increments
        )
        return acc_total, grouped

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class MetaLearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"] != "ncm":
            if args["model_name"] == "adapter" and "_adapter" not in args["convnet_type"]:
                raise NotImplementedError("Adapter requires Adapter backbone")
            if args["model_name"] == "ssf" and "_ssf" not in args["convnet_type"]:
                raise NotImplementedError("SSF requires SSF backbone")

            self._network = SimpleVitNet(args, True)
            self._batch_size = args["batch_size"]

            self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8

        self.args = args
        self.num_workers = args.get("num_workers", 0)
        self.inner_lr = self.args.get("inner_lr", 0.01)
        self.meta_lr = self.args.get("meta_lr", 0.01)
        self.meta_momentum = self.args.get("meta_momentum", 0.0)
        self.meta_weight_decay = self.args.get("meta_weight_decay", 0.0)
        self.use_style_aug = True if args["mode"] == 'style_meta' else False

        self.kl_temp = self.args.get("kl_temp", 1.0)
        self.lambda_kl = self.args.get("lambda_kl", 0.5)

        # Task-0 checkpoint path (optional)
        self.use_task0_ckpt = args.get("use_task0_ckpt", False)
        self.task0_ckpt = args.get("task0_ckpt", None)

    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def ptm_statistic(self, trainloader):
        self.ptm.eval()
        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = self.ptm.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
            Features_f = torch.cat(Features_f, dim=0)
            label_list = torch.cat(label_list, dim=0)

        self.ptm_mean = []
        self.ptm_var = []
        self.ptm_std = []
        self.ptm_cov = []
        for class_index in np.unique(self.train_dataset.labels):
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            self.ptm_mean.append(Features_f[data_index].mean(0))
            self.ptm_var.append(Features_f[data_index].var(dim=0, keepdim=True))
            self.ptm_std.append(Features_f[data_index].std(dim=0, keepdim=True))
            deviation = Features_f - self.ptm_mean[class_index]
            cov = torch.matmul(deviation.T, deviation) / (Features_f.size(0) - 1)
            self.ptm_cov.append(cov)

    def replace_fc(self, trainloader):
        self._network = self._network.eval()

        Features_f = []
        label_list = []
        self.new_pt = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # prototype-based cosine head update
        for class_index in np.unique(self.train_dataset.labels):
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            class_prototype = Features_f[data_index].mean(0)
            self._network.fc.weight.data[class_index] = class_prototype

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)

        # creates a new head with a new number of classes (if CIL)
        self._network.update_fc(self._classes_seen_so_far)

        logging.info(":::::::::::::::: Starting Task {}: ".format(self._cur_task + 1)
                     + "Classes {}-{}".format(self._known_classes, self._classes_seen_so_far - 1))
        self.class_increments.append([self._known_classes, self._classes_seen_so_far - 1])
        self._get_loaders(data_manager)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)

    def freeze_backbone(self, is_first_session=False):
        # Freeze the parameters for ViT.
        if isinstance(self._network.convnet, nn.Module):
            for name, param in self._network.convnet.named_parameters():
                if is_first_session:
                    if (
                        "head." not in name
                        and "ssf_scale" not in name
                        and "ssf_shift_" not in name
                    ):
                        param.requires_grad = False
                else:
                    param.requires_grad = False

    def _get_loaders(self, data_manager):
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(self._batch_size),
            shuffle=True,
            num_workers=self.num_workers,
        )

        train_dataset_for_CPs = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train",
            mode="test",
        )
        self.train_loader_for_CPs = DataLoader(
            train_dataset_for_CPs,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._classes_seen_so_far), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self.num_workers
        )

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)

        # ----- Task 0: meta-initialization (SAFE-style) -----
        if self._cur_task == 0:
            args_ptm = {}
            args_ptm["convnet_type"] = self.args["convnet_type"].rpartition("_")[0]
            self.ptm = SimpleVitNet(args_ptm, True).to(self._device)
            if self.args["slow_diag"] or self.args["slow_rdn"]:
                self.ptm_statistic(train_loader_for_CPs)
            self.ptm.eval()

            if "ssf" in self.args["convnet_type"]:
                self.freeze_backbone(is_first_session=True)

            if self.use_task0_ckpt and os.path.isfile(self.task0_ckpt):
                state = torch.load(self.task0_ckpt, map_location=self._device)
                self._network.load_state_dict(state["network"])
                self._known_classes = state.get("known_classes", self._known_classes)
                self._classes_seen_so_far = state.get(
                    "classes_seen_so_far", self._classes_seen_so_far
                )
                logging.info(f"Loaded Task-0 checkpoint from {self.task0_ckpt}")
            else:
                optimizer = optim.SGD(
                    [{"params": self._network.parameters()}],
                    momentum=0.9,
                    lr=self.args["body_lr"],
                    weight_decay=self.weight_decay,
                )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
                )
                self._init_train(train_loader, test_loader, optimizer, scheduler)

                if self.task0_ckpt is not None:
                    torch.save(
                        {
                            "network": self._network.state_dict(),
                            "known_classes": self._known_classes,
                            "classes_seen_so_far": self._classes_seen_so_far,
                            "args": self.args,
                        },
                        self.task0_ckpt,
                    )
                    logging.info(f"Saved Task-0 checkpoint to {self.task0_ckpt}")

        # ----- Task > 0: MAML-style meta-training -----
        elif self._cur_task > 0:
            if "ssf" in self.args["convnet_type"]:
                self.freeze_backbone(is_first_session=True)

            optimizer_meta = optim.SGD(
                self._network.parameters(),
                momentum=self.meta_momentum,
                lr=self.meta_lr,
                weight_decay=self.meta_weight_decay,
            )
            self._meta_train(train_loader, test_loader, optimizer_meta)

        self.replace_fc(train_loader_for_CPs)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """Slow learning (task 0)"""
        prog_bar = tqdm(range(int(self.args["tuned_epoch"])))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_cm = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)

                if self.use_style_aug:
                    logits_style = self._network(inputs, use_style_aug=True)["logits"]
                    p_clean = F.log_softmax(logits / self.kl_temp, dim=-1)
                    p_style = F.softmax(logits_style / self.kl_temp, dim=-1)
                    kl_loss = F.kl_div(p_clean, p_style, reduction="batchmean") * (self.kl_temp ** 2)
                    loss += self.lambda_kl * kl_loss

                losses += loss

                if self.args["slow_rdn"] or self.args["slow_diag"]:
                    loss_cm = self.slow_cm(inputs) # correlation matrix loss
                    loss += loss_cm
                    losses_cm += loss_cm

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss_ce {:.3f}, Loss_cm {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / len(train_loader),
                losses_cm / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
            if self.use_wandb:
                wandb.log(
                    {
                        "task": self._cur_task,
                        "epoch": epoch + 1,
                        "loss_ce": losses / len(train_loader),
                        "loss_cm": losses_cm / len(train_loader),
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "phase": "slow",
                    }
                )

        logging.info(info)

    def _inner_update_maml(self, fast_weights, x_sub, y_sub, second_order=False):
        """MAML inner update using SGD"""
        params = self._get_param_dict(fast_weights)
        out = functional_call(self._network, params, (x_sub,), {"return_features": True})
        logit = out["logits"]
        loss_inner = F.cross_entropy(logit, y_sub)

        if self.use_style_aug:
            out_style = functional_call(self._network, params, (x_sub,), {"use_style_aug": True, "return_features": True})
            logits_style = out_style["logits"]
            p_clean = F.log_softmax(logit / self.kl_temp, dim=-1)
            p_style = F.softmax(logits_style / self.kl_temp, dim=-1)
            kl_loss = F.kl_div(p_clean, p_style, reduction="batchmean") * (self.kl_temp ** 2)
            loss_inner += self.lambda_kl * kl_loss

        # θ' = θ - α ∇_θ ℓ
        grads = torch.autograd.grad(
            loss_inner, fast_weights, create_graph=second_order, retain_graph=second_order,
        )
        new_fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        return new_fast_weights, loss_inner.detach()

    def _meta_train(self, train_loader, test_loader, meta_optimizer):
        """Task > 0: MAML-based inner + outer"""
        prog_bar = tqdm(range(int(self.args["follow_epoch"])))

        n_inner_steps = self.args.get("n_inner_steps", 2) # 4
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            inner_losses, meta_losses = 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                batch_size = inputs.size(0)
                if batch_size < n_inner_steps:
                    n_steps = 1
                    inner_bs = batch_size
                else:
                    n_steps = n_inner_steps
                    inner_bs = batch_size // n_steps

                # ---------- 1) inner loop ----------
                fast_weights = [p for p in self._network.parameters() if p.requires_grad]
                for k in range(n_steps):
                    start = k * inner_bs
                    end = batch_size if k == n_steps - 1 else (k + 1) * inner_bs
                    x_sub = inputs[start:end]
                    y_sub = targets[start:end]

                    fast_weights, loss_inner = self._inner_update_maml(
                        fast_weights, x_sub, y_sub,
                    )
                    inner_losses += loss_inner

                # ---------- 2) meta step ----------
                meta_loss = self._meta_loss(fast_weights, inputs, targets)
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()
                meta_losses += meta_loss.detach()

                with torch.no_grad():
                    logits_full = self._network(inputs)["logits"]
                    _, preds = torch.max(logits_full, dim=1)
                    correct += preds.eq(targets).cpu().sum()
                    total += len(targets)

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)

            info = (
                "Task {}, Epoch {}/{} => Inner_CE {:.3f}, Meta_BCL {:.3f}, Train_ACC {:.2f}, Test_ACC {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                int(self.args["follow_epoch"]),
                inner_losses / len(train_loader),
                meta_losses / max(len(train_loader), 1),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
            if self.use_wandb:
                wandb.log(
                    {
                        "task": self._cur_task,
                        "epoch": epoch + 1,
                        "inner_ce": inner_losses / len(train_loader),
                        "meta_bcl": meta_losses / max(len(train_loader), 1),
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "phase": "fast_meta",
                    }
                )

        logging.info(info)

    def _meta_loss(self, fast_weights, inputs, targets):
        params = self._get_param_dict(fast_weights)
        out = functional_call(self._network, params, (inputs,), {"return_features": True})
        features_fast = out["features"]    # (B, D)
        old_proto = self._network.fc.weight[: self._known_classes]  # (K_old, D)

        f_all = torch.cat([old_proto, features_fast], dim=0)  # (K_old + B, D)
        targets_bcl = torch.cat(
            [torch.arange(self._known_classes, device=self._device), targets], dim=0,
        )
        logits_bcl = self._network.fc(f_all)
        loss_bcl = F.cross_entropy(logits_bcl, targets_bcl)

        if self.args.get("meta_cm", False):
            cm_loss = self.slow_cm(inputs)
            loss_bcl = loss_bcl + self.args.get("lambda_meta_cm", 0.05) * cm_loss

        return loss_bcl

    def _get_param_dict(self, fast_weights):
        param_dict = OrderedDict()
        i = 0
        for name, p in self._network.named_parameters():
            if not p.requires_grad:
                continue
            param_dict[name] = fast_weights[i]
            i += 1
        return param_dict

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def slow_cm(self, inputs):
        features_s = self._network.convnet(inputs)
        features_s = F.normalize(features_s, p=2, dim=-1)

        with torch.no_grad():
            self.ptm.eval()
            features_t = self.ptm.convnet(inputs)
            features_t = F.normalize(features_t, p=2, dim=-1)

        c = torch.matmul(features_s.T, features_t)
        c.div_(features_s.shape[0])
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()

        loss_blt = self.args["slow_diag"] * on_diag + self.args["slow_rdn"] * off_diag
        return loss_blt
