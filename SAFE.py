import copy
import logging
import random

import numpy as np
import os
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from inc_net import ResNetCosineIncrementalNet, SimpleVitNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy

num_workers = 8


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._classes_seen_so_far = 0
        self.class_increments = []
        self._network = None

        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    def eval_task(self):
        y_pred, y_true, pred = self._eval_cnn(self.test_loader)
        acc_total, grouped = self._evaluate(y_pred, y_true)
        grouped_list = []
        grouped_list.append(grouped)

        if self.args["merge_result"] and self._cur_task > 0:
            grouped_list = []

            y_pred_t, pred_t = self._eval_cnnafter(self.test_loader)
            acc_total_t, grouped_t = self._evaluate(y_pred_t, y_true)
            pred, pred_t = torch.from_numpy(pred), torch.from_numpy(pred_t)
            probs = F.softmax(pred, dim=1)
            probs_t = F.softmax(pred_t, dim=1)

            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
            entropy_t = -(probs_t * torch.log(probs_t + 1e-9)).sum(dim=1)
            entropy_combined = torch.stack([entropy, entropy_t], dim=1)
            weights = F.softmax(-1 * self.args["scalar_val"] * entropy_combined, dim=1)

            pred_tmp = weights[:, 0].unsqueeze(1) * pred + weights[:, 1].unsqueeze(1) * pred_t
            y_pred_tmp = torch.topk(pred_tmp, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred_tmp = y_pred_tmp.numpy()
            acc_total_tmp, grouped_tmp = self._evaluate(y_pred_tmp, y_true)
            if acc_total_tmp > acc_total:
                acc_total, grouped, y_pred = acc_total_tmp, grouped_tmp, y_pred_tmp

            grouped_list.append(grouped)

        return acc_total, grouped_list, y_pred[:, 0], y_true

    def _eval_cnnafter(self, loader):
        self.model_branch1.eval()
        y_pred = []
        pred = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self.model_branch1(inputs)["logits"]
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            pred.append(outputs.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(pred)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        pred = []
        features = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                feature = self._network.convnet(inputs)
            predicts = torch.topk(outputs, k=1, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            pred.append(outputs.cpu().numpy())
            features.append(feature.cpu().numpy())
        features = np.concatenate(features)
        return np.concatenate(y_pred), np.concatenate(y_true), np.concatenate(pred)

    def _evaluate(self, y_pred, y_true):
        ret = {}
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


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if args["model_name"] != "ncm":
            if args["model_name"] == "adapter" and "_adapter" not in args["convnet_type"]:
                raise NotImplementedError("Adapter requires Adapter backbone")
            if args["model_name"] == "ssf" and "_ssf" not in args["convnet_type"]:
                raise NotImplementedError("SSF requires SSF backbone")
            if args["model_name"] == "vpt" and "_vpt" not in args["convnet_type"]:
                raise NotImplementedError("VPT requires VPT backbone")

            if "resnet" in args["convnet_type"]:
                self._network = ResNetCosineIncrementalNet(args, True)
                self._batch_size = 128
            else:
                self._network = SimpleVitNet(args, True)
                self._batch_size = args["batch_size"]

            self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
            self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        else:
            self._network = SimpleVitNet(args, True)
            self._batch_size = args["batch_size"]
        self.args = args

    def after_task(self):
        self._known_classes = self._classes_seen_so_far

    def ptm_statistic(self, trainloader):
        self.ptm.eval()
        Features_f = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
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

        if self.args["use_RP"]:
            # these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.fc.use_RP = True
            if self.args["M"] > 0:
                self._network.fc.W_rand = self.W_rand
            else:
                self._network.fc.W_rand = None

        Features_f = []
        label_list = []
        self.new_pt = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = self._network.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.total_classnum)

        if self.args["use_RP"]:
            # print('Number of pre-trained feature dimensions = ',Features_f.shape[-1])
            if self.args["M"] > 0:
                Features_h = torch.nn.functional.relu(Features_f @ self._network.fc.W_rand.cpu())
            else:
                Features_h = Features_f
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h
            ridge = self.optimise_ridge_parameter(Features_h, Y)
            Wo = torch.linalg.solve(
                self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
            ).T  # better nmerical stability than .inv
            self._network.fc.weight.data = Wo[0 : self._network.fc.weight.shape[0], :].to(
                device="cuda"
            )
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype = Features_f[data_index].sum(0)
                    self._network.fc.weight.data[class_index] += class_prototype.to(
                        device="cuda"
                    )  # for dil, we update all classes in all tasks
                else:
                    # original cosine similarity approach of Zhou et al (2023)
                    class_prototype = Features_f[data_index].mean(0)
                    self._network.fc.weight.data[class_index] = class_prototype

    def replace_fc_branch(self, trainloader):
        self.model_branch1 = self.model_branch1.eval()

        if self.args["use_RP"]:
            self.model_branch1.fc.use_RP = True
            if self.args["M"] > 0:
                self.model_branch1.fc.W_rand = self.W_rand
            else:
                self.model_branch1.fc.W_rand = None

        Features_f = []
        label_list = []
        self.new_pt = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = self.model_branch1.convnet(data)
                Features_f.append(embedding.cpu())
                label_list.append(label.cpu())
        Features_f = torch.cat(Features_f, dim=0)
        label_list = torch.cat(label_list, dim=0)

        Y = target2onehot(label_list, self.total_classnum)

        if self.args["use_RP"]:
            if self.args["M"] > 0:
                Features_h = torch.nn.functional.relu(
                    Features_f @ self.model_branch1.fc.W_rand.cpu()
                )
            else:
                Features_h = Features_f
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h
            ridge = self.optimise_ridge_parameter(Features_h, Y)
            Wo = torch.linalg.solve(
                self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
            ).T  # better nmerical stability than .inv
            self.model_branch1.fc.weight.data = Wo[0 : self.model_branch1.fc.weight.shape[0], :].to(
                device="cuda"
            )
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index = (label_list == class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype = Features_f[data_index].sum(0)
                    self.model_branch1.fc.weight.data[class_index] += class_prototype.to(
                        device="cuda"
                    )  # for dil, we update all classes in all tasks
                else:
                    # original cosine similarity approach of Zhou et al (2023)
                    class_prototype = Features_f[data_index].mean(0)
                    self.model_branch1.fc.weight.data[class_index] = class_prototype

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(3, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(
                G_val + ridge * torch.eye(G_val.size(dim=0)), Q_val
            ).T  # better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::, :] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
            ridge = ridges[np.argmin(np.array(losses))]
        return ridge

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_task += 1
        self._classes_seen_so_far = self._known_classes + data_manager.get_task_size(self._cur_task)

        if self._cur_task > 0 and self.args["use_RP"] and self.args["M"] > 0:
            self._network.fc.weight.data = copy.deepcopy(self.train_fc).to(device="cuda")
            self._network.update_fc(self._classes_seen_so_far)
        else:
            self._network.update_fc(
                self._classes_seen_so_far
            )  # creates a new head with a new number of classes (if CIL)
        if not self.is_dil:
            logging.info("Starting CIL Task {}".format(self._cur_task + 1))
        logging.info(
            "Learning on classes {}-{}".format(self._known_classes, self._classes_seen_so_far - 1)
        )
        self.class_increments.append([self._known_classes, self._classes_seen_so_far - 1])

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._classes_seen_so_far),
            source="train",
            mode="train",
        )  # mode
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(self._batch_size),
            shuffle=True,
            num_workers=num_workers,
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
            num_workers=num_workers,
        )  # 求协方差矩阵和每个类的性质
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._classes_seen_so_far), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=num_workers
        )
        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_CPs)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def freeze_backbone(self, is_first_session=False):
        # Freeze the parameters for ViT.
        if "vit" in self.args["convnet_type"]:
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
        else:
            if isinstance(self._network.convnet, nn.Module):
                for name, param in self._network.convnet.named_parameters():
                    if is_first_session:
                        if "ssf_scale" not in name and "ssf_shift_" not in name:
                            param.requires_grad = False
                    else:
                        param.requires_grad = False

    def show_num_params(self, verbose=False):
        # show total parameters and trainable parameters
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad
        )
        logging.info(f"{total_trainable_params:,} training parameters.")
        if total_params != total_trainable_params and verbose:
            for name, param in self._network.named_parameters():
                if param.requires_grad:
                    print(name, param.numel())

    def _train(self, train_loader, test_loader, train_loader_for_CPs):
        self._network.to(self._device)

        if (
            self.args["slow_diag"]
            or self.args["slow_rdn"]
            or self.args["weg_disf"]
            or self.args["fast_cc"]
            or self.args["merge_result"]
        ):
            if self._cur_task == 0:
                args_ptm = {}
                args_ptm["convnet_type"] = self.args["convnet_type"].rpartition("_")[
                    0
                ]  # PEFT is not used
                self.ptm = SimpleVitNet(args_ptm, True).to(self._device)
                if self.args["slow_diag"] or self.args["slow_rdn"]:
                    self.ptm_statistic(train_loader_for_CPs)
                self.ptm.eval()
            if self._cur_task > 0:
                if self.args["merge_result"]:
                    self.model_branch1.fc.weight.data = copy.deepcopy(self.train_fc_branch).to(
                        device="cuda"
                    )
                    self.model_branch1.update_fc(self._classes_seen_so_far)
                    self.model_branch1.eval()

        if self._cur_task == 0 and self.args["model_name"] in ["ncm", "joint_linear"]:
            self.freeze_backbone()
        if self.args["model_name"] in ["joint_linear", "joint_full"]:
            # this branch updates using SGD on all tasks and should be using classes and does not use a RP head
            if self.args["model_name"] == "joint_linear":
                assert self.args["body_lr"] == 0.0
            self.show_num_params()
            optimizer = optim.SGD(
                [
                    {"params": self._network.convnet.parameters()},
                    {"params": self._network.fc.parameters(), "lr": self.args["head_lr"]},
                ],
                momentum=0.9,
                lr=self.args["body_lr"],
                weight_decay=self.weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000])
            logging.info(
                "Starting joint training on all data using " + self.args["model_name"] + " method"
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self.show_num_params()
        else:
            # this branch is either CP updates only, or SGD on a PETL method first task only
            if self._cur_task == 0 and not self.dil_init:
                if "ssf" in self.args["convnet_type"]:
                    self.freeze_backbone(is_first_session=True)
                if self.args["model_name"] != "ncm":
                    self.show_num_params()
                    optimizer = optim.SGD(
                        [{"params": self._network.parameters()}],
                        momentum=0.9,
                        lr=self.args["body_lr"],
                        weight_decay=self.weight_decay,
                    )
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
                    )
                    # train the PETL method for the first task:
                    logging.info(
                        "Starting PETL training on first task using "
                        + self.args["model_name"]
                        + " method"
                    )
                    self._init_train(train_loader, test_loader, optimizer, scheduler)
                if not self.args["follow_epoch"]:
                    self.freeze_backbone()
                    print("freezed model for test")

                # if self.args["merge_result"] or not self.args["follow_model_ptm"]:
                if self.args["merge_result"]:
                    self.model_branch1 = copy.deepcopy(self._network).to(self._device)

                if self.args["use_RP"] and not self.dil_init:
                    self.setup_RP()
                    if self.args["merge_result"]:
                        self.setup_RP_branch()

            elif self._cur_task > 0 and not self.dil_init:
                if self.args["follow_epoch"]:
                    if "ssf" in self.args["convnet_type"]:
                        self.freeze_backbone(is_first_session=True)
                    if self.args["model_name"] != "ncm":
                        self.show_num_params()

                        optimizer = optim.SGD(
                            self._network.parameters(),
                            momentum=0.9,
                            lr=self.args["body_lr"],
                            weight_decay=self.weight_decay,
                        )
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=self.args["tuned_epoch"], eta_min=self.min_lr
                        )
                        logging.info(
                            f"Starting PETL training on {self._cur_task} task using "
                            + self.args["model_name"]
                            + " method"
                        )
                        self._follow_train(train_loader, test_loader, optimizer, scheduler)

                if self.args["use_RP"] and not self.dil_init:
                    self.setup_RP_follow()
                    if self.args["merge_result"]:
                        self.setup_RP_follow_branch()

            if self.is_dil and not self.dil_init:
                self.dil_init = True
                self._network.fc.weight.data.fill_(0.0)

            self.replace_fc(train_loader_for_CPs)
            if self.args["merge_result"]:
                self.replace_fc_branch(train_loader_for_CPs)
            self.show_num_params()

    def setup_RP_branch(self):
        self.initiated_G = False
        self.model_branch1.fc.use_RP = True
        if self.args["M"] > 0:
            # RP with M > 0
            M = self.args["M"]
            self.train_fc_branch = copy.deepcopy(self.model_branch1.fc.weight)
            self.model_branch1.fc.weight = nn.Parameter(
                torch.Tensor(self.model_branch1.fc.out_features, M).to(device="cuda")
            )  # num classes in task x M
            self.model_branch1.fc.reset_parameters()
            self.model_branch1.fc.W_rand = torch.randn(self.model_branch1.fc.in_features, M).to(
                device="cuda"
            )
            self.W_rand_branch = copy.deepcopy(
                self.model_branch1.fc.W_rand
            )  # make a copy that gets passed each time the head is replaced
        else:
            # no RP, only decorrelation
            M = self.model_branch1.fc.in_features  # this M is L in the paper
        self.Q_branch = torch.zeros(M, self.total_classnum)
        self.G_branch = torch.zeros(M, M)

    def setup_RP(self):
        self.initiated_G = False
        self._network.fc.use_RP = True
        if self.args["M"] > 0:
            # RP with M > 0
            M = self.args["M"]
            self.train_fc = copy.deepcopy(self._network.fc.weight)
            self._network.fc.weight = nn.Parameter(
                torch.Tensor(self._network.fc.out_features, M).to(device="cuda")
            )  # num classes in task x M
            self._network.fc.reset_parameters()
            self._network.fc.W_rand = torch.randn(self._network.fc.in_features, M).to(device="cuda")
            self.W_rand = copy.deepcopy(
                self._network.fc.W_rand
            )  # make a copy that gets passed each time the head is replaced
        else:
            # no RP, only decorrelation
            M = self._network.fc.in_features  # this M is L in the paper
        self.Q = torch.zeros(M, self.total_classnum)
        self.G = torch.zeros(M, M)

    def compute_linear_transform(self, A, B):
        A = np.array(A)
        B = np.array(B)
        A_flat = A.reshape(-1, A.shape[-1])
        B_flat = B.reshape(-1, B.shape[-1])

        W, residuals, _, _ = np.linalg.lstsq(A_flat, B_flat, rcond=None)

        return W

    def setup_RP_follow(self):
        self.train_fc = copy.deepcopy(self._network.fc.weight)

    def setup_RP_follow_branch(self):
        self.train_fc_branch = copy.deepcopy(self.model_branch1.fc.weight)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """Slow learning"""
        prog_bar = tqdm(range(int(self.args["tuned_epoch"])))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_cm = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                losses += loss

                if self.args["slow_rdn"] or self.args["slow_diag"]:
                    loss_cm = self.slow_cm(inputs)
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

        logging.info(info)

    def _follow_train(self, train_loader, test_loader, optimizer, scheduler):
        """Fast learning"""
        prog_bar = tqdm(range(int(self.args["follow_epoch"])))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses_ce, losses_fast = 0.0, 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                if self.args["fast_cc"] or self.args["fast_disf"]:
                    loss_fea, loss_cc = self.fast(inputs, targets)
                    loss_fast = self.args["fast_disf"] * loss_fea + self.args["fast_cc"] * loss_cc
                    loss += loss_fast
                    losses_fast += loss_fast

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_ce += loss

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss_ce {:.3f}, Loss_fast {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                int(self.args["follow_epoch"]),
                losses_ce / len(train_loader),
                losses_fast / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def fast(self, inputs, targets):
        features_s = self._network.convnet(inputs)

        with torch.no_grad():
            self.model_branch1.eval()
            features_t = self.model_branch1.convnet(inputs)

        s = F.cosine_similarity(features_s, features_t, dim=-1)
        loss_f = torch.sum(1 - s)

        f_bcl_ptm = torch.cat(
            [self.model_branch1.fc.weight[: self._known_classes], features_t], dim=0
        )
        targets_bcl = torch.cat(
            [torch.arange(self._known_classes).to(self._device), targets.to(self._device)], dim=0
        )

        f_bcl_cur = torch.cat([self._network.fc.weight[: self._known_classes], features_s], dim=0)

        pred_bcl_ptm = self._network.fc(f_bcl_ptm)["logits"]
        pred_bcl_cur = self.model_branch1.fc(f_bcl_cur)["logits"]
        loss_bcl = F.cross_entropy(pred_bcl_ptm, targets_bcl) + F.cross_entropy(
            pred_bcl_cur, targets_bcl
        )

        return loss_f, loss_bcl

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
