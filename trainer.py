import copy
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import wandb
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from SAFE import Learner
from MetaSAFE import MetaLearner


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    ave_accs = []
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        ave_acc = _train(args)
        ave_accs.append(ave_acc)
    return ave_accs


def _train(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=args["logfilename"] + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Starting new run")
    _set_random()
    _set_device(args)
    print_args(args)

    if args.get("use_wandb", False) and wandb.run is None:
        wandb.init(
            project=args.get("wandb_project", "SAFE"),
            name=args.get("wandb_run_name", None),
            config=args,
            settings=wandb.Settings(quiet=True),
        )

    if args.get("mode", "base") == "base":
        model = Learner(args)
    else:
        model = MetaLearner(args)

    model.dil_init = False
    model.is_dil = False
    data_manager = DataManager(
        args["dataset"],
        args["first_source"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        use_input_norm=args["use_input_norm"],
    )
    num_tasks = data_manager.nb_tasks 
    acc_curve = []
    for i in range(10):
        acc_curve.append({"top1_total": [], "ave_acc": []})

    classes_df = None
    logging.info("Pre-trained network parameters: {}".format(count_parameters(model._network)))
    for task in range(num_tasks):

        if classes_df is None:
            classes_df = pd.DataFrame()
            classes_df["init"] = -1 * np.ones(data_manager._test_data.shape[0])

        model.incremental_train(data_manager)
        acc_total, acc_grouped, predicted_classes, true_classes = model.eval_task()
        model.after_task()

        l = 0
        for d in acc_grouped:
            cur = d
            n = cur.values()
            m = np.round(np.mean(list(n)), 2)

            acc_curve[l]["top1_total"].append(acc_total)
            acc_curve[l]["ave_acc"].append(m)
            logging.info("Group Accuracies: {}".format(cur))
            l += 1
        logging.info("Ave Acc curve: {}".format(acc_curve[0]["ave_acc"]))
        logging.info("Top1 curve: {}".format(acc_curve[0]["top1_total"]))

        if args.get("use_wandb", False):
            wandb.log(
                {
                    "task": task,
                    "task_acc_total": acc_total,
                    "task_ave_acc": acc_curve[0]["ave_acc"][-1],
                }
            )
    logging.info("Finishing run")
    logging.info("")

    if args.get("use_wandb", False):
        log_dict = {
            'top1_total': acc_curve[0]["top1_total"][-1],
        }
        wandb.log(log_dict)
        wandb.finish()
    return acc_curve[0]["top1_total"][-1]


def save_results(args, top1_total, ave_acc, model, classes_df):
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    output_df = pd.DataFrame()
    output_df["top1_total"] = top1_total
    output_df["ave_acc"] = ave_acc
    output_df.to_csv("./results/" + args["dataset"] + "_publish_" + str(args["ID"]) + ".csv")

    if not os.path.exists("./results/class_preds/"):
        os.makedirs("./results/class_preds/")
    classes_df.to_csv(
        "./results/class_preds/"
        + args["dataset"]
        + "_class_preds_publish_"
        + str(args["ID"])
        + ".csv"
    )


def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
