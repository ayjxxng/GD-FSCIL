import argparse
from trainer import train
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="domainnet", type=str, choices=["domainnet"])
    parser.add_argument("--mode", default="style_meta", type=str, choices=["base", "meta", "style_meta"])
    parser.add_argument("--first_source", default="real", type=str, choices=["real", "infograph", "painting", "sketch"])
    parser.add_argument("--aug_p", default=0.9, type=float)

    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--model_name", default="adapter", type=str, choices=["adapter", "ssf"])
    parser.add_argument(
        "--convnet_type",
        default="pretrained_vit_b16_224_adapter",
        type=str,
        choices=["pretrained_vit_b16_224_adapter", "pretrained_vit_b16_224_ssf"],
    )

    parser.add_argument("--init_cls", default=240, type=int)
    parser.add_argument("--increment", default=35, type=int)

    parser.add_argument("--seed", default=1998, type=int)
    parser.add_argument("--batch_size", default=48, type=int)
    parser.add_argument("--n_inner_steps", default=4, type=int) # NEW
    parser.add_argument("--tuned_epoch", default=20, type=int)
    parser.add_argument("--follow_epoch", default=15, type=int)

    parser.add_argument("--body_lr", default=0.05, type=float)
    parser.add_argument("--head_lr", default=0.01, type=float)
    parser.add_argument("--meta_lr", default=0.01, type=float)
    parser.add_argument("--inner_lr", default=0.01, type=float)
    parser.add_argument("--meta_momentum", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--meta_weight_decay", default=0.0, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--use_RP", default=True, type=bool)
    parser.add_argument("--M", default=768, type=int)
    parser.add_argument("--use_input_norm", default=False, type=bool)
    parser.add_argument("--fast_disf", default=0.5, type=float)
    parser.add_argument("--slow_diag", default=5e-10, type=float)
    parser.add_argument("--slow_rdn", default=5e-9, type=float)

    parser.add_argument("--merge_result", default=1, type=float)
    parser.add_argument("--fast_cc", default=1, type=float)
    parser.add_argument("--scalar_val", default=1, type=float)

    parser.add_argument("--device", default=[0], type=int, nargs="+")
    parser.add_argument("--save_path", default="logs", type=str)
    parser.add_argument("--use_task0_ckpt", default=True, type=bool)

    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='SAFE')
    return parser.parse_args()


def main():
    args = parse_argument()
    args = vars(args)
    args["seed"] = [args["seed"]]
    args["do_not_save"] = False

    unique_id = datetime.now().strftime("%m-%d-%H-%M")
    args['unique_id'] = unique_id
    if args['mode'] == 'style_meta':
        args["exp_name"] = f"{unique_id}_{args['mode']}_{args['first_source']}_p{args['aug_p']}_LR{args['meta_lr']}_{args['meta_weight_decay']}_{args['seed'][0]}"
        args["task0_ckpt"] = f"./checkpoints/{args['model_name']}/{args['dataset']}/{args['first_source']}/{args['mode']}/model_task0_seed{args['seed'][0]}_lr{args['body_lr']}_d{args['weight_decay']}_p{args['aug_p']}.pth"
    else:
        args["exp_name"] = f"{unique_id}_{args['mode']}_{args['first_source']}_LR{args['meta_lr']}_{args['meta_weight_decay']}_{args['seed'][0]}"
        args["task0_ckpt"] = f"./checkpoints/{args['model_name']}/{args['dataset']}/{args['first_source']}/{args['mode']}/model_task0_seed{args['seed'][0]}_lr{args['body_lr']}_d{args['weight_decay']}.pth"

    args["wandb_run_name"] = args["exp_name"]
    os.makedirs(os.path.dirname(args["task0_ckpt"]), exist_ok=True)
    args["logfilename"] = "{}/{}/{}/{}/{}/{}".format(
        args["save_path"],
        args["model_name"],
        args["dataset"],
        args["first_source"],
        args["mode"],
        args["exp_name"],
    )
    os.makedirs(os.path.dirname(args["logfilename"]), exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
