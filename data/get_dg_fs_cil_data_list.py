import numpy as np

if __name__ == "__main__":
    domains = ["real", "infograph", "painting", "sketch", "clipart", "quickdraw"]
    n = 5
    n_classes = 345

    infograph = [
        "leaf",
        "flashlight",
        "ladder",
        "umbrella",
        "fork",
        "shoe",
        "panda",
        "teapot",
        "bush",
        "washing_machine",
        "saxophone",
        "frog",
        "police_car",
        "traffic_light",
        "train",
        "baseball_bat",
        "stitches",
        "stove",
        "pear",
        "arm",
        "lobster",
        "snowflake",
        "wheel",
        "squirrel",
        "popsicle",
        "cruise_ship",
        "pencil",
        "trumpet",
        "snorkel",
        "helmet",
        "rake",
        "feather",
        "bench",
        "remote_control",
        "toilet",
    ]
    painting = [
        "giraffe",
        "flower",
        "jail",
        "parrot",
        "pants",
        "drill",
        "church",
        "flying_saucer",
        "map",
        "suitcase",
        "carrot",
        "mailbox",
        "palm_tree",
        "hockey_stick",
        "skyscraper",
        "axe",
        "mountain",
        "dragon",
        "steak",
        "chair",
        "chandelier",
        "knife",
        "floor_lamp",
        "backpack",
        "airplane",
        "pool",
        "waterslide",
        "penguin",
        "table",
        "bridge",
        "cat",
        "laptop",
        "necklace",
        "megaphone",
        "couch",
    ]
    sketch = [
        "dolphin",
        "hamburger",
        "paint_can",
        "candle",
        "bucket",
        "sun",
        "microwave",
        "piano",
        "banana",
        "sandwich",
        "coffee_cup",
        "duck",
        "potato",
        "sleeping_bag",
        "key",
        "skull",
        "snowman",
        "skateboard",
        "tiger",
        "pizza",
        "mushroom",
        "submarine",
        "face",
        "lantern",
        "guitar",
        "wine_bottle",
        "spoon",
        "ice_cream",
        "bed",
        "clock",
        "diving_board",
        "spider",
        "teddy-bear",
        "hospital",
        "motorbike",
    ]

    # Source dataset
    for i, domain in enumerate(domains[:4]):
        data_list = []
        with open("/workspaces/SAFE/data/DomainNet/" + domain + "_train.txt", "r") as f:
            txt = f.readlines()
        cnt = np.zeros(n_classes)
        for line in txt:
            idx = int(line.split(" ")[-1])
            if cnt[idx] == n:
                continue
            else:
                cls = line.split(" ")[0].split("/")[1]
                if (
                    (i == 0 and idx < 240)
                    or (i == 1 and cls in infograph)
                    or (i == 2 and cls in painting)
                    or (i == 3 and cls in sketch)
                ):
                    cnt[idx] += 1
                    data_list.append(line)
        with open("./DomainNet/fs/" + domain + "_train.txt", "w") as f:
            f.write("".join(data_list))

    # Target dataset
    for i, domain in enumerate(domains[4:]):
        data_list = []
        with open("/workspaces/SAFE/data/DomainNet/" + domain + "_test.txt", "r") as f:
            txt = f.readlines()
        cnt = np.zeros(n_classes)
        for line in txt:
            idx = int(line.split(" ")[-1])
            if cnt[idx] == n:
                continue
            else:
                cnt[idx] += 1
                data_list.append(line)
        with open("./DomainNet/fs/" + domain + "_test.txt", "w") as f:
            f.write("".join(data_list))
