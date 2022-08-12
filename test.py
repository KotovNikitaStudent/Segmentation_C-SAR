import os
from operator import add
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from skimage import io
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from utils import create_dir, seeding
import segmentation_models_pytorch as smp
from models.mpresnet import MPResNet
from models.incfcn import Inc_FCN
import torch.nn as nn 
from models.modeling.deeplab import DeepLab


ROOT_DIR = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc"
# ROOT_DIR = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc_median"
# ROOT_DIR = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc_lee"
# ROOT_DIR = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc_sar_cam"
# NET_NAME = 'UNet_ResNet34_rmsprop_dice'
# NET_NAME = 'MPResNet34_adam_bce'
NET_NAME = 'IncFCN_rmsprop_dice'
# NET_NAME = 'Deeplabv3plus_ResNet50_rmsprop_dice'
DATA_NAME = 'Germany_Africa_field'
# DATA_NAME = 'Germany_Africa_field_median'
# DATA_NAME = 'Germany_Africa_field_lee'
# DATA_NAME = 'Germany_Africa_field_sar_cam'
working_path = os.path.dirname(os.path.abspath(__file__))

args = {
    "device": 4,
    "encoder": "resnet50",
    "classes": 1,
    "channels": 3,
    "save_pred": True,
    "weight": os.path.join(working_path, "weight", DATA_NAME, NET_NAME),
    "results": os.path.join(working_path, "results", DATA_NAME, NET_NAME),
    "imgs_pred": os.path.join(working_path, "imgs_pred", DATA_NAME, NET_NAME)
}


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred, zero_division=0)
    score_f1 = f1_score(y_true, y_pred, zero_division=0)
    score_recall = recall_score(y_true, y_pred, zero_division=0)
    score_precision = precision_score(y_true, y_pred, zero_division=0)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (256, 256, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (256, 256, 3)
    return mask


def main():
    seeding(42)
    create_dir(args["results"])
    create_dir(args["imgs_pred"])

    test_x = sorted(glob(ROOT_DIR + "/test/images/*"))
    test_y = sorted(glob(ROOT_DIR + "/test/masks/*"))

    size = (256, 256)

    os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    device = torch.device("cuda")

    # device = torch.cuda.set_device(args['device'])
    # device = torch.device(f"cuda:{args['device']}")
    
    # model = DeepLab(backbone='xception', num_classes=1)

    # model = smp.DeepLabV3Plus(encoder_name=args["encoder"], 
    #                encoder_weights="imagenet", 
    #                in_channels=args["channels"], 
    #                classes=args["classes"])

    # model = smp.Unet(encoder_name=args["encoder"], 
    #                encoder_weights="imagenet", 
    #                in_channels=args["channels"], 
    #                classes=args["classes"])

    # model = MPResNet(args["channels"], num_classes=args["classes"])

    model = Inc_FCN(args["channels"], num_classes=args["classes"])    
    model = nn.DataParallel(model)
    model = model.to(device)

    all_weight = sorted(glob(os.path.join(args["weight"], "*.pth"), recursive=True))[-1:]

    for w in tqdm(all_weight, total=len(all_weight)):
        model.load_state_dict(torch.load(w, map_location=device))
        model.eval()

        metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]

        finfo_path = os.path.join(args["results"], os.path.basename(w).split(".")[0] + ".txt")
        finfo = open(finfo_path, "w+")

        for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
            name = x.split("/")[-1].split(".")[0]

            image = io.imread(x)
            x = np.transpose(image, (2, 0, 1))      ## (3, 256, 256)
            x = np.expand_dims(x, axis=0)           ## (1, 3, 256, 256)
            x = x.astype(np.float32)
            x = torch.from_numpy(x)
            x = x.to(device)

            mask = io.imread(y)
            y = np.expand_dims(mask, axis=0)            ## (1, 256, 256)
            y = np.expand_dims(y, axis=0)               ## (1, 1, 256, 256)
            y = y.astype(np.float32)
            y = torch.from_numpy(y)
            y = y.to(device)

            with torch.no_grad():
                pred_y = model(x)
                # pred_y, _ = model(x)
                pred_y = torch.sigmoid(pred_y)

                score = calculate_metrics(y, pred_y)
                metrics_score = list(map(add, metrics_score, score))
                pred_y = pred_y[0].cpu().numpy()        ## (1, 256, 256)
                pred_y = np.squeeze(pred_y, axis=0)     ## (256, 256)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)

            if args["save_pred"]:
                ori_mask = mask_parse(mask)
                pred_y = mask_parse(pred_y)
                line = np.ones((size[1], 10, 3)) * 128
                cat_images = np.concatenate([image * 255, line, ori_mask * 255, line, pred_y * 255], axis=1)
                io.imsave(os.path.join(args["imgs_pred"], f"{name}.png"), cat_images)

        jaccard = metrics_score[0]/len(test_x)
        f1 = metrics_score[1]/len(test_x)
        recall = metrics_score[2]/len(test_x)
        precision = metrics_score[3]/len(test_x)
        acc = metrics_score[4]/len(test_x)

        finfo.write(f"Jaccard: {jaccard:1.4f}\n")
        finfo.write(f"F1: {f1:1.4f}\n")
        finfo.write(f"Recall: {recall:1.4f}\n")
        finfo.write(f"Precision: {precision:1.4f}\n")
        finfo.write(f"Accuracy: {acc:1.4f}\n")
        finfo.close()


if __name__ == "__main__":
    main()
