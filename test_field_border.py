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
import torch.nn as nn 
import torch.nn.functional as F


ROOT_DIR = "/raid/n.kotov1/sar_data/set0_p256_s128_3ch_fsplit_sat_resc_border"
# NET_NAME = 'IncFCN_rmsprop_dice'
NET_NAME = 'Deeplabv3plus_ResNet50_rmsprop_dice'
DATA_NAME = 'Germany_Africa_field_border'
working_path = os.path.dirname(os.path.abspath(__file__))

args = {
    "device": 3,
    "encoder": "resnet50",
    "classes": 2,
    "channels": 3,
    "save_pred": False,
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

    # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5'
    # device = torch.device("cuda")

    device = torch.cuda.set_device(args['device'])
    device = torch.device(f"cuda:{args['device']}")

    model = smp.DeepLabV3Plus(encoder_name=args["encoder"], 
                   encoder_weights="imagenet", 
                   in_channels=args["channels"], 
                   classes=args["classes"])

    # model = Inc_FCN(args["channels"], num_classes=args["classes"])    
    # model = nn.DataParallel(model)
    model = model.to(device)

    all_weight = sorted(glob(os.path.join(args["weight"], "*.pth"), recursive=True))

    for w in tqdm(all_weight, total=len(all_weight)):
        model.load_state_dict(torch.load(w, map_location=device))
        model.eval()

        metrics_score = np.zeros((2, 5))
        metrics_score = list(metrics_score)

        finfo_path = os.path.join(args["results"], os.path.basename(w).split(".")[0] + ".txt")[-1:]
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
                pred_y = F.softmax(pred_y, dim=1)
                print(pred_y.min(),pred_y.max())

                transform_y = torch.zeros((pred_y.shape))

                for i in range(transform_y.shape[1]):
                    transform_y[:, i, :, :] = torch.where(y.float() == torch.tensor(i+1).float().cuda(), y.float(), torch.tensor(0).float().cuda())

                multiclass_metrics = list()
                
                for i in range(transform_y.shape[1]):
                    score = calculate_metrics(transform_y[:, i, :, :], pred_y[:, i, :, :])
                    multiclass_metrics.append(score)

                metrics_score = list(map(add, metrics_score, multiclass_metrics))

                pred_y = pred_y[0].cpu().numpy()        ## (1, 256, 256)
                pred_y = pred_y > 0.5
                pred_y = np.array(pred_y, dtype=np.uint8)

            if args["save_pred"]:
                ori_mask = mask_parse(mask)
                pred_y_field = mask_parse(pred_y[0, :, :])
                pred_y_border = mask_parse(pred_y[1, :, :])
                line = np.ones((size[1], 10, 3)) * 255
                cat_images = np.concatenate([image * 255, line, ori_mask * 255, line, pred_y_field * 127, line, pred_y_border * 127], axis=1)
                io.imsave(os.path.join(args["imgs_pred"], f"{name}.png"), cat_images)

        jaccard_f = metrics_score[0][0]/len(test_x)
        f1_f = metrics_score[0][1]/len(test_x)
        recall_f = metrics_score[0][2]/len(test_x)
        precision_f = metrics_score[0][3]/len(test_x)
        acc_f = metrics_score[0][4]/len(test_x)

        jaccard_b = metrics_score[1][0]/len(test_x)
        f1_b = metrics_score[1][1]/len(test_x)
        recall_b = metrics_score[1][2]/len(test_x)
        precision_b = metrics_score[1][3]/len(test_x)
        acc_b = metrics_score[1][4]/len(test_x)
        
        finfo.write(f"Jaccard: field {jaccard_f:1.4f} border {jaccard_b:1.4f}\n")
        finfo.write(f"F1: field {f1_f:1.4f} border {f1_b:1.4f}\n")
        finfo.write(f"Recall: field {recall_f:1.4f} border {recall_b:1.4f}\n")
        finfo.write(f"Precision: field {precision_f:1.4f} border {precision_b:1.4f}\n")
        finfo.write(f"Accuracy: field {acc_f:1.4f} border {acc_b:1.4f}\n")
        finfo.close()


if __name__ == "__main__":
    main()
