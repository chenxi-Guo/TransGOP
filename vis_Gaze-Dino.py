import os, sys
import torch, json
import numpy as np
import json
import skimage
from main import build_model_main
from util.slconfig import SLConfig
from PIL import ImageDraw, ImageFont
from datasets import build_dataset
# from datasets.coco import ConvertCocoPolysToMask
from util.visualizer import COCOVisualizer
import cv2
from util import box_ops
from torchvision import transforms
from PIL import Image
import datasets.transforms as T


def resize(img,width,height):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask_h = height
    mask_w = width
    img = np.array(img)
    cur_h = height
    cur_w = width
    img = cv2.resize(img, (cur_w, cur_h))
    start_y = (mask_h - img.shape[0]) // 2
    start_x = (mask_w - img.shape[1]) // 2
    mask = np.zeros([mask_h, mask_w, 3]).astype(np.uint8)
    mask[start_y: start_y + img.shape[0], start_x: start_x + img.shape[1], :] = img
    return mask

def ConvertCocoPolysToMask(image, target, target_gaze):
    w, h = image.size

    image_id = target["image_id"]
    image_id = torch.tensor([image_id])

    anno = target["annotations"]
    anno_gaze = target_gaze["annotations"]

    # gaze
    eye = [obj["head_point"] for obj in anno_gaze]

    gaze_point = [obj["gaze_point"] for obj in anno_gaze]

    try:
        gaze_point = torch.as_tensor(gaze_point, dtype=torch.float32).reshape(2)
    except:
        pass
    try:
        gaze_box = [obj["gaze_bbox"] for obj in anno_gaze]
    except:
        pass
    gaze_box = torch.as_tensor(gaze_box, dtype=torch.float32).reshape(4)
    gaze_box[2:] += gaze_box[:2]
    gaze_box[0::2].clamp_(min=0, max=w)
    gaze_box[1::2].clamp_(min=0, max=h)
    gaze_box_category_id = [obj["category_id"] for obj in anno_gaze]
    gaze_box_category_id = torch.tensor(gaze_box_category_id, dtype=torch.int64)

    # head box
    k = 0.1
    eyex, eyey = eye[0]
    x_min = eyex - (0.15 * w)
    y_min = eyey - (0.15 * h)
    x_max = eyex + (0.15 * w)
    y_max = eyey + (0.15 * h)
    # x_min = eyex - 0.15 * (h / 2)
    # y_min = eyey - 0.15 * (h / 2)
    # x_max = eyex + 0.15 * (h / 2)
    # y_max = eyey + 0.15 * (h / 2)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max < 0:
        x_max = 0
    if y_max < 0:
        y_max = 0
    x_min -= k * abs(x_max - x_min)
    y_min -= k * abs(y_max - y_min)
    x_max += k * abs(x_max - x_min)
    y_max += k * abs(y_max - y_min)
    if x_min < 0:
        x_min = 0
    if y_min < 0:
        y_min = 0
    if x_max < 0:
        x_max = 0
    if y_max < 0:
        y_max = 0
    if x_min > w:
        x_min = w
    if y_min > h:
        y_min = h
    if x_max > w:
        x_max = w
    if y_max > h:
        y_max = h
    head_box = [x_min, y_min, x_max, y_max]

    head_box = torch.as_tensor(head_box, dtype=torch.float32).reshape(4)
    eye = torch.as_tensor(eye, dtype=torch.float32).reshape(2)
    target_gaze = {}
    target_gaze["head_box"] = head_box
    target_gaze["eye"] = eye
    target_gaze["gaze_box"] = gaze_box
    target_gaze["gaze_point"] = gaze_point
    target_gaze["orig_size"] = torch.as_tensor([int(h), int(w)])
    target_gaze["labels"] = gaze_box_category_id
    target_gaze["image_id"] = image_id
    target_gaze["size"] = torch.as_tensor([int(h), int(w)])

    # gaze #

    # object detection #
    anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)

    keypoints = None
    if anno and "keypoints" in anno[0]:
        keypoints = [obj["keypoints"] for obj in anno]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    if keypoints is not None:
        keypoints = keypoints[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes

    target["image_id"] = image_id
    if keypoints is not None:
        target["keypoints"] = keypoints

    # for conversion to coco api
    area = torch.tensor([obj["area"] for obj in anno])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    # object detection #

    return image, target, target_gaze

def generate_att_map(image_, heatmap, image_name, gaze_box, img_save_dir):
    unloader = transforms.ToPILImage()
    a = heatmap
    hmap = a.cpu().clone()
    hmap = unloader(hmap)
    hmap.save('gaze.jpg')
    heatmap_path=os.getcwd()+'/gaze.jpg'
    # image=image_
    res = img_save_dir + image_name
    # img = skimage.io.imread(image_)
    ##################################
    # img=image_
    # img=Image.open(image)
    img=np.array(image_)
    ##################################
    width=img.shape[1]
    height=img.shape[0]
    img_new = resize(img,width,height)
    amap = cv2.cvtColor(skimage.io.imread(heatmap_path), cv2.COLOR_RGB2BGR)
    new_map = cv2.resize(amap, (img_new.shape[1], img_new.shape[0]))
    normed_mask = new_map / np.max(new_map)
    normed_mask = np.uint8(255 * normed_mask)
    normed_mask = cv2.applyColorMap(normed_mask, cv2.COLORMAP_JET)
    normed_mask = cv2.addWeighted(img_new, 0.9, normed_mask, 0.9, 0)
    skimage.io.imsave(res, cv2.cvtColor(normed_mask, cv2.COLOR_BGR2RGB))
    # img=Image.open(res)
    # img.show()
    x=0
    return x

if __name__ == '__main__':
    model_config_path = "config/DINO/DINO_4scale.py" # change the path of the model config file
    model_checkpoint_path = "logs_256_pretrain_real/best_gaze.pth" # change the path of the model checkpoint
    # See our Model Zoo section in README.md for more details about our pretrained models.
    img_dir = "/data1/gcx002/Datasets/gooreal/gaze-dino/val2017"
    img_save_dir = "vis_GOP_real/"
    ann = "/data1/gcx002/Datasets/gooreal/gaze-dino/annotations/instances_val2017.json"

    args = SLConfig.fromfile(model_config_path)
    args.device = 'cuda'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    n = 0
    for i in os.listdir(img_dir):
        n += 1
        image = Image.open(os.path.join(img_dir, i)).convert("RGB")
        # image_path = "/data1/gcx002/Datasets/goosynth10000/gaze_dino/val2017/10269.png"
        image_name = i
        w, h = image.size
        if os.path.exists(os.path.join(img_save_dir, i)):
            print('文件已存在')
            continue
        else:
            # image = Image.open(image_path).convert("RGB")# load image
            # transform images
            with open (ann,'r') as f:
                json_data = json.load(f)

            img_data = json_data['images']
            gaze_ann = json_data['annotations_gaze']
            od_ann = json_data['annotations']

            for i in img_data:
                if i['file_name'] == image_name:
                    id = i['id']
                    pass

            target_gaze = []
            target_gaze.append(gaze_ann[id])
            target = []

            for o in od_ann:
                if o['image_id'] == id:
                    target.append(o)

            target_gaze = {'image_id': id, 'annotations': target_gaze}
            target = {'image_id': id, 'annotations': target}
            img, target, target_gaze = ConvertCocoPolysToMask(image, target, target_gaze)

            transform = T.Compose([
                    T.RandomResize([224]),
                    T.gaze_postprocess(224, 64),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            img, target, target_gaze, face, head_channel, gaze_heatmap = transform(img, target, target_gaze)

            output, gaze_output = model.cuda()(img[None].cuda(), face[None].cuda(), head_channel[None].cuda())
            # output = model.cuda()(image[None].cuda())

            gaze_output = (gaze_output - gaze_output.min()) / (gaze_output.max() - gaze_output.min())
            # output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

            # vslzr = COCOVisualizer()
            #
            # scores = output['scores']
            # labels = output['labels']
            # boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
            # select_mask = scores > thershold
            #
            # box_label = [id2name[int(item)] for item in labels[select_mask]]
            # pred_dict = {
            #     'boxes': boxes[select_mask],
            #     'size': torch.Tensor([image.shape[1], image.shape[2]]),
            #     'box_label': box_label,
            #     'image_id': i.split('.')[-2]
            # }
            # vslzr.visualize(image, pred_dict, savedir="/data1/gcx002/Datasets/gooreal/DINO_query1000_visualazition_gooreal/",
            #                     show_in_console=False)
            gaze_box = target_gaze['gaze_box']
            mul = torch.tensor([w, h, w, h])
            # mul = torch.tensor([640, 480, 640, 480])
            gaze_box = gaze_box * mul

            gaze_box[0] = gaze_box[0] - gaze_box[2] / 2
            gaze_box[1] = gaze_box[1] - gaze_box[3] / 2
            gaze_box[2] = gaze_box[0] + gaze_box[2]
            gaze_box[3] = gaze_box[1] + gaze_box[3]

            draw = ImageDraw.Draw(image)

            left, top, right, bottom = gaze_box
            top = int(top)
            left = int(left)
            bottom = int(bottom)
            right = int(right)
            for j in range(4):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline='red')


            generate_att_map(image, gaze_output.squeeze(1)[-1], image_name, gaze_box, img_save_dir)
            print(n)
    print('done !')


