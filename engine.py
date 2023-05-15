# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import torch.nn as nn
from util.utils import slprint, to_device
import numpy as np
import torch
import csv
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets, target_gaze, face, head_channel, gaze_heatmap in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        faces = []
        head_channels = []
        gaze_heatmaps = []
        gaze_boxes = []

        for f in face:
            faces.append(f)
        faces = torch.stack(faces, 0)
        faces = faces.to(device)

        for h in head_channel:
            head_channels.append(h)
        head_channels = torch.stack(head_channels, 0)
        head_channels = head_channels.to(device)

        for g in gaze_heatmap:
            gaze_heatmaps.append(g)
        gaze_heatmaps = torch.stack(gaze_heatmaps, 0)
        gaze_heatmaps = gaze_heatmaps.to(device)

        for b in target_gaze:
            gaze_box = b['gaze_box']
            gaze_boxes.append(gaze_box)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                # enter network #
                outputs, gaze_outputs = model(samples, faces, head_channels, targets)
            else:
                outputs, gaze_outputs = model(samples, faces, head_channels)
        
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            gaze_heatmap_pred = gaze_outputs.squeeze(1)

            # GOO Loss
            # l2 loss computed only for inside case
            mse_loss = nn.MSELoss(reduce=False)
            l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmaps)
            l2_loss = torch.mean(l2_loss, dim=1)
            l2_loss = torch.mean(l2_loss, dim=1)
            gaze_loss = torch.mean(l2_loss)
            loss_amp_factor = 1000
            loss_dict.update({'loss_gaze': gaze_loss})
            weight_dict.update({'loss_gaze': loss_amp_factor})

            # GOD loss
            box_energy_loss = compute_energy_loss(gaze_boxes, gaze_heatmap_pred) # gaze_heatmap_pred
            loss_energy_factor = 10
            loss_dict.update({'loss_energy': box_energy_loss})
            weight_dict.update({'loss_energy': loss_energy_factor})


            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)



        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        # False
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        # False

        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # False
        if args.onecyclelr:
            lr_scheduler.step()
        # False

        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    all_gazepoints = []
    all_predmap = []
    all_gtmap = []
    total_error = []
    gaze_val_loss = 0

    for samples, targets, target_gaze, face, head_channel, gaze_heatmap in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        faces = []
        head_channels = []
        gaze_heatmaps = []
        gaze_boxes = []
        for f in face:
            faces.append(f)
        faces = torch.stack(faces, 0)
        faces = faces.to(device)

        for h in head_channel:
            head_channels.append(h)
        head_channels = torch.stack(head_channels, 0)
        head_channels = head_channels.to(device)

        for g in gaze_heatmap:
            gaze_heatmaps.append(g)
        gaze_heatmaps = torch.stack(gaze_heatmaps, 0)
        gaze_heatmaps = gaze_heatmaps.to(device)

        for b in target_gaze:
            gaze_box = b['gaze_box']
            gaze_boxes.append(gaze_box)

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, gaze_outputs = model(samples, faces, head_channels, targets)
            else:
                outputs, gaze_outputs = model(samples, faces, head_channels)
            # outputs = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        gaze_heatmap_pred = gaze_outputs.squeeze(1)
        # GOO Loss
        # l2 loss computed only for inside case
        mse_loss = nn.MSELoss(reduce=False)
        l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmaps)
        l2_loss = torch.mean(l2_loss, dim=1)
        l2_loss = torch.mean(l2_loss, dim=1)
        gaze_loss = torch.mean(l2_loss)
        gaze_val_loss += gaze_loss
        loss_amp_factor = 1000

        box_energy_loss = compute_energy_loss(gaze_boxes, gaze_heatmap_pred)  # gaze_heatmap_pred
        loss_energy_factor = 10
        loss_dict.update({'loss_energy': box_energy_loss})
        weight_dict.update({'loss_energy': loss_energy_factor})


        loss_dict.update({'loss_gaze': gaze_loss})
        weight_dict.update({'loss_gaze': loss_amp_factor})


        # Obtaining GAZE eval metrics
        final_output = [g.cpu().data.numpy() for g in gaze_heatmap_pred]  # gaze_heatmap_pred.cpu().data.numpy()
        target_gaze_point = [g['gaze_point'].cpu().data.numpy() for g in target_gaze]  # target_gaze[0]['gaze_point'].cpu().data.numpy()
        eye_position = [g['eye'].cpu().data.numpy() for g in target_gaze]  # target_gaze[0]['eye'].cpu().data.numpy()

        # f_point = final_output
        # gt_point = target_gaze_point
        # eye_point = eye_position
        for f_point, gt_point, eye_point in \
                zip(final_output, target_gaze_point, eye_position):
            out_size = 64  # Size of heatmap
            heatmap = np.copy(f_point)
            f_point = f_point.reshape([out_size, out_size])

            h_index, w_index = np.unravel_index(f_point.argmax(), f_point.shape) # the index of largest
            f_point = np.array([w_index / out_size, h_index / out_size])
            f_error = f_point - gt_point
            f_dist = np.sqrt(f_error[0] ** 2 + f_error[1] ** 2)

            # angle
            f_direction = f_point - eye_point
            gt_direction = gt_point - eye_point

            norm_f = (f_direction[0] ** 2 + f_direction[1] ** 2) ** 0.5
            norm_gt = (gt_direction[0] ** 2 + gt_direction[1] ** 2) ** 0.5

            f_cos_sim = (f_direction[0] * gt_direction[0] + f_direction[1] * gt_direction[1]) / \
                        (norm_gt * norm_f + 1e-6)
            f_cos_sim = np.maximum(np.minimum(f_cos_sim, 1.0), -1.0)
            f_angle = np.arccos(f_cos_sim) * 180 / np.pi

            # AUC calculation
            heatmap = np.squeeze(heatmap)
            heatmap = cv2.resize(heatmap, (5, 5))
            gt_heatmap = np.zeros((5, 5))
            x, y = list(map(int, gt_point * 5))
            gt_heatmap[y, x] = 1.0

            all_gazepoints.append(f_point)
            all_predmap.append(heatmap)
            all_gtmap.append(gt_heatmap)
            total_error.append([f_dist, f_angle])



        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gaze
    l2, ang = np.mean(np.array(total_error), axis=0)
    all_gazepoints = np.vstack(all_gazepoints)
    all_predmap = np.stack(all_predmap).reshape([-1])
    all_gtmap = np.stack(all_gtmap).reshape([-1])
    auc = roc_auc_score(all_gtmap, all_predmap)
    rows = [auc, l2, ang]

    with open('logs/score.csv', 'a', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(rows)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res



def compute_energy_loss(box, heatmap):
    '''
    Use ground truth box and predicted heatmap to compute the energy aggregation loss
    Input:
        box : list[cx, cy, w, h] normlized
        heatmap : [batch_size, output_size, output_size]
    '''
    batch_size = heatmap.size()[0]
    power, total_power = 0., 0.
    for i in range(batch_size):
        cur_box = box[i]
        cur_heatmap = heatmap[i]
        xmin, ymin, xmax, ymax = cur_box[0] - cur_box[2]/2, cur_box[1] - cur_box[3]/2, \
            cur_box[0] + cur_box[2]/2, cur_box[1] + cur_box[3]/2

        xmin, ymin, xmax, ymax = math.floor(xmin * 64), math.floor(ymin * 64), math.ceil(xmax * 64), math.ceil(ymax * 64)

        # total_power = total_power + torch.sum(cur_heatmap)
        box_w = xmax - xmin + 1
        box_h = ymax - ymin + 1
        power = power + torch.sum(cur_heatmap[ymin: min(ymax + 1, 64), xmin: min(xmax + 1, 64)]) / (box_w * box_h)
    if power < 0:
        energy_rate = 0.99
    else:
        energy_rate = 1 - power / batch_size
    # total_power = total_power / batch_size
    # energy_rate = power / total_power
    return energy_rate