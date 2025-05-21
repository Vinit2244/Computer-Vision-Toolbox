import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator
from visualization import FasterRCNNVisualizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


class FasterRCNNWithVisualization(torch.nn.Module):
    """
    Wrapper around Faster RCNN to capture intermediate outputs for visualization
    """
    def __init__(self, model, visualizer, val_images, val_targets, hyperparams_name):
        super().__init__()
        self.model = model
        self.visualizer = visualizer
        self.val_images = val_images
        self.val_targets = val_targets
        self.hyperparams_name = hyperparams_name
        self.iteration = 0
        
        # Hook for RPN to capture objectness scores and proposals
        self.rpn_objectness_scores = None
        self.rpn_proposals = None
        self.rpn_positive_anchors = None
        self.rpn_negative_anchors = None
        self.rpn_all_anchors = None
        
        # Hook for ROI heads to capture final boxes and scores
        self.roi_final_boxes = None
        self.roi_final_scores = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture intermediate outputs"""
        # Hook for RPN forward
        def rpn_forward_hook(module, input, output):
            # output is (proposals, proposal_losses)
            if self.training:
                self.rpn_objectness_scores = module.head.objectness_scores
                self.rpn_proposals = output[0]
                
                # Get anchor assignments if available
                if hasattr(module, 'positive_anchors') and hasattr(module, 'negative_anchors'):
                    self.rpn_positive_anchors = module.positive_anchors
                    self.rpn_negative_anchors = module.negative_anchors
                    self.rpn_all_anchors = module.all_anchors
        
        # Hook for ROI heads forward
        def roi_heads_forward_hook(module, input, output):
            # output is (detections, detector_losses)
            if self.training:
                # Extract final boxes and scores from detections
                if len(output[0]) > 0:
                    self.roi_final_boxes = [det['boxes'] for det in output[0]]
                    self.roi_final_scores = [det['scores'] for det in output[0]]
        
        # Register hooks
        self.model.rpn.register_forward_hook(rpn_forward_hook)
        self.model.roi_heads.register_forward_hook(roi_heads_forward_hook)
    
    def forward(self, images, targets=None):
        """Forward pass through the model"""
        return self.model(images, targets)
    
    def train(self, mode=True):
        """Set training mode"""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)
    
    def visualize_validation_batch(self):
        """Visualize the current state on validation batch"""
        # Skip if no validation data
        if not self.val_images or not self.val_targets:
            return
        
        # Set to eval mode temporarily
        was_training = self.training
        self.eval()
        
        with torch.no_grad():
            # Forward pass on validation batch
            val_output = self(self.val_images)
            
            # Visualize for each image in the batch
            for i in range(len(self.val_images)):
                # 1. Visualize objectness maps
                if self.rpn_objectness_scores:
                    self.visualizer.visualize_objectness_map(
                        None,  # feature maps not needed
                        self.rpn_objectness_scores,
                        i,
                        self.iteration
                    )
                
                # 2. Visualize object proposals
                if self.rpn_proposals and i < len(self.rpn_proposals):
                    self.visualizer.visualize_proposals(
                        self.val_images[i],
                        self.rpn_proposals[i],
                        i,
                        self.iteration,
                        f"{self.hyperparams_name}"
                    )
                
                # 3. Visualize anchor assignments
                if (self.rpn_positive_anchors and self.rpn_negative_anchors and 
                    i < len(self.rpn_positive_anchors) and i < len(self.rpn_negative_anchors)):
                    self.visualizer.visualize_anchor_assignments(
                        self.val_images[i],
                        self.rpn_positive_anchors[i],
                        self.rpn_negative_anchors[i],
                        self.rpn_all_anchors[i] if self.rpn_all_anchors else None,
                        i,
                        self.iteration
                    )
                
                # 4. Visualize ROI predictions
                if (self.rpn_proposals and self.roi_final_boxes and self.roi_final_scores and
                    i < len(self.rpn_proposals) and i < len(self.roi_final_boxes) and i < len(self.roi_final_scores)):
                    self.visualizer.visualize_roi_predictions(
                        self.val_images[i],
                        self.rpn_proposals[i],
                        self.roi_final_boxes[i],
                        self.roi_final_scores[i],
                        i,
                        self.iteration
                    )
        
        # Restore training mode
        if was_training:
            self.train()
        
        # Increment iteration counter
        self.iteration += 1


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']
    model_config = config['model_params']
    
    # Extract hyperparameters for visualization
    hyperparams = {
        'rpn_fg_threshold': model_config['rpn_fg_threshold'],
        'rpn_bg_threshold': model_config['rpn_bg_threshold'],
        'rpn_batch_size': model_config['rpn_batch_size'],
        'rpn_pos_fraction': model_config['rpn_pos_fraction'],
        'roi_batch_size': model_config['roi_batch_size'],
        'roi_pos_fraction': model_config['roi_pos_fraction']
    }
    
    # Create a name for this hyperparameter set
    hyperparams_name = f"fg{hyperparams['rpn_fg_threshold']}_bg{hyperparams['rpn_bg_threshold']}_rpnbs{hyperparams['rpn_batch_size']}_rpnpf{hyperparams['rpn_pos_fraction']}"
    
    # Create visualizer
    visualizer = FasterRCNNVisualizer(output_dir=f"visualizations/{hyperparams_name}")

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Load datasets
    st_train = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    st_val = SceneTextDataset('val', root_dir=dataset_config['root_dir'])

    train_dataset = DataLoader(st_train,
                               batch_size=4,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_function)
    
    val_dataset = DataLoader(st_val,
                             batch_size=4,
                             shuffle=False,
                             num_workers=4,
                             collate_fn=collate_function)

    # Get a fixed validation batch for visualization
    val_iter = iter(val_dataset)
    val_ims, val_targets, _ = next(val_iter)
    val_images = [im.float().to(device) for im in val_ims]
    val_processed_targets = []
    for target in val_targets:
        processed_target = {}
        processed_target['boxes'] = target['bboxes'].float().to(device)
        processed_target['labels'] = target['labels'].long().to(device)
        val_processed_targets.append(processed_target)

    # Create model with custom RPN parameters
    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        min_size=model_config['min_im_size'],
        max_size=model_config['max_im_size'],
        rpn_pre_nms_top_n_train=model_config['rpn_train_prenms_topk'],
        rpn_pre_nms_top_n_test=model_config['rpn_test_prenms_topk'],
        rpn_post_nms_top_n_train=model_config['rpn_train_topk'],
        rpn_post_nms_top_n_test=model_config['rpn_test_topk'],
        rpn_nms_thresh=model_config['rpn_nms_threshold'],
        rpn_fg_iou_thresh=model_config['rpn_fg_threshold'],
        rpn_bg_iou_thresh=model_config['rpn_bg_threshold'],
        rpn_batch_size_per_image=model_config['rpn_batch_size'],
        rpn_positive_fraction=model_config['rpn_pos_fraction'],
        box_batch_size_per_image=model_config['roi_batch_size'],
        box_positive_fraction=model_config['roi_pos_fraction'],
        box_score_thresh=model_config['roi_score_threshold'],
        box_nms_thresh=model_config['roi_nms_threshold'],
        box_detections_per_img=model_config['roi_topk_detections']
    )
    
    # Modify RPN to store anchor assignments
    original_assign_targets = faster_rcnn_model.rpn.assign_targets_to_anchors
    
    def assign_targets_with_storage(self, anchors, targets):
        labels, matched_gt_boxes = original_assign_targets(anchors, targets)
        
        # Store positive and negative anchors for visualization
        self.positive_anchors = []
        self.negative_anchors = []
        self.all_anchors = []
        
        for i, (anchor_per_image, label_per_image) in enumerate(zip(anchors, labels)):
            positive_indices = torch.where(label_per_image == 1)[0]
            negative_indices = torch.where(label_per_image == 0)[0]
            
            self.positive_anchors.append(anchor_per_image[positive_indices])
            self.negative_anchors.append(anchor_per_image[negative_indices])
            self.all_anchors.append(anchor_per_image)
        
        return labels, matched_gt_boxes
    
    faster_rcnn_model.rpn.assign_targets_to_anchors = assign_targets_with_storage.__get__(faster_rcnn_model.rpn)

    # Replace box predictor for the correct number of classes
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'])

    # Wrap model with visualization capabilities
    model = FasterRCNNWithVisualization(
        faster_rcnn_model, 
        visualizer, 
        val_images, 
        val_processed_targets,
        hyperparams_name
    )

    model.train()
    model.to(device)
    
    if not os.path.exists(train_config['task_name']):
        os.makedirs(train_config['task_name'], exist_ok=True)

    optimizer = torch.optim.SGD(
        lr=train_config['lr'],
        params=filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=5E-5, 
        momentum=0.9
    )

    num_epochs = train_config['num_epochs']
    step_count = 0
    visualization_interval = args.vis_interval  # Visualize every N steps

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            
            images = [im.float().to(device) for im in ims]
            batch_losses = model(images, targets)
            
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()
            step_count += 1
            
            # Visualize at regular intervals
            if step_count % visualization_interval == 0:
                model.visualize_validation_batch()
        
        print(f'Finished epoch {i}')
        
        # Save model checkpoint
        torch.save(
            model.state_dict(), 
            os.path.join(
                train_config['task_name'],
                f"tv_frcnn_r50fpn_{hyperparams_name}_{train_config['ckpt_name']}"
            )
        )
        
        # Print loss information
        loss_output = ''
        loss_output += f'RPN Classification Loss : {np.mean(rpn_classification_losses):.4f}'
        loss_output += f' | RPN Localization Loss : {np.mean(rpn_localization_losses):.4f}'
        loss_output += f' | FRCNN Classification Loss : {np.mean(frcnn_classification_losses):.4f}'
        loss_output += f' | FRCNN Localization Loss : {np.mean(frcnn_localization_losses):.4f}'
        print(loss_output)
        
        # Visualize at the end of each epoch
        model.visualize_validation_batch()
    
    # Create animations from the saved visualizations
    visualizer.create_all_animations(list(range(len(val_images))), model.iteration)
    
    print('Done Training...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training with visualization')
    parser.add_argument('--config', dest='config_path', default='config/st.yaml', type=str)
    parser.add_argument('--vis-interval', dest='vis_interval', default=50, type=int,
                        help='Interval for visualization (in steps)')
    args = parser.parse_args()
    train(args) 