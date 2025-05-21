import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
from tqdm import tqdm

class FasterRCNNVisualizer:
    def __init__(self, output_dir="visualizations"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for different visualizations
        self.objectness_dir = os.path.join(output_dir, "objectness_maps")
        self.proposals_dir = os.path.join(output_dir, "object_proposals")
        self.anchor_dir = os.path.join(output_dir, "anchor_assignments")
        self.roi_dir = os.path.join(output_dir, "roi_predictions")
        
        os.makedirs(self.objectness_dir, exist_ok=True)
        os.makedirs(self.proposals_dir, exist_ok=True)
        os.makedirs(self.anchor_dir, exist_ok=True)
        os.makedirs(self.roi_dir, exist_ok=True)
        
        # Set up color maps
        self.cmap = cm.get_cmap('viridis')
        
    def tensor_to_numpy(self, tensor):
        """Convert a tensor to numpy array for visualization"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def visualize_objectness_map(self, feature_maps, objectness_scores, image_idx, iteration, level_idx=None):
        """
        Visualize objectness score heatmap for feature maps
        
        Args:
            feature_maps: List of feature maps from backbone
            objectness_scores: List of objectness scores from RPN
            image_idx: Index of the image in the batch
            iteration: Current iteration number
            level_idx: If provided, only visualize this level
        """
        levels = range(len(objectness_scores)) if level_idx is None else [level_idx]
        
        for level in levels:
            # Get objectness score for this level
            objectness = objectness_scores[level][image_idx]
            
            # Convert to numpy and get the positive channel (object probability)
            objectness_np = self.tensor_to_numpy(objectness[1])  # Channel 1 is for positive class
            
            # Create directory for this image if it doesn't exist
            img_dir = os.path.join(self.objectness_dir, f"img_{image_idx}")
            os.makedirs(img_dir, exist_ok=True)
            
            # Normalize for better visualization
            objectness_np = (objectness_np - objectness_np.min()) / (objectness_np.max() - objectness_np.min() + 1e-8)
            
            # Create heatmap
            plt.figure(figsize=(8, 8))
            plt.imshow(objectness_np, cmap='hot')
            plt.colorbar(label='Objectness Score')
            plt.title(f"Objectness Map - Level {level} - Iteration {iteration}")
            plt.savefig(os.path.join(img_dir, f"level_{level}_iter_{iteration}.png"))
            plt.close()
    
    def visualize_proposals(self, image, proposals, image_idx, iteration, filename=None):
        """
        Visualize object proposals on an image
        
        Args:
            image: Original image tensor
            proposals: Proposals from RPN
            image_idx: Index of the image in the batch
            iteration: Current iteration number
            filename: Optional filename for the image
        """
        # Convert image tensor to numpy
        image_np = self.tensor_to_numpy(image.permute(1, 2, 0))
        
        # Normalize image for visualization
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create a copy of the image for drawing
        img = (image_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Draw proposals
        proposals_np = self.tensor_to_numpy(proposals)
        for box in proposals_np:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Save the image
        img_dir = os.path.join(self.proposals_dir, f"img_{image_idx}")
        os.makedirs(img_dir, exist_ok=True)
        
        save_path = os.path.join(img_dir, f"iter_{iteration}.png")
        if filename:
            save_path = os.path.join(img_dir, f"{filename}_iter_{iteration}.png")
            
        img_pil.save(save_path)
        
        return img_pil
    
    def visualize_anchor_assignments(self, image, positive_anchors, negative_anchors, all_anchors, 
                                    image_idx, iteration, max_anchors=10):
        """
        Visualize positive and negative anchor assignments
        
        Args:
            image: Original image tensor
            positive_anchors: Anchors with high IoU with ground truth
            negative_anchors: Anchors with low IoU with ground truth
            all_anchors: All anchors for reference
            image_idx: Index of the image in the batch
            iteration: Current iteration number
            max_anchors: Maximum number of anchors to visualize
        """
        # Convert image tensor to numpy
        image_np = self.tensor_to_numpy(image.permute(1, 2, 0))
        
        # Normalize image for visualization
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create a copy of the image for drawing
        img = (image_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Draw a subset of all anchors (very light gray)
        if all_anchors is not None:
            all_anchors_np = self.tensor_to_numpy(all_anchors)
            # Sample a small subset to avoid cluttering
            indices = np.random.choice(len(all_anchors_np), min(100, len(all_anchors_np)), replace=False)
            for idx in indices:
                box = all_anchors_np[idx]
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)
        
        # Draw negative anchors (red)
        if negative_anchors is not None:
            neg_anchors_np = self.tensor_to_numpy(negative_anchors)
            # Limit the number of negative anchors to visualize
            for i, box in enumerate(neg_anchors_np[:max_anchors]):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        
        # Draw positive anchors (green)
        if positive_anchors is not None:
            pos_anchors_np = self.tensor_to_numpy(positive_anchors)
            # Limit the number of positive anchors to visualize
            for i, box in enumerate(pos_anchors_np[:max_anchors]):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
        # Save the image
        img_dir = os.path.join(self.anchor_dir, f"img_{image_idx}")
        os.makedirs(img_dir, exist_ok=True)
        
        img_pil.save(os.path.join(img_dir, f"iter_{iteration}.png"))
        
        return img_pil
    
    def visualize_roi_predictions(self, image, rpn_proposals, final_boxes, scores, image_idx, iteration):
        """
        Visualize the difference between RPN proposals and final predictions
        
        Args:
            image: Original image tensor
            rpn_proposals: Proposals from RPN
            final_boxes: Final bounding box predictions
            scores: Classification scores
            image_idx: Index of the image in the batch
            iteration: Current iteration number
        """
        # Convert image tensor to numpy
        image_np = self.tensor_to_numpy(image.permute(1, 2, 0))
        
        # Normalize image for visualization
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create a copy of the image for drawing
        img = (image_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw RPN proposals (blue)
        rpn_proposals_np = self.tensor_to_numpy(rpn_proposals)
        for box in rpn_proposals_np:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=2)
        
        # Draw final boxes (green) with scores
        if final_boxes is not None and scores is not None:
            final_boxes_np = self.tensor_to_numpy(final_boxes)
            scores_np = self.tensor_to_numpy(scores)
            
            for i, (box, score) in enumerate(zip(final_boxes_np, scores_np)):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                
                # Draw score
                score_text = f"{score:.2f}"
                draw.text((x1, y1-15), score_text, fill=(0, 255, 0), font=font)
        
        # Save the image
        img_dir = os.path.join(self.roi_dir, f"img_{image_idx}")
        os.makedirs(img_dir, exist_ok=True)
        
        img_pil.save(os.path.join(img_dir, f"iter_{iteration}.png"))
        
        return img_pil
    
    def create_animation(self, image_dir, output_path, fps=2):
        """
        Create an animation from a sequence of images
        
        Args:
            image_dir: Directory containing the images
            output_path: Path to save the animation
            fps: Frames per second
        """
        images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
        
        if not images:
            print(f"No images found in {image_dir}")
            return
        
        # Read the first image to get dimensions
        frame = cv2.imread(images[0])
        height, width, _ = frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Add each image to the video
        for image_path in images:
            frame = cv2.imread(image_path)
            video.write(frame)
        
        # Release the video writer
        video.release()
        print(f"Animation saved to {output_path}")
    
    def create_all_animations(self, image_indices, iteration_count):
        """
        Create animations for all visualizations
        
        Args:
            image_indices: List of image indices
            iteration_count: Number of iterations
        """
        # Create animations for objectness maps
        for img_idx in image_indices:
            img_dir = os.path.join(self.objectness_dir, f"img_{img_idx}")
            if os.path.exists(img_dir):
                # For each feature level
                level_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
                for level_dir in level_dirs:
                    level_path = os.path.join(img_dir, level_dir)
                    output_path = os.path.join(self.objectness_dir, f"img_{img_idx}_{level_dir}_animation.mp4")
                    self.create_animation(level_path, output_path)
        
        # Create animations for proposals
        for img_idx in image_indices:
            img_dir = os.path.join(self.proposals_dir, f"img_{img_idx}")
            if os.path.exists(img_dir):
                output_path = os.path.join(self.proposals_dir, f"img_{img_idx}_animation.mp4")
                self.create_animation(img_dir, output_path)
        
        # Create animations for anchor assignments
        for img_idx in image_indices:
            img_dir = os.path.join(self.anchor_dir, f"img_{img_idx}")
            if os.path.exists(img_dir):
                output_path = os.path.join(self.anchor_dir, f"img_{img_idx}_animation.mp4")
                self.create_animation(img_dir, output_path)
        
        # Create animations for ROI predictions
        for img_idx in image_indices:
            img_dir = os.path.join(self.roi_dir, f"img_{img_idx}")
            if os.path.exists(img_dir):
                output_path = os.path.join(self.roi_dir, f"img_{img_idx}_animation.mp4")
                self.create_animation(img_dir, output_path) 