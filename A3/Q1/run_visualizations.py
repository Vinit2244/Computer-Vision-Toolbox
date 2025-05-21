import os
import argparse
import yaml
import subprocess
from tqdm import tqdm

def create_config_with_hyperparams(base_config_path, output_config_path, hyperparams):
    """
    Create a new config file with modified hyperparameters
    
    Args:
        base_config_path: Path to the base config file
        output_config_path: Path to save the new config file
        hyperparams: Dictionary of hyperparameters to modify
    """
    # Read the base config
    with open(base_config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Update hyperparameters
    for key, value in hyperparams.items():
        if key in config['model_params']:
            config['model_params'][key] = value
    
    # Save the new config
    with open(output_config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
    return output_config_path

def run_training_with_hyperparams(base_config_path, hyperparams_sets, vis_interval=50, num_epochs=5):
    """
    Run training with different hyperparameter sets
    
    Args:
        base_config_path: Path to the base config file
        hyperparams_sets: List of dictionaries with hyperparameters to modify
        vis_interval: Interval for visualization (in steps)
        num_epochs: Number of epochs to train for
    """
    # Create configs directory if it doesn't exist
    configs_dir = "configs_hyperparams"
    os.makedirs(configs_dir, exist_ok=True)
    
    # Run training for each hyperparameter set
    for i, hyperparams in enumerate(hyperparams_sets):
        print(f"Running training with hyperparameter set {i+1}/{len(hyperparams_sets)}")
        print(f"Hyperparameters: {hyperparams}")
        
        # Create a new config file with the hyperparameters
        config_name = f"config_set_{i+1}.yaml"
        config_path = os.path.join(configs_dir, config_name)
        create_config_with_hyperparams(base_config_path, config_path, hyperparams)
        
        # Modify the number of epochs in the config
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config['train_params']['num_epochs'] = num_epochs
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        # Run the training script
        cmd = [
            "python", "train_with_visualization.py",
            "--config", config_path,
            "--vis-interval", str(vis_interval)
        ]
        
        subprocess.run(cmd)
        
        print(f"Finished training with hyperparameter set {i+1}/{len(hyperparams_sets)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Faster RCNN training with different hyperparameter sets')
    parser.add_argument('--config', dest='config_path', default='config/st.yaml', type=str,
                        help='Path to the base config file')
    parser.add_argument('--vis-interval', dest='vis_interval', default=50, type=int,
                        help='Interval for visualization (in steps)')
    parser.add_argument('--num-epochs', dest='num_epochs', default=5, type=int,
                        help='Number of epochs to train for')
    args = parser.parse_args()
    
    # Define hyperparameter sets to try
    hyperparams_sets = [
        # Set 1: Default values
        {
            'rpn_fg_threshold': 0.7,
            'rpn_bg_threshold': 0.3,
            'rpn_batch_size': 256,
            'rpn_pos_fraction': 0.5,
            'roi_batch_size': 128,
            'roi_pos_fraction': 0.25
        },
        # Set 2: Higher IoU thresholds
        {
            'rpn_fg_threshold': 0.8,
            'rpn_bg_threshold': 0.4,
            'rpn_batch_size': 256,
            'rpn_pos_fraction': 0.5,
            'roi_batch_size': 128,
            'roi_pos_fraction': 0.25
        },
        # Set 3: Different batch sizes and positive fractions
        {
            'rpn_fg_threshold': 0.7,
            'rpn_bg_threshold': 0.3,
            'rpn_batch_size': 512,
            'rpn_pos_fraction': 0.3,
            'roi_batch_size': 256,
            'roi_pos_fraction': 0.4
        }
    ]
    
    run_training_with_hyperparams(args.config_path, hyperparams_sets, args.vis_interval, args.num_epochs) 