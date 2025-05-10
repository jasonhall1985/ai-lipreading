#!/usr/bin/env python
# Evaluation script for LipNet models

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from jiwer import wer, cer
import random
import json

# Import model and dataset components
from improved_model import (
    build_improved_lipnet, decode_prediction, 
    CHAR_LIST, CHAR_TO_NUM, NUM_TO_CHAR
)

# Import dataset processing
from datasets.dataset_loader import (
    DatasetConfig, prepare_datasets, LRS2Dataset, LRWDataset
)

def setup_gpu_memory_growth():
    """Set up GPU memory growth to prevent TensorFlow from allocating all GPU memory at once"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled on {len(gpus)} GPUs")
        except RuntimeError as e:
            print(f"Error setting up GPU memory growth: {e}")

def load_model_from_checkpoint(checkpoint_path, img_height=50, img_width=100, max_len=75):
    """Load a model from checkpoint weights"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    # Build model
    model = build_improved_lipnet(
        img_size=(img_height, img_width),
        max_len=max_len
    )
    
    # Load weights
    try:
        model.load_weights(checkpoint_path)
        print(f"Loaded model weights from {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights: {e}")
    
    return model

def evaluate_on_dataset(model, dataset, num_samples=None, output_dir=None):
    """
    Evaluate model on a dataset
    
    Args:
        model: The LipNet model
        dataset: Dataset to evaluate on
        num_samples: Number of samples to evaluate (None for all)
        output_dir: Directory to save results (None for no saving)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not hasattr(dataset, "samples") or len(dataset.samples) == 0:
        print(f"Dataset has no samples")
        return {
            "wer": 1.0,
            "cer": 1.0,
            "num_samples": 0,
            "correct": 0
        }
    
    # Limit number of samples if requested
    samples = dataset.samples
    if num_samples is not None and num_samples < len(samples):
        samples = random.sample(samples, num_samples)
    
    # Store results
    ground_truths = []
    predictions = []
    sample_results = []
    
    # Evaluate each sample
    for sample in tqdm(samples, desc="Evaluating"):
        # Process the sample
        processed_sample = dataset.prepare_sample(sample)
        if processed_sample is None:
            continue
            
        # Make prediction
        try:
            # Add batch dimension if needed
            video = processed_sample['video']
            if len(video.shape) == 3:
                video = np.expand_dims(video, 0)
            elif len(video.shape) == 4:
                video = np.expand_dims(video, 0)
                
            # Get prediction
            prediction = model.predict(video, verbose=0)
            
            # Decode prediction
            text_prediction = decode_prediction(prediction[0])
            
            # Ground truth
            ground_truth = processed_sample['text']
            
            # Store results
            ground_truths.append(ground_truth)
            predictions.append(text_prediction)
            
            # Store sample result
            sample_results.append({
                "id": sample.get("id", ""),
                "ground_truth": ground_truth,
                "prediction": text_prediction,
                "correct": text_prediction.lower() == ground_truth.lower()
            })
            
        except Exception as e:
            print(f"Error evaluating sample {sample.get('id', '')}: {e}")
    
    # Skip if no valid samples
    if len(ground_truths) == 0:
        print("No valid samples for evaluation")
        return {
            "wer": 1.0,
            "cer": 1.0,
            "num_samples": 0,
            "correct": 0
        }
    
    # Calculate metrics
    try:
        word_error_rate = wer(ground_truths, predictions)
    except:
        word_error_rate = 1.0
        
    try:
        char_error_rate = cer(ground_truths, predictions)
    except:
        char_error_rate = 1.0
    
    # Count exact matches
    correct = sum(1 for g, p in zip(ground_truths, predictions) if g.lower() == p.lower())
    accuracy = correct / len(ground_truths) if ground_truths else 0
    
    # Print results
    print(f"Evaluation results on {len(ground_truths)} samples:")
    print(f"Word Error Rate (WER): {word_error_rate:.4f}")
    print(f"Character Error Rate (CER): {char_error_rate:.4f}")
    print(f"Exact Match Accuracy: {accuracy:.4f} ({correct}/{len(ground_truths)})")
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics = {
            "wer": float(word_error_rate),
            "cer": float(char_error_rate),
            "accuracy": float(accuracy),
            "num_samples": len(ground_truths),
            "correct": correct
        }
        
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save sample results
        with open(os.path.join(output_dir, "sample_results.json"), "w") as f:
            json.dump(sample_results, f, indent=2)
        
        # Save some example predictions
        num_examples = min(10, len(ground_truths))
        examples = []
        
        for i in range(num_examples):
            examples.append({
                "ground_truth": ground_truths[i],
                "prediction": predictions[i],
                "correct": ground_truths[i].lower() == predictions[i].lower()
            })
            
        with open(os.path.join(output_dir, "examples.json"), "w") as f:
            json.dump(examples, f, indent=2)
    
    return {
        "wer": word_error_rate,
        "cer": char_error_rate,
        "accuracy": accuracy,
        "num_samples": len(ground_truths),
        "correct": correct
    }

def visualize_predictions(model, dataset, num_samples=5, output_dir=None):
    """
    Visualize some predictions from the model
    
    Args:
        model: The LipNet model
        dataset: Dataset to visualize from
        num_samples: Number of samples to visualize
        output_dir: Directory to save visualizations
    """
    if not hasattr(dataset, "samples") or len(dataset.samples) == 0:
        print(f"Dataset has no samples")
        return
    
    # Randomly select samples
    samples = random.sample(dataset.samples, min(num_samples, len(dataset.samples)))
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample
    for i, sample in enumerate(samples):
        # Get video path
        video_path = sample['video_path']
        ground_truth = sample['text']
        
        # Process the sample
        processed_sample = dataset.prepare_sample(sample)
        if processed_sample is None:
            continue
            
        # Make prediction
        try:
            # Add batch dimension if needed
            video = processed_sample['video']
            if len(video.shape) == 3:
                video = np.expand_dims(video, 0)
            elif len(video.shape) == 4:
                video = np.expand_dims(video, 0)
                
            # Get prediction
            prediction = model.predict(video, verbose=0)
            
            # Decode prediction
            text_prediction = decode_prediction(prediction[0])
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original video frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[0].imshow(frame)
                axes[0].set_title("Original Video")
                axes[0].axis("off")
            cap.release()
            
            # Processed mouth region
            mouth_frame = video[0, 0, :, :, 0]  # First frame
            axes[1].imshow(mouth_frame, cmap='gray')
            axes[1].set_title("Processed Mouth Region")
            axes[1].axis("off")
            
            # Text results
            axes[2].text(0.1, 0.7, f"Ground Truth: {ground_truth}", fontsize=12)
            axes[2].text(0.1, 0.5, f"Prediction: {text_prediction}", fontsize=12)
            correct = ground_truth.lower() == text_prediction.lower()
            color = "green" if correct else "red"
            axes[2].text(0.1, 0.3, f"Correct: {correct}", fontsize=12, color=color)
            axes[2].axis("off")
            
            # Set title
            fig.suptitle(f"Sample {i+1}: {os.path.basename(video_path)}")
            
            # Save or show
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Error visualizing sample {sample.get('id', '')}: {e}")

def main():
    """Main function for evaluating LipNet models"""
    parser = argparse.ArgumentParser(description="Evaluate LipNet models")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint file")
    parser.add_argument("--img-height", type=int, default=50,
                       help="Image height for model input")
    parser.add_argument("--img-width", type=int, default=100,
                       help="Image width for model input")
    parser.add_argument("--sequence-length", type=int, default=75,
                       help="Sequence length for model input")
    
    # Dataset parameters
    parser.add_argument("--grid-path", type=str, default=None,
                       help="Path to GRID corpus dataset")
    parser.add_argument("--lrs2-path", type=str, default=None,
                       help="Path to LRS2 dataset")
    parser.add_argument("--lrw-path", type=str, default=None,
                       help="Path to LRW dataset")
    parser.add_argument("--split", type=str, default="val",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to evaluate (None for all)")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save evaluation results")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize some predictions")
    parser.add_argument("--visualize-samples", type=int, default=5,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # Check that at least one dataset is provided
    if not args.grid_path and not args.lrs2_path and not args.lrw_path:
        parser.error("At least one dataset path must be provided")
    
    # Set up GPU memory growth
    setup_gpu_memory_growth()
    
    # Create dataset configuration
    config = DatasetConfig()
    config.img_height = args.img_height
    config.img_width = args.img_width
    config.max_seq_length = args.sequence_length
    
    # Load model
    try:
        model = load_model_from_checkpoint(
            args.checkpoint,
            img_height=args.img_height,
            img_width=args.img_width,
            max_len=args.sequence_length
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare datasets
    datasets = {}
    
    if args.grid_path:
        # Import function to load GRID dataset
        try:
            from train_improved import load_grid_corpus
            grid_data = load_grid_corpus(args.grid_path)
            if args.split == "train":
                datasets["grid"] = grid_data[0]  # train set
            else:
                datasets["grid"] = grid_data[1]  # val set
        except Exception as e:
            print(f"Error loading GRID dataset: {e}")
    
    if args.lrs2_path:
        try:
            lrs2_dataset = LRS2Dataset(config, args.lrs2_path, split=args.split)
            datasets["lrs2"] = lrs2_dataset
        except Exception as e:
            print(f"Error loading LRS2 dataset: {e}")
    
    if args.lrw_path:
        try:
            lrw_dataset = LRWDataset(config, args.lrw_path, split=args.split)
            datasets["lrw"] = lrw_dataset
        except Exception as e:
            print(f"Error loading LRW dataset: {e}")
    
    # Evaluate on each dataset
    results = {}
    
    for name, dataset in datasets.items():
        print(f"\nEvaluating on {name} dataset:")
        output_dir = os.path.join(args.output_dir, name) if args.output_dir else None
        
        results[name] = evaluate_on_dataset(
            model, dataset, num_samples=args.num_samples, output_dir=output_dir
        )
        
        # Visualize if requested
        if args.visualize:
            viz_dir = os.path.join(output_dir, "visualizations") if output_dir else None
            visualize_predictions(
                model, dataset, num_samples=args.visualize_samples, output_dir=viz_dir
            )
    
    # Print overall results
    if results:
        print("\nOverall results:")
        
        # Calculate weighted average of metrics
        total_samples = sum(r["num_samples"] for r in results.values())
        avg_wer = sum(r["wer"] * r["num_samples"] for r in results.values()) / total_samples if total_samples else 1.0
        avg_cer = sum(r["cer"] * r["num_samples"] for r in results.values()) / total_samples if total_samples else 1.0
        total_correct = sum(r["correct"] for r in results.values())
        avg_accuracy = total_correct / total_samples if total_samples else 0.0
        
        print(f"Total samples evaluated: {total_samples}")
        print(f"Average Word Error Rate (WER): {avg_wer:.4f}")
        print(f"Average Character Error Rate (CER): {avg_cer:.4f}")
        print(f"Overall Accuracy: {avg_accuracy:.4f} ({total_correct}/{total_samples})")
        
        # Save overall metrics if output directory is provided
        if args.output_dir:
            overall_metrics = {
                "total_samples": total_samples,
                "avg_wer": float(avg_wer),
                "avg_cer": float(avg_cer),
                "avg_accuracy": float(avg_accuracy),
                "total_correct": total_correct,
                "dataset_results": {name: result for name, result in results.items()}
            }
            
            with open(os.path.join(args.output_dir, "overall_metrics.json"), "w") as f:
                json.dump(overall_metrics, f, indent=2)

if __name__ == "__main__":
    main() 