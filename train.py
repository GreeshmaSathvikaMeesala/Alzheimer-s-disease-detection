import os
from utils.data_preprocessing import DataPreprocessor
from utils.model_builder import AlzheimerModel
from utils.model_evaluator import ModelEvaluator
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    preprocessor = DataPreprocessor()
    train_generator, val_generator = preprocessor.create_data_generators('combined_images')
    model = AlzheimerModel(num_classes=4)
    final_model_path = 'models/final_model.h5'
    if os.path.exists(final_model_path):
        print("Loading existing trained model...")
        model.load_model(final_model_path)
    else:
        print("No valid trained model found. Building new model...")
        model.build_model()
        checkpoint_path = 'models/best_model.h5'
        if os.path.exists(checkpoint_path):
            try:
                print("Attempting to load weights from previous checkpoint...")
                model.model.load_weights(checkpoint_path)
                print("Successfully loaded weights from checkpoint")
            except Exception as e:
                print(f"Could not load weights from checkpoint: {str(e)}")
                print("Starting training with fresh weights")
        print("\nStarting model training...")
        history = model.train_model(train_generator, val_generator)
        print("\nSaving final model...")
        model.save_model()
    print("\nEvaluating model...")
    evaluator = ModelEvaluator(model.model)
    metrics = evaluator.evaluate_model(val_generator)
    if metrics:
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nROC AUC Scores:")
        for class_name, score in metrics['roc_auc'].items():
            print(f"{class_name}: {score:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    evaluator.plot_confusion_matrix(val_generator)
    
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main() 