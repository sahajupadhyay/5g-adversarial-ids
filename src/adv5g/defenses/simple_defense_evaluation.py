"""
Simple Defense Evaluation for Phase 2B
Compares baseline vs robust models using noise-based robustness testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
from datetime import datetime


def load_models():
    """Load baseline and robust models for comparison"""
    models = {}
    
    # Load baseline model and its transformers
    baseline_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/rf_advanced.joblib'
    transformer_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/feature_transformers.joblib'
    scaler_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/scaler_advanced.joblib'
    
    if os.path.exists(baseline_path):
        baseline_model = joblib.load(baseline_path)
        transformers = joblib.load(transformer_path) if os.path.exists(transformer_path) else None
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        models['Baseline_RF'] = {
            'model': baseline_model,
            'transformers': transformers,
            'scaler': scaler
        }
        print("✅ Loaded baseline model with transformers")
    
    # Load robust model (doesn't need transformers since it uses full 43 features)
    robust_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/simple_robust_rf.joblib'
    if os.path.exists(robust_path):
        robust_model = joblib.load(robust_path)
        models['Robust_RF'] = {
            'model': robust_model,
            'transformers': None,
            'scaler': None
        }
        print("✅ Loaded robust model")
    
    return models


def load_test_data():
    """Load test data"""
    data_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/data/processed'
    
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))
    
    print(f"✅ Loaded test data: {X_test.shape}")
    return X_test, y_test


def apply_feature_transformation(X, transformers, scaler):
    """Apply feature transformation for baseline model"""
    if transformers is None or scaler is None:
        return X
    
    # Apply the feature transformation pipeline step by step
    X_transformed = X
    
    # transformers is a tuple of (VarianceThreshold, SelectKBest, PCA)
    for transformer in transformers:
        X_transformed = transformer.transform(X_transformed)
    
    # Apply scaling
    X_scaled = scaler.transform(X_transformed)
    
    return X_scaled


def generate_noisy_examples(X, noise_level=0.1):
    """Generate noisy test examples"""
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    
    # Clip to reasonable bounds
    X_noisy = np.clip(X_noisy, -3.0, 3.0)
    return X_noisy


def evaluate_model_robustness(model_info, X_test, y_test, model_name):
    """Evaluate a single model's robustness"""
    print(f"\n=== Evaluating {model_name} ===")
    
    model = model_info['model']
    transformers = model_info.get('transformers')
    scaler = model_info.get('scaler')
    
    # Apply transformations if needed (for baseline model)
    if transformers is not None and scaler is not None:
        X_test_processed = apply_feature_transformation(X_test, transformers, scaler)
        print(f"Applied feature transformation: {X_test.shape} -> {X_test_processed.shape}")
    else:
        X_test_processed = X_test
        print(f"Using full features: {X_test_processed.shape}")
    
    # Clean performance
    clean_pred = model.predict(X_test_processed)
    clean_accuracy = accuracy_score(y_test, clean_pred)
    print(f"Clean Accuracy: {clean_accuracy:.3f}")
    
    # Test against different noise levels
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    robustness_results = {'clean': clean_accuracy}
    
    for noise_level in noise_levels:
        # Generate noisy version
        X_noisy = generate_noisy_examples(X_test, noise_level)
        
        # Apply same transformations to noisy data
        if transformers is not None and scaler is not None:
            X_noisy_processed = apply_feature_transformation(X_noisy, transformers, scaler)
        else:
            X_noisy_processed = X_noisy
        
        noisy_pred = model.predict(X_noisy_processed)
        noisy_accuracy = accuracy_score(y_test, noisy_pred)
        
        accuracy_drop = clean_accuracy - noisy_accuracy
        relative_drop = accuracy_drop / clean_accuracy if clean_accuracy > 0 else 0
        
        robustness_results[f'noise_{noise_level}'] = {
            'accuracy': noisy_accuracy,
            'drop': accuracy_drop,
            'relative_drop': relative_drop
        }
        
        print(f"  Noise {noise_level}: {noisy_accuracy:.3f} (drop: {relative_drop:.1%})")
    
    # Calculate average robustness
    avg_robust_accuracy = np.mean([
        robustness_results[f'noise_{noise}']['accuracy'] 
        for noise in noise_levels
    ])
    
    robustness_results['avg_robustness'] = avg_robust_accuracy
    print(f"Average Robustness: {avg_robust_accuracy:.3f}")
    
    return robustness_results


def compare_models(baseline_results, robust_results):
    """Compare baseline vs robust model performance"""
    print("\n" + "="*60)
    print("DEFENSE EVALUATION SUMMARY")
    print("="*60)
    
    # Clean performance comparison
    baseline_clean = baseline_results['clean']
    robust_clean = robust_results['clean']
    clean_improvement = robust_clean - baseline_clean
    
    print(f"\n📊 CLEAN PERFORMANCE")
    print(f"Baseline RF:     {baseline_clean:.3f}")
    print(f"Robust RF:       {robust_clean:.3f}")
    print(f"Improvement:     {clean_improvement:+.3f} ({clean_improvement/baseline_clean:.1%})")
    
    # Robustness comparison
    baseline_robust = baseline_results['avg_robustness']
    robust_robust = robust_results['avg_robustness']
    robustness_improvement = robust_robust - baseline_robust
    
    print(f"\n🛡️ ROBUSTNESS PERFORMANCE")
    print(f"Baseline RF:     {baseline_robust:.3f}")
    print(f"Robust RF:       {robust_robust:.3f}")
    print(f"Improvement:     {robustness_improvement:+.3f} ({robustness_improvement/baseline_robust:.1%})")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    
    if robustness_improvement > 0.02:  # 2% improvement threshold
        print("✅ SUCCESS: Robust model shows significant robustness improvement!")
        defense_success = True
    elif robustness_improvement > 0:
        print("⚠️ PARTIAL: Robust model shows some robustness improvement.")
        defense_success = True
    else:
        print("❌ FAILED: Robust model does not improve robustness.")
        defense_success = False
    
    if clean_improvement > -0.05:  # Allow small clean accuracy drop
        print("✅ MAINTAINED: Clean accuracy well preserved.")
    else:
        print("⚠️ WARNING: Significant clean accuracy drop.")
    
    # Calculate defense effectiveness score
    # Balance robustness improvement vs clean accuracy preservation
    defense_score = 0.6 * robustness_improvement + 0.4 * max(0, clean_improvement)
    
    print(f"\n🏆 DEFENSE EFFECTIVENESS SCORE: {defense_score:.3f}")
    
    if defense_score > 0.02:
        print("🎉 EXCELLENT: Strong defense implementation!")
    elif defense_score > 0.01:
        print("👍 GOOD: Effective defense with good trade-offs.")
    elif defense_score > 0:
        print("📈 FAIR: Some improvement, but could be better.")
    else:
        print("📉 POOR: Defense needs improvement.")
    
    return {
        'defense_success': defense_success,
        'defense_score': defense_score,
        'clean_improvement': clean_improvement,
        'robustness_improvement': robustness_improvement,
        'baseline_results': baseline_results,
        'robust_results': robust_results
    }


def create_comparison_visualization(comparison_results, save_dir='reports'):
    """Create visualization comparing models"""
    os.makedirs(save_dir, exist_ok=True)
    
    baseline_results = comparison_results['baseline_results']
    robust_results = comparison_results['robust_results']
    
    # Extract data for plotting
    noise_levels = [0.05, 0.1, 0.2, 0.3, 0.5]
    baseline_accuracies = [baseline_results['clean']] + [
        baseline_results[f'noise_{noise}']['accuracy'] for noise in noise_levels
    ]
    robust_accuracies = [robust_results['clean']] + [
        robust_results[f'noise_{noise}']['accuracy'] for noise in noise_levels
    ]
    
    x_labels = ['Clean'] + [f'Noise {noise}' for noise in noise_levels]
    x_pos = np.arange(len(x_labels))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    width = 0.35
    plt.bar(x_pos - width/2, baseline_accuracies, width, 
            label='Baseline RF', alpha=0.8, color='red')
    plt.bar(x_pos + width/2, robust_accuracies, width,
            label='Robust RF', alpha=0.8, color='green')
    
    plt.xlabel('Test Condition')
    plt.ylabel('Accuracy')
    plt.title('Baseline vs Robust Model Performance')
    plt.xticks(x_pos, x_labels, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add performance annotations
    for i, (baseline_acc, robust_acc) in enumerate(zip(baseline_accuracies, robust_accuracies)):
        improvement = robust_acc - baseline_acc
        if improvement > 0.01:
            plt.annotate(f'+{improvement:.2f}', 
                        xy=(i, max(baseline_acc, robust_acc) + 0.02),
                        ha='center', color='green', fontweight='bold')
        elif improvement < -0.01:
            plt.annotate(f'{improvement:.2f}', 
                        xy=(i, max(baseline_acc, robust_acc) + 0.02),
                        ha='center', color='red', fontweight='bold')
    
    save_path = os.path.join(save_dir, 'defense_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison chart saved to: {save_path}")


def save_evaluation_report(comparison_results, save_dir='reports'):
    """Save evaluation report"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed results as JSON
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_type': 'Phase2B_Defense_Evaluation',
        'summary': {
            'defense_success': comparison_results['defense_success'],
            'defense_score': comparison_results['defense_score'],
            'clean_improvement': comparison_results['clean_improvement'],
            'robustness_improvement': comparison_results['robustness_improvement']
        },
        'detailed_results': {
            'baseline': comparison_results['baseline_results'],
            'robust': comparison_results['robust_results']
        }
    }
    
    json_path = os.path.join(save_dir, 'phase2b_defense_evaluation.json')
    with open(json_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Save summary as markdown
    md_path = os.path.join(save_dir, 'phase2b_defense_summary.md')
    with open(md_path, 'w') as f:
        f.write("# Phase 2B: Defense Evaluation Summary\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Comparison\n\n")
        f.write("| Metric | Baseline RF | Robust RF | Improvement |\n")
        f.write("|--------|-------------|-----------|-------------|\n")
        
        baseline_clean = comparison_results['baseline_results']['clean']
        robust_clean = comparison_results['robust_results']['clean']
        baseline_robust = comparison_results['baseline_results']['avg_robustness']
        robust_robust = comparison_results['robust_results']['avg_robustness']
        
        f.write(f"| Clean Accuracy | {baseline_clean:.3f} | {robust_clean:.3f} | {robust_clean-baseline_clean:+.3f} |\n")
        f.write(f"| Avg Robustness | {baseline_robust:.3f} | {robust_robust:.3f} | {robust_robust-baseline_robust:+.3f} |\n")
        
        f.write(f"\n## Overall Assessment\n\n")
        f.write(f"- **Defense Success**: {'✅ Yes' if comparison_results['defense_success'] else '❌ No'}\n")
        f.write(f"- **Defense Score**: {comparison_results['defense_score']:.3f}\n")
        f.write(f"- **Robustness Improvement**: {comparison_results['robustness_improvement']:.3f}\n")
    
    print(f"📄 Report saved to: {json_path}")
    print(f"📄 Summary saved to: {md_path}")


def main():
    """Main evaluation pipeline"""
    print("🛡️ PHASE 2B: DEFENSE EVALUATION")
    print("="*50)
    
    # Load models and data
    models = load_models()
    X_test, y_test = load_test_data()
    
    if len(models) < 2:
        print("❌ ERROR: Need both baseline and robust models for comparison!")
        return
    
    # Evaluate both models
    baseline_results = evaluate_model_robustness(
        models['Baseline_RF'], X_test, y_test, 'Baseline_RF'
    )
    
    robust_results = evaluate_model_robustness(
        models['Robust_RF'], X_test, y_test, 'Robust_RF'
    )
    
    # Compare results
    comparison_results = compare_models(baseline_results, robust_results)
    
    # Create visualizations
    create_comparison_visualization(comparison_results)
    
    # Save report
    save_evaluation_report(comparison_results)
    
    print("\n🎯 PHASE 2B EVALUATION COMPLETE!")


if __name__ == "__main__":
    main()
