#!/usr/bin/env python3
"""
Final Comparative Analysis: Adversarial Training Results
Comprehensive comparison of all models developed
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results():
    """Load all evaluation results."""
    results_dir = Path('results/robustness_evaluation')
    
    # Find the latest results files
    files = list(results_dir.glob('*.json'))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("Available result files:")
    for i, file in enumerate(files):
        print(f"{i}: {file.name}")
    
    if len(files) >= 2:
        # Load latest two files (our model and friend's model)
        with open(files[0], 'r') as f:
            friend_results = json.load(f)
        with open(files[1], 'r') as f:
            our_results = json.load(f)
            
        return our_results, friend_results
    else:
        print("Need at least 2 result files for comparison")
        return None, None

def create_comparison_summary():
    """Create comprehensive comparison summary."""
    
    our_results, friend_results = load_results()
    
    if our_results is None:
        print("Could not load results for comparison")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ADVERSARIAL TRAINING ANALYSIS")
    print("="*80)
    
    # Model identification
    our_model = "Enhanced Adversarial Model (Ours)"
    friend_model = "Friend's Robust Model"
    
    print(f"\n📊 CLEAN PERFORMANCE COMPARISON")
    print("-" * 40)
    
    our_clean = our_results['clean']
    friend_clean = friend_results['clean']
    
    print(f"{our_model:<35} | {friend_model}")
    print("-" * 75)
    print(f"F1 Score:     {our_clean['f1']:.4f}                | {friend_clean['f1']:.4f}")
    print(f"Precision:    {our_clean['precision']:.4f}                | {friend_clean['precision']:.4f}")
    print(f"Recall:       {our_clean['recall']:.4f}                | {friend_clean['recall']:.4f}")
    print(f"Accuracy:     {our_clean['accuracy']:.4f}                | {friend_clean['accuracy']:.4f}")
    
    # Winner analysis
    our_better_clean = our_clean['f1'] > friend_clean['f1']
    clean_diff = abs(our_clean['f1'] - friend_clean['f1'])
    winner_clean = our_model if our_better_clean else friend_model
    print(f"\n🏆 Clean Performance Winner: {winner_clean} (+{clean_diff:.4f} F1)")
    
    print(f"\n⚔️  ADVERSARIAL ROBUSTNESS COMPARISON")
    print("-" * 50)
    
    # Extract key attack results
    attacks_to_compare = ['fgsm_eps_0.001', 'fgsm_eps_0.01', 'pgd_eps_0.001', 'pgd_eps_0.01', 'cw']
    
    print(f"{'Attack':<15} | {'Our F1':<8} | {'Friend F1':<8} | {'Our Drop':<8} | {'Friend Drop':<8} | {'Winner'}")
    print("-" * 85)
    
    robustness_wins = {'ours': 0, 'friend': 0}
    total_our_drop = 0
    total_friend_drop = 0
    
    for attack in attacks_to_compare:
        if attack in our_results['adversarial'] and attack in friend_results['adversarial']:
            our_f1 = our_results['adversarial'][attack]['f1']
            friend_f1 = friend_results['adversarial'][attack]['f1']
            our_drop = our_clean['f1'] - our_f1
            friend_drop = friend_clean['f1'] - friend_f1
            
            winner = "Ours" if our_f1 > friend_f1 else "Friend"
            if our_f1 > friend_f1:
                robustness_wins['ours'] += 1
            else:
                robustness_wins['friend'] += 1
                
            total_our_drop += our_drop
            total_friend_drop += friend_drop
            
            print(f"{attack:<15} | {our_f1:<8.4f} | {friend_f1:<8.4f} | {our_drop:<8.4f} | {friend_drop:<8.4f} | {winner}")
    
    avg_our_drop = total_our_drop / len(attacks_to_compare)
    avg_friend_drop = total_friend_drop / len(attacks_to_compare)
    
    print(f"\n📈 ROBUSTNESS SUMMARY")
    print("-" * 30)
    print(f"Average F1 Drop (Ours):    {avg_our_drop:.4f}")
    print(f"Average F1 Drop (Friend):  {avg_friend_drop:.4f}")
    print(f"Robustness Winner:         {'Friend' if avg_friend_drop < avg_our_drop else 'Ours'}")
    print(f"Attack Wins - Ours: {robustness_wins['ours']}, Friend: {robustness_wins['friend']}")
    
    print(f"\n🎯 OVERALL ASSESSMENT")
    print("-" * 25)
    
    # Calculate overall scores
    our_overall_score = our_clean['f1'] - avg_our_drop
    friend_overall_score = friend_clean['f1'] - avg_friend_drop
    
    print(f"Overall Score (Clean F1 - Avg Drop):")
    print(f"  Ours:   {our_clean['f1']:.4f} - {avg_our_drop:.4f} = {our_overall_score:.4f}")
    print(f"  Friend: {friend_clean['f1']:.4f} - {avg_friend_drop:.4f} = {friend_overall_score:.4f}")
    
    overall_winner = "Our Enhanced Adversarial Model" if our_overall_score > friend_overall_score else "Friend's Robust Model"
    print(f"\n🏆 OVERALL WINNER: {overall_winner}")
    
    print(f"\n📋 KEY INSIGHTS")
    print("-" * 20)
    
    if our_better_clean:
        print(f"✅ Our model achieves superior clean performance (+{clean_diff:.4f} F1)")
    else:
        print(f"❌ Friend's model has better clean performance (+{clean_diff:.4f} F1)")
    
    if avg_our_drop < avg_friend_drop:
        print(f"✅ Our model shows better adversarial robustness (smaller F1 drop)")
    else:
        print(f"❌ Friend's model shows better adversarial robustness (smaller F1 drop)")
        
    print(f"\n🔬 TECHNICAL ANALYSIS")
    print("-" * 25)
    print("Training Approach Comparison:")
    print("- Our Model: Enhanced baseline (50 epochs) + Adversarial training (14 epochs)")
    print("- Friend's Model: Traditional adversarial training from scratch")
    print()
    print("Architecture: Both use identical BaselineIDS (80→256→128→64→1)")
    print("Training Data: Identical PFCP dataset with same preprocessing")
    print()
    
    # Success rate analysis
    print("Attack Success Rate Analysis:")
    for attack in ['fgsm_eps_0.001', 'pgd_eps_0.001']:
        if attack in our_results['adversarial'] and attack in friend_results['adversarial']:
            our_success = our_results['adversarial'][attack]['attack_success_rate']
            friend_success = friend_results['adversarial'][attack]['attack_success_rate']
            print(f"  {attack}: Ours={our_success:.4f}, Friend={friend_success:.4f}")
    
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 20)
    
    if our_overall_score > friend_overall_score:
        print("🎉 SUCCESS: Our adversarial training approach is superior!")
        print("✅ Enhanced baseline + adversarial fine-tuning strategy worked")
        print("✅ 50-epoch pre-training provided excellent foundation")
        print("✅ Progressive adversarial training preserved clean performance")
    else:
        print("📈 LEARNING: Friend's approach shows advantages in specific areas")
        print("🔍 Consider deeper adversarial training or different attack strategies")
        print("🔍 Investigate friend's training methodology for robustness gains")
    
    print(f"\n🎯 FINAL METRICS SUMMARY")
    print("-" * 30)
    print(f"Model Performance Ranking:")
    if our_overall_score > friend_overall_score:
        print(f"1. Our Enhanced Adversarial Model (Score: {our_overall_score:.4f})")
        print(f"2. Friend's Robust Model (Score: {friend_overall_score:.4f})")
    else:
        print(f"1. Friend's Robust Model (Score: {friend_overall_score:.4f})")
        print(f"2. Our Enhanced Adversarial Model (Score: {our_overall_score:.4f})")
    
    print(f"\nClean Performance: {winner_clean}")
    print(f"Adversarial Robustness: {'Friend' if avg_friend_drop < avg_our_drop else 'Ours'}")
    print(f"Best Balanced Approach: {overall_winner}")

if __name__ == "__main__":
    create_comparison_summary()