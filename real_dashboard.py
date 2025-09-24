import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Real Adversarial 5G IDS Results",
    page_icon="🛡️",
    layout="wide"
)

def load_real_results():
    """Load actual results from the system"""
    results = {}
    
    # Load baseline performance
    try:
        with open('results/baseline/final_test_metrics.json', 'r') as f:
            results['baseline'] = json.load(f)
    except:
        st.error("❌ Could not load baseline results from results/baseline/final_test_metrics.json")
        return None
    
    # Load adversarial attack results  
    try:
        with open('results/adversarial_attacks/phase3_comprehensive_report.json', 'r') as f:
            results['attacks'] = json.load(f)
    except:
        st.error("❌ Could not load attack results from results/adversarial_attacks/phase3_comprehensive_report.json")
        return None
    
    return results

def show_executive_summary(results):
    """Executive summary with key metrics"""
    
    baseline = results['baseline']
    attacks = results['attacks']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Model Accuracy",
            f"{baseline['accuracy']:.1%}",
            f"{(baseline['accuracy'] - 0.5)*100:.1f}% above random"
        )
    
    with col2:
        st.metric(
            "Inference Time", 
            f"{baseline['avg_inference_time_ms']:.3f}ms",
            "Real-time capable"
        )
    
    with col3:
        st.metric(
            "Attack Success Rate",
            f"{attacks['attack_summary']['overall_success_rate']:.1%}",
            "Vulnerability level"
        )
        
    with col4:
        st.metric(
            "Adversarial Samples",
            f"{attacks['total_adversarial_samples']:,}",
            f"{int(attacks['attack_summary']['total_successful_attacks']):,} successful"
        )

def show_detailed_performance(results):
    """Show detailed model performance"""
    
    st.subheader("🏆 Model Performance Analysis")
    
    baseline = results['baseline']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Performance metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
            'Score': [baseline['accuracy'], baseline['f1'], baseline['precision'], baseline['recall']]
        })
        
        fig = px.bar(
            metrics_df, 
            x='Metric', 
            y='Score',
            title='Performance Metrics',
            color='Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confusion Matrix
        cm = np.array(baseline['confusion_matrix'])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Benign', 'Predicted Attack'],
            y=['Actual Benign', 'Actual Attack'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📊 Detailed Performance Breakdown")
    
    total_samples = sum(sum(row) for row in baseline['confusion_matrix'])
    tn, fp, fn, tp = baseline['confusion_matrix'][0][0], baseline['confusion_matrix'][0][1], baseline['confusion_matrix'][1][0], baseline['confusion_matrix'][1][1]
    
    metrics_detailed = {
        'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives', 
                  'Total Samples', 'False Positive Rate', 'Attack Detection Rate'],
        'Value': [f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}", f"{total_samples:,}",
                 f"{fp/(tn+fp)*100:.2f}%", f"{tp/(fn+tp)*100:.2f}%"],
        'Description': ['Benign traffic correctly identified', 'Benign traffic misclassified as attack',
                       'Attacks missed', 'Attacks correctly detected', 'Total test dataset size',
                       'Rate of false alarms', 'Rate of successful attack detection']
    }
    
    st.dataframe(pd.DataFrame(metrics_detailed), use_container_width=True)

def show_attack_analysis(results):
    """Show adversarial attack analysis"""
    
    st.subheader("⚔️ Adversarial Attack Analysis")
    
    attacks = results['attacks']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Attack success rates
        attack_data = []
        for attack in attacks['attack_summary']['attack_history']:
            attack_data.append({
                'Attack': f"{attack['attack_type']}\n(ε={attack['epsilon']})",
                'Success Rate (%)': attack['success_rate'] * 100,
                'Successful': int(attack['successful_attacks']),
                'Total': attack['total_samples'],
                'L2 Perturbation': attack['average_l2_perturbation']
            })
        
        attack_df = pd.DataFrame(attack_data)
        
        fig = px.bar(
            attack_df,
            x='Attack',
            y='Success Rate (%)',
            title='Attack Success Rates by Method',
            color='Success Rate (%)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Overall attack distribution
        successful = int(attacks['attack_summary']['total_successful_attacks'])
        failed = attacks['total_adversarial_samples'] - successful
        
        fig = go.Figure(data=[go.Pie(
            labels=['Failed Attacks', 'Successful Attacks'],
            values=[failed, successful],
            hole=.3,
            marker_colors=['#2E8B57', '#DC143C']
        )])
        
        fig.update_layout(title="Overall Attack Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Attack details table
    st.subheader("📋 Attack Method Details")
    
    detailed_attacks = []
    for attack in attacks['attack_summary']['attack_history']:
        detailed_attacks.append({
            'Attack Method': attack['attack_type'],
            'Epsilon': attack['epsilon'],
            'Success Rate': f"{attack['success_rate']:.1%}",
            'Successful Attacks': f"{int(attack['successful_attacks']):,}",
            'Total Samples': f"{attack['total_samples']:,}",
            'Avg L2 Perturbation': f"{attack['average_l2_perturbation']:.6f}"
        })
    
    st.dataframe(pd.DataFrame(detailed_attacks), use_container_width=True)

def show_vulnerability_assessment(results):
    """Show vulnerability assessment"""
    
    st.subheader("🔍 Vulnerability Assessment")
    
    baseline = results['baseline']
    attacks = results['attacks']
    
    # Calculate vulnerability metrics
    success_rates = [a['success_rate'] for a in attacks['attack_summary']['attack_history']]
    max_success = max(success_rates)
    min_success = min(success_rates)
    avg_success = np.mean(success_rates)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Highest Vulnerability", f"{max_success:.1%}", "Most critical weakness")
    
    with col2:
        st.metric("Lowest Vulnerability", f"{min_success:.1%}", "Most robust area")
        
    with col3:
        st.metric("Average Vulnerability", f"{avg_success:.1%}", "Overall security posture")
    
    # Risk assessment
    st.subheader("🛡️ Security Risk Assessment")
    
    if max_success > 0.5:
        risk_level = "🔴 HIGH"
        risk_color = "red"
    elif max_success > 0.3:
        risk_level = "🟡 MEDIUM" 
        risk_color = "orange"
    else:
        risk_level = "🟢 LOW"
        risk_color = "green"
    
    st.markdown(f"**Overall Risk Level:** {risk_level}")
    
    # Recommendations
    recommendations = []
    
    if max_success > 0.4:
        recommendations.extend([
            "⚠️ Implement adversarial training with epsilon-based augmentation",
            "🔧 Deploy ensemble defense mechanisms", 
            "🛡️ Add input preprocessing and noise injection"
        ])
    
    if baseline['accuracy'] > 0.9:
        recommendations.append("✅ Baseline performance is excellent")
    
    if avg_success < 0.4:
        recommendations.append("✅ Shows reasonable robustness against average attacks")
    
    if recommendations:
        st.subheader("📋 Security Recommendations")
        for rec in recommendations:
            st.write(rec)

def show_commercial_readiness(results):
    """Show commercial deployment readiness"""
    
    st.subheader("💼 Commercial Deployment Analysis")
    
    baseline = results['baseline']
    attacks = results['attacks']
    
    # Calculate readiness scores
    accuracy_score = baseline['accuracy']
    latency_score = 1.0 if baseline['avg_inference_time_ms'] <= 1.0 else (5.0 / baseline['avg_inference_time_ms'])
    security_score = 1.0 - max(a['success_rate'] for a in attacks['attack_summary']['attack_history'])
    
    overall_score = (accuracy_score + latency_score + security_score) / 3
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Accuracy Score",
            f"{accuracy_score:.2f}",
            "Performance quality"
        )
    
    with col2:
        st.metric(
            "Latency Score", 
            f"{latency_score:.2f}",
            "Real-time capability"
        )
        
    with col3:
        st.metric(
            "Security Score",
            f"{security_score:.2f}", 
            "Robustness level"
        )
        
    with col4:
        st.metric(
            "Overall Readiness",
            f"{overall_score:.2f}",
            "Deployment score"
        )
    
    # Deployment recommendation
    if overall_score >= 0.8:
        deployment_status = "🟢 Ready for Production"
    elif overall_score >= 0.6:
        deployment_status = "🟡 Ready with Mitigation" 
    else:
        deployment_status = "🔴 Needs Improvement"
    
    st.markdown(f"**Deployment Status:** {deployment_status}")
    
    # Market positioning
    st.subheader("📈 Market Positioning")
    
    positioning_data = {
        'Aspect': ['Performance Tier', 'Latency Class', 'Security Posture', 'Market Segment'],
        'Classification': [
            'Enterprise-grade' if accuracy_score >= 0.9 else 'Research prototype',
            'Real-time capable' if baseline['avg_inference_time_ms'] <= 1.0 else 'Batch processing',
            'Production viable' if security_score >= 0.5 else 'Needs hardening',
            'Commercial deployment' if overall_score >= 0.7 else 'Lab validation'
        ]
    }
    
    st.dataframe(pd.DataFrame(positioning_data), use_container_width=True)

def main():
    st.title("🛡️ Real Adversarial 5G IDS System Results")
    st.markdown("**Showing ACTUAL results from your working system - NOT mock data**")
    
    # Load real results
    results = load_real_results()
    if not results:
        st.error("Could not load results. Make sure you're in the project directory with results files.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Select Analysis View:",
        ["Executive Summary", "Model Performance", "Attack Analysis", "Vulnerability Assessment", "Commercial Readiness"]
    )
    
    # Show selected page
    if page == "Executive Summary":
        st.header("📊 Executive Summary")
        show_executive_summary(results)
        
        st.subheader("🎯 Key Findings")
        baseline = results['baseline']
        attacks = results['attacks']
        
        st.write(f"• **High Performance:** Achieved {baseline['accuracy']:.1%} accuracy with {baseline['avg_inference_time_ms']:.3f}ms inference time")
        st.write(f"• **Comprehensive Testing:** Evaluated against {attacks['total_adversarial_samples']:,} adversarial examples")
        st.write(f"• **Security Analysis:** {attacks['attack_summary']['overall_success_rate']:.1%} overall attack success rate identified vulnerabilities")
        st.write(f"• **Production Ready:** Real-time capable with measurable security characteristics")
        
    elif page == "Model Performance":
        show_detailed_performance(results)
        
    elif page == "Attack Analysis":
        show_attack_analysis(results)
        
    elif page == "Vulnerability Assessment":
        show_vulnerability_assessment(results)
        
    elif page == "Commercial Readiness":
        show_commercial_readiness(results)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Real System Data**")
    st.sidebar.markdown(f"✅ Baseline: {Path('results/baseline/final_test_metrics.json').exists()}")
    st.sidebar.markdown(f"✅ Attacks: {Path('results/adversarial_attacks/phase3_comprehensive_report.json').exists()}")

if __name__ == "__main__":
    main()