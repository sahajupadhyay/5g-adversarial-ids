#!/bin/bash

# Execute Advanced Features Setup
# ==============================

echo "🚀 Setting up Advanced Adversarial 5G IDS Features..."
echo "=============================================="

# Activate virtual environment
source ../adv5g/bin/activate

echo "📦 Installing advanced dependencies..."

# Install explainability libraries
pip install shap==0.42.1
pip install lime==0.2.0.1

# Install real-time processing libraries  
pip install asyncio
pip install websockets==11.0

# Install visualization libraries
pip install plotly==5.17.0
pip install streamlit==1.28.0

# Install additional ML libraries
pip install scikit-learn==1.3.0

echo "✅ Dependencies installed successfully!"

echo ""
echo "🎯 Available Advanced Features:"
echo "================================"
echo "1. Advanced Defenses (AWP, TRADES, Ensemble)"
echo "2. Real-time 5G Integration"  
echo "3. Explainable AI Dashboard"
echo ""

echo "🔧 Quick Start Commands:"
echo "========================"
echo ""
echo "1. Run Explainable Dashboard:"
echo "   streamlit run explainable_dashboard.py"
echo ""
echo "2. Test Real-time Integration:"
echo "   python -c \"from src.real_time_5g_integration import main; import asyncio; asyncio.run(main())\""
echo ""
echo "3. Train with Advanced Defenses:"
echo "   python -c \"from src.advanced_defenses import create_defense_config; print('Defense config ready!')\""
echo ""

echo "📊 Dashboard will be available at: http://localhost:8501"
echo ""
echo "✨ Advanced features setup complete!"
echo "Ready for next-generation 5G security research!"