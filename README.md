# 5G Adversarial IDS - Deep Learning Implementation

## 🎯 Project Overview

Industrial-grade deep learning implementation of an adversarially robust 5G Intrusion Detection System following the **Red Team vs Blue Team** methodology for systematic security validation.

## 🏗️ Architecture Philosophy

- **Zero-Trust AI Security**: Models are vulnerable until empirically proven robust
- **Adversarial Training**: On-the-fly adversarial example generation during training
- **Production-Ready**: Enterprise-grade tooling with CLI and dashboard interfaces
- **Systematic Validation**: Red team attacks followed by blue team hardening

## 📊 Dataset Information

- **Source**: Combined 5G network flow and PFCP protocol datasets
- **Total Samples**: ~388K (293K network flow + 95K PFCP protocol)
- **Features**: ~70 carefully selected robust features
- **Class Distribution**: Severely imbalanced (~3% malicious vs 97% normal)
- **Evaluation Metric**: F1-score (critical for imbalanced data)

## 🚀 Implementation Phases

### Phase 1: Foundation & Data Pipeline ✅
- Robust data preprocessing with stratified splits
- Feature standardization and schema management
- Data quality validation framework

### Phase 2: Blue Team - Baseline Defense 🔄
- Deep MLP architecture (256→128→64→1)
- Weighted loss function for class imbalance
- Baseline performance on clean data

### Phase 3: Red Team - Adversarial Assault 📋
- FGSM and PGD attack implementations
- Systematic vulnerability assessment
- Adversarial dataset generation

### Phase 4: Blue Team - Hardened Defense 📋
- Adversarial training with on-the-fly attack generation
- Robust model with improved adversarial resilience
- Robustness-accuracy trade-off optimization

### Phase 5: Operationalization & Deployment 📋
- Professional CLI tools for engineers
- Streamlit dashboard for analysts
- Production inference pipeline

## 🛡️ Security Features

- **Gradient-Based Attacks**: True FGSM and PGD implementations
- **Adversarial Training**: Combined clean + adversarial training batches
- **Robust Architecture**: Designed for adversarial resilience
- **Input Validation**: Comprehensive data sanitization
- **Security Scanning**: Automated vulnerability assessment

## 🔧 Technical Stack

- **Deep Learning**: PyTorch with CUDA support
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Adversarial ML**: Custom implementations + Cleverhans
- **Web Interface**: Streamlit dashboard
- **CLI Tools**: Click/Typer with rich formatting
- **Testing**: Comprehensive pytest suite
- **Deployment**: Docker containerization

## 📈 Performance Targets

- **Baseline Model**: F1-score > 0.85 on clean data
- **Attack Success**: Demonstrate >50% vulnerability
- **Robust Model**: >80% defense effectiveness
- **Clean Performance**: Maintain >80% F1-score after hardening

## 🚦 Getting Started

### Prerequisites
```bash
python >= 3.8
cuda >= 11.8 (for GPU acceleration)
```

### Installation
```bash
git clone <repository-url>
cd 5G_ADVERSARIAL_IDS_DEEP_LEARNING
pip install -r requirements.txt
```

### Quick Start
```bash
# Phase 1: Process data
python src/data/preprocessing.py --config config/data_config.py

# Phase 2: Train baseline model
python src/training/baseline_trainer.py --config config/training_config.py

# Phase 3: Generate adversarial attacks
python src/attacks/pgd.py --model models/baseline/baseline_model.pt

# Phase 4: Train robust model
python src/training/robust_trainer.py --config config/training_config.py

# Phase 5: Launch dashboard
streamlit run src/deployment/dashboard.py
```

## 📚 Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Security Analysis](docs/SECURITY_ANALYSIS.md)
- [Phase Execution Log](PHASE_EXECUTION_LOG.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/end_to_end/

# Coverage report
pytest --cov=src tests/
```

## 📊 Project Status

Current Phase: **Phase 1 - Foundation & Data Pipeline**

See [PHASE_EXECUTION_LOG.md](PHASE_EXECUTION_LOG.md) for detailed progress tracking.

## 🤝 Contributing

1. Follow industrial-grade coding standards
2. Comprehensive error handling with typed exceptions
3. Complete docstrings and type hints
4. Security-first implementation approach
5. Performance optimization mindset

## 📄 License

[Specify License]

## 👥 Team

- **Project Lead**: Sahaj Upadhyay
- **Methodology**: Based on teammate's research methodology

## 📞 Support

For technical support or questions about implementation details, please refer to the comprehensive documentation in the `docs/` directory.

---

**Note**: This project implements production-grade adversarial ML security. All code follows enterprise standards for robustness, security, and maintainability.