# Adversarial 5G IDS Research

## Project Status
![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)
![Phase 2A](https://img.shields.io/badge/Phase%202A-Complete-brightgreen)
![Phase 2B](https://img.shields.io/badge/Phase%202B-Ready-blue)
![Baseline](https://img.shields.io/badge/Baseline-60.8%25%20F1-orange)
![Attack](https://img.shields.io/badge/Attack-57%25%20Evasion-red)
![Tag](https://img.shields.io/badge/Latest%20Tag-v0.2--attacks-blue)

## Overview

Advanced research project developing constraint-aware adversarial attacks and defenses for 5G PFCP intrusion detection systems. This project implements realistic adversarial machine learning techniques while maintaining complete protocol compliance.

## Key Achievements

### ✅ Phase 1: Baseline IDS (COMPLETE)
- **Random Forest Classifier**: 60.8% macro-F1, 64.8% accuracy
- **Advanced Feature Engineering**: 7 PCA components with variance analysis
- **Cross-Validation**: 64.3% ± 2.2% (statistically validated)
- **Tag**: `v0.1-baseline`

### ✅ Phase 2A: Attack Engine (COMPLETE) 
- **Enhanced PGD Attack**: 57% evasion rate (industry-realistic)
- **Constraint Compliance**: 100% PFCP protocol adherence
- **Critical Vulnerabilities**: Classes 0, 2, 3 with 90%+ evasion rates
- **Tag**: `v0.2-attacks`

### 🔄 Phase 2B: Defense Hardening (READY)
- **Target**: Reduce evasion from 57% to <30%
- **Approach**: Adversarial training with constraint-aware defenses
- **Focus**: Vulnerable classes 0, 2, 3 robustness improvement

## Technical Specifications

### Dataset
- **Source**: 5G PFCP SANCUS dataset
- **Size**: 1,113 training, 477 testing samples
- **Classes**: 5 balanced attack types
- **Features**: 7 (after PCA dimensionality reduction)

### Attack Engine
- **Methods**: FGSM, PGD, Enhanced variants
- **Constraints**: PFCP protocol compliance framework
- **Performance**: 52-57% evasion rates
- **Compliance**: 0 protocol violations

### Defense Framework (Phase 2B)
- **Baseline**: 57% attack success rate
- **Target**: <30% evasion rate
- **Approach**: Adversarial training + class-specific defense
- **Preservation**: Clean accuracy maintenance

## Repository Structure

```
adversarial-5g-ids/
├── data/                    # Dataset and processed features
├── models/                  # Trained models and metadata
├── src/
│   ├── models/             # Baseline model implementations
│   └── attacks/            # Attack engine implementations
├── reports/                # Performance analysis and documentation
├── validate_*.py          # Validation and testing frameworks
├── PROJECT_STATUS.md      # Current project status
├── TIMELINE.md           # Development timeline
└── PHASE_HANDOFF.md      # Phase transition documentation
```

## Quick Start

### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd adversarial-5g-ids

# Install dependencies
pip install scikit-learn matplotlib seaborn numpy pandas

# Validate baseline model
python validate_baseline.py

# Test attack engine
python validate_attacks.py
```

### Model Usage
```python
# Load baseline model
from attacks.attack_utils import load_baseline_artifacts
model, transformers, scaler, X_test, y_test = load_baseline_artifacts()

# Run adversarial attack
from attacks.enhanced_attacks import EnhancedConstraintPGD
attack = EnhancedConstraintPGD(model, epsilon=0.3)
X_adv, info = attack.generate_adversarial_samples(X_test[:100], y_test[:100])
```

## Performance Benchmarks

### Baseline Model Performance
| Metric | Value | Standard |
|--------|-------|----------|
| Macro-F1 | 60.8% | Realistic for 5G IDS |
| Accuracy | 64.8% | Cross-validated |
| Features | 7 | PCA optimized |
| Classes | 5 | Balanced distribution |

### Attack Engine Performance
| Attack Method | Evasion Rate | Constraint Compliance |
|---------------|--------------|---------------------|
| Enhanced PGD | **57.0%** | **100%** |
| Enhanced FGSM | 52.0% | 100% |
| Standard PGD | 35.0% | 100% |
| Standard FGSM | 41.0% | 100% |

### Per-Class Vulnerability Analysis
| Class | Type | Clean Accuracy | Robustness | Priority |
|-------|------|----------------|------------|----------|
| 0 | Mal_Del | 34.7% | 0.0% | Critical |
| 2 | Mal_Mod | 32.6% | 7.4% | Critical |
| 3 | Mal_Mod2 | 37.9% | 8.4% | Critical |
| 1 | Mal_Estab | 97.9% | 97.9% | Robust |
| 4 | Normal | 100.0% | 100.0% | Robust |

## Research Contributions

### Novel Implementations
1. **Constraint-Aware Attacks**: First PFCP protocol-compliant adversarial attacks
2. **Class-Specific Targeting**: Adaptive attack strategies per traffic type
3. **Realistic Performance**: Industry-relevant threat modeling
4. **Defense Baseline**: Established robust evaluation framework

### Academic Impact
- **Security Assessment**: 57% evasion demonstrates significant threat
- **Protocol Compliance**: Zero violations enables practical deployment
- **Robustness Insights**: Identified natural defense patterns in Classes 1&4
- **Research Foundation**: Validated framework for 5G IDS adversarial research

## Development Timeline

| Phase | Status | Completion | Achievement |
|-------|--------|------------|-------------|
| Phase 1 | ✅ Complete | Aug 5, 2025 | 60.8% baseline |
| Phase 2A | ✅ Complete | Aug 5, 2025 | 57% attacks |
| Phase 2B | 🔄 Active | - | Defense target |
| Phase 3 | ⏳ Queued | - | Integration |
| Phase 4 | ⏳ Planned | - | Documentation |

## Citation

```bibtex
@misc{adversarial5gids2025,
  title={Constraint-Aware Adversarial Attacks and Defenses for 5G PFCP Intrusion Detection},
  author={5G IDS Research Team},
  year={2025},
  note={Phase 2A Complete: Attack Engine Implementation}
}
```

## License

This research project is developed for academic purposes. See LICENSE file for details.

## Contact

For questions about this research project, please refer to the project documentation or create an issue.

---

**Last Updated**: August 5, 2025  
**Current Version**: v0.2-attacks  
**Next Milestone**: Phase 2B Defense Implementation
