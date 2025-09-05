# Adversarial 5G IDS - CLI User Guide

## Overview

The **Adversarial 5G IDS CLI** provides a comprehensive command-line interface for 5G PFCP intrusion detection with adversarial machine learning capabilities. This guide covers all available commands and their usage.

## Installation & Setup

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)
- 5G PFCP dataset (included with project)

### Quick Start
```bash
# Navigate to project directory
cd /path/to/adversarial-5g-ids-main

# Activate virtual environment
source /path/to/adv5g/bin/activate

# Test installation
python src/cli/adv5g_cli.py --version
```

## Command Reference

### Main CLI Usage
```bash
python src/cli/adv5g_cli.py <command> [options]
```

### Available Commands

#### 1. `detect` - Threat Detection
**Purpose**: Analyze network traffic for potential security threats using trained ML models.

```bash
# Basic threat detection with robust model
python src/cli/adv5g_cli.py detect --data sample --model robust

# Detailed analysis with custom threshold
python src/cli/adv5g_cli.py detect --data sample --model both --threshold 0.6 --detailed

# Save results to file
python src/cli/adv5g_cli.py detect --data /path/to/data.csv --output results.json
```

**Options**:
- `--data` / `-d`: Input data file (CSV, NPY, or "sample")
- `--model` / `-m`: Model type (baseline, robust, both) [default: robust]
- `--output` / `-o`: Output file for results (JSON format)
- `--threshold`: Detection threshold (default: 0.5)
- `--detailed`: Show detailed analysis for each sample
- `--batch-size`: Processing batch size (default: 100)

#### 2. `attack` - Adversarial Attack Generation
**Purpose**: Test model robustness using various adversarial attack methods.

```bash
# Enhanced PGD attack on both models
python src/cli/adv5g_cli.py attack --method enhanced_pgd --target both --samples 100

# FGSM attack with custom epsilon
python src/cli/adv5g_cli.py attack --method fgsm --epsilon 0.2 --samples 50

# Attack with constraint checking
python src/cli/adv5g_cli.py attack --method pgd --constraint-check --save-adversarial attacks.npy
```

**Options**:
- `--method` / `-m`: Attack method (fgsm, pgd, enhanced_pgd, noise) [default: enhanced_pgd]
- `--target`: Target model(s) (baseline, robust, both) [default: both]
- `--epsilon` / `-e`: Attack strength (perturbation budget)
- `--samples` / `-n`: Number of samples to attack (default: 100)
- `--output` / `-o`: Output file for results (JSON)
- `--save-adversarial`: Save adversarial examples (NPY format)
- `--class-target`: Target specific class (0-4)
- `--steps`: Number of attack steps (for iterative methods)
- `--constraint-check`: Verify PFCP protocol compliance

#### 3. `defend` - Defense Evaluation
**Purpose**: Test and analyze the effectiveness of adversarial defenses.

```bash
# Comprehensive defense evaluation
python src/cli/adv5g_cli.py defend --evaluate --compare-models --robustness-test

# Custom noise levels testing
python src/cli/adv5g_cli.py defend --robustness-test --noise-levels 0.1,0.2,0.4

# Detailed report generation
python src/cli/adv5g_cli.py defend --detailed-report --save-metrics metrics.json
```

**Options**:
- `--evaluate`: Run defense evaluation
- `--compare-models`: Compare baseline vs robust model performance
- `--robustness-test`: Run comprehensive robustness testing
- `--noise-levels`: Comma-separated noise levels (default: 0.05,0.1,0.2,0.3,0.5)
- `--samples` / `-n`: Number of samples for evaluation (default: 100)
- `--output` / `-o`: Output file for results (JSON)
- `--detailed-report`: Generate detailed analysis report
- `--save-metrics`: Save detailed metrics to file

#### 4. `analyze` - System Analysis & Reporting
**Purpose**: Create comprehensive security analysis reports and performance visualizations.

```bash
# Generate comprehensive report
python src/cli/adv5g_cli.py analyze --generate-report --system-status --security-assessment

# HTML report with plots
python src/cli/adv5g_cli.py analyze --generate-report --format html --include-plots

# System status check
python src/cli/adv5g_cli.py analyze --system-status --model-analysis
```

**Options**:
- `--generate-report`: Generate comprehensive security report
- `--system-status`: Show current system status and capabilities
- `--model-analysis`: Analyze model performance and characteristics
- `--security-assessment`: Perform security assessment and risk analysis
- `--output` / `-o`: Output file for report
- `--format`: Report format (json, html, txt, markdown) [default: json]
- `--include-plots`: Include performance plots in report
- `--detailed`: Generate detailed analysis with extended metrics

#### 5. `demo` - Interactive Demonstrations
**Purpose**: Showcase system capabilities with interactive demonstrations.

```bash
# Full pipeline demonstration
python src/cli/adv5g_cli.py demo --full-pipeline --scenario technical

# Quick presentation demo
python src/cli/adv5g_cli.py demo --quick --scenario presentation

# Interactive attack/defense demo
python src/cli/adv5g_cli.py demo --attack-demo --defense-demo --interactive
```

**Options**:
- `--full-pipeline`: Run complete pipeline demonstration
- `--interactive`: Enable interactive mode with user prompts
- `--quick`: Run quick demo (reduced samples and iterations)
- `--attack-demo`: Demonstrate adversarial attack capabilities
- `--defense-demo`: Demonstrate defense and robustness capabilities
- `--real-time`: Simulate real-time threat detection
- `--scenario`: Demo scenario type (presentation, technical, business)

## Global Options

Available for all commands:
- `--version`: Show program version
- `--verbose` / `-v`: Enable verbose output
- `--config`: Path to configuration file
- `--help` / `-h`: Show help message

## Example Workflows

### 1. Complete Security Assessment
```bash
# Step 1: Detect threats in sample data
python src/cli/adv5g_cli.py detect --data sample --model robust --detailed

# Step 2: Test attack resistance
python src/cli/adv5g_cli.py attack --method enhanced_pgd --target robust --samples 100

# Step 3: Evaluate defenses
python src/cli/adv5g_cli.py defend --evaluate --robustness-test --detailed-report

# Step 4: Generate comprehensive report
python src/cli/adv5g_cli.py analyze --generate-report --security-assessment --format html
```

### 2. Research & Development
```bash
# Compare attack methods
python src/cli/adv5g_cli.py attack --method fgsm --samples 50 --output fgsm_results.json
python src/cli/adv5g_cli.py attack --method pgd --samples 50 --output pgd_results.json

# Robustness analysis across noise levels
python src/cli/adv5g_cli.py defend --robustness-test --noise-levels 0.05,0.1,0.15,0.2,0.25,0.3

# Model performance comparison
python src/cli/adv5g_cli.py defend --compare-models --save-metrics comparison.json
```

### 3. Demonstration & Presentation
```bash
# Quick overview for stakeholders
python src/cli/adv5g_cli.py demo --quick --scenario business

# Technical deep dive
python src/cli/adv5g_cli.py demo --full-pipeline --scenario technical

# Interactive presentation
python src/cli/adv5g_cli.py demo --interactive --scenario presentation
```

## Performance Benchmarks

### Tested Performance Metrics
- **Robust Model Accuracy**: 66.5%
- **Attack Success Rate**: 50-57% (Enhanced PGD)
- **Protocol Compliance**: 100%
- **Processing Speed**: ~20ms per sample
- **Defense Effectiveness**: Excellent (score: 0.025)

### System Requirements
- **Memory**: 2GB+ RAM recommended
- **Storage**: 500MB for models and data
- **CPU**: Multi-core recommended for batch processing
- **Python**: 3.9+ with scientific computing libraries

## Troubleshooting

### Common Issues

1. **Missing Model Files**
   ```bash
   # Check model availability
   python src/cli/adv5g_cli.py analyze --system-status
   ```

2. **Data Format Issues**
   ```bash
   # Use sample data for testing
   python src/cli/adv5g_cli.py detect --data sample
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch size
   python src/cli/adv5g_cli.py detect --batch-size 50
   ```

### Getting Help
- Use `--help` with any command for detailed options
- Check system status: `python src/cli/adv5g_cli.py analyze --system-status`
- Run quick demo: `python src/cli/adv5g_cli.py demo --quick`

## Advanced Usage

### Configuration Files
Create a JSON configuration file for custom settings:
```json
{
  "models": {
    "robust_path": "/custom/path/to/model.joblib"
  },
  "attack": {
    "default_epsilon": 0.3,
    "default_steps": 20
  },
  "output": {
    "verbose": true,
    "colored": true
  }
}
```

Use with: `python src/cli/adv5g_cli.py --config config.json <command>`

### Batch Processing
```bash
# Process multiple files
for file in data/*.csv; do
  python src/cli/adv5g_cli.py detect --data "$file" --output "results/$(basename $file .csv)_results.json"
done
```

## Integration Examples

### Python Script Integration
```python
import subprocess
import json

# Run detection and get results
result = subprocess.run([
    'python', 'src/cli/adv5g_cli.py', 'detect',
    '--data', 'sample',
    '--model', 'robust',
    '--output', 'temp_results.json'
], capture_output=True, text=True)

# Load results
with open('temp_results.json', 'r') as f:
    detection_results = json.load(f)
```

### Real-time Monitoring
```bash
# Continuous monitoring script
while true; do
  python src/cli/adv5g_cli.py detect --data /live/traffic.csv --output /logs/detection_$(date +%s).json
  sleep 60
done
```

---

*For additional support and advanced features, refer to the project documentation and code examples.*
