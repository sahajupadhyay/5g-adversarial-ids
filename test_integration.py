#!/usr/bin/env python3
"""
Adversarial 5G IDS - System Integration Test Suite
Comprehensive testing of all CLI components and functionality

Author: 5G IDS Research Team
Version: v0.3-defenses
Date: September 2025
"""

import subprocess
import json
import os
import sys
import time
from datetime import datetime

class CLIIntegrationTester:
    def __init__(self, project_root):
        self.project_root = project_root
        self.cli_path = os.path.join(project_root, "src/cli/adv5g_cli.py")
        self.python_path = "/Users/sahajupadhyay/Desktop/Capstone/adv5g/bin/python"
        self.test_results = {}
        self.start_time = None
        
    def run_cli_command(self, command_args, timeout=120):
        """Run a CLI command and return result"""
        cmd = [self.python_path, self.cli_path] + command_args
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=self.project_root
            )
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Command timed out after {timeout}s',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }

    def test_version_and_help(self):
        """Test basic CLI functionality"""
        print("🧪 Testing CLI Version and Help...")
        
        # Test version
        result = self.run_cli_command(['--version'])
        self.test_results['version'] = {
            'success': result['success'] and 'v0.3-defenses' in result['stdout'],
            'details': result
        }
        
        # Test main help
        result = self.run_cli_command(['--help'])
        self.test_results['main_help'] = {
            'success': result['success'] and 'detect' in result['stdout'],
            'details': result
        }
        
        # Test individual command help
        commands = ['detect', 'attack', 'defend', 'analyze', 'demo']
        for cmd in commands:
            result = self.run_cli_command([cmd, '--help'])
            self.test_results[f'{cmd}_help'] = {
                'success': result['success'],
                'details': result
            }

    def test_detect_command(self):
        """Test threat detection functionality"""
        print("🔍 Testing Threat Detection...")
        
        # Basic detection test
        result = self.run_cli_command([
            'detect', '--data', 'sample', '--model', 'robust'
        ])
        self.test_results['detect_basic'] = {
            'success': result['success'] and 'Detection Results Summary' in result['stdout'],
            'details': result
        }
        
        # Detailed detection test
        result = self.run_cli_command([
            'detect', '--data', 'sample', '--model', 'robust', '--detailed'
        ])
        self.test_results['detect_detailed'] = {
            'success': result['success'] and 'Detailed Sample Analysis' in result['stdout'],
            'details': result
        }

    def test_attack_command(self):
        """Test adversarial attack functionality"""
        print("⚔️ Testing Adversarial Attacks...")
        
        # Enhanced PGD attack test
        result = self.run_cli_command([
            'attack', '--method', 'enhanced_pgd', '--target', 'robust', 
            '--samples', '25', '--constraint-check'
        ])
        self.test_results['attack_enhanced_pgd'] = {
            'success': result['success'] and 'Attack Results Summary' in result['stdout'],
            'details': result
        }
        
        # FGSM attack test
        result = self.run_cli_command([
            'attack', '--method', 'fgsm', '--target', 'robust', '--samples', '25'
        ])
        self.test_results['attack_fgsm'] = {
            'success': result['success'] and 'Attack Results Summary' in result['stdout'],
            'details': result
        }

    def test_defend_command(self):
        """Test defense evaluation functionality"""
        print("🛡️ Testing Defense Evaluation...")
        
        # Basic defense evaluation
        result = self.run_cli_command([
            'defend', '--evaluate', '--samples', '50'
        ])
        self.test_results['defend_evaluate'] = {
            'success': result['success'] and 'Defense Evaluation Suite' in result['stdout'],
            'details': result
        }
        
        # Robustness testing
        result = self.run_cli_command([
            'defend', '--robustness-test', '--samples', '50'
        ])
        self.test_results['defend_robustness'] = {
            'success': result['success'] and 'robustness' in result['stdout'].lower(),
            'details': result
        }

    def test_analyze_command(self):
        """Test analysis and reporting functionality"""
        print("📊 Testing Analysis and Reporting...")
        
        # System status analysis
        result = self.run_cli_command([
            'analyze', '--system-status'
        ])
        self.test_results['analyze_status'] = {
            'success': result['success'] and 'System Status Overview' in result['stdout'],
            'details': result
        }
        
        # Security assessment
        result = self.run_cli_command([
            'analyze', '--security-assessment'
        ])
        self.test_results['analyze_security'] = {
            'success': result['success'] and 'Security Assessment' in result['stdout'],
            'details': result
        }

    def test_demo_command(self):
        """Test demonstration functionality"""
        print("🎯 Testing Demonstrations...")
        
        # Quick demo
        result = self.run_cli_command([
            'demo', '--quick', '--scenario', 'presentation'
        ])
        self.test_results['demo_quick'] = {
            'success': result['success'] and 'Pipeline demonstration completed' in result['stdout'],
            'details': result
        }
        
        # Attack demo
        result = self.run_cli_command([
            'demo', '--attack-demo', '--quick'
        ])
        self.test_results['demo_attack'] = {
            'success': result['success'],
            'details': result
        }

    def test_pipeline_integration(self):
        """Test full pipeline integration"""
        print("🚀 Testing Full Pipeline Integration...")
        
        # Full pipeline demo
        result = self.run_cli_command([
            'demo', '--full-pipeline', '--scenario', 'technical'
        ], timeout=180)  # Longer timeout for full pipeline
        
        self.test_results['pipeline_full'] = {
            'success': result['success'] and 'Pipeline demonstration completed' in result['stdout'],
            'details': result
        }

    def test_error_handling(self):
        """Test error handling and edge cases"""
        print("⚠️ Testing Error Handling...")
        
        # Invalid command
        result = self.run_cli_command(['invalid_command'])
        self.test_results['error_invalid_command'] = {
            'success': not result['success'],  # Should fail
            'details': result
        }
        
        # Invalid file path
        result = self.run_cli_command([
            'detect', '--data', '/nonexistent/file.csv'
        ])
        self.test_results['error_invalid_file'] = {
            'success': not result['success'],  # Should fail
            'details': result
        }

    def run_all_tests(self):
        """Run complete test suite"""
        print("=" * 80)
        print("🧪 ADVERSARIAL 5G IDS - SYSTEM INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"Start Time: {datetime.now()}")
        print(f"Python Path: {self.python_path}")
        print(f"CLI Path: {self.cli_path}")
        print(f"Project Root: {self.project_root}")
        print()
        
        self.start_time = time.time()
        
        try:
            # Run all test categories
            self.test_version_and_help()
            self.test_detect_command()
            self.test_attack_command()
            self.test_defend_command()
            self.test_analyze_command()
            self.test_demo_command()
            self.test_pipeline_integration()
            self.test_error_handling()
            
        except KeyboardInterrupt:
            print("\n❌ Test suite interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            return False
        
        return self.generate_report()

    def generate_report(self):
        """Generate comprehensive test report"""
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Count results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results.values() if test['success'])
        failed_tests = total_tests - passed_tests
        
        print("\n" + "=" * 80)
        print("📋 TEST RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Duration: {duration:.2f} seconds")
        print()
        
        # Detailed results by category
        categories = {
            'CLI Basics': ['version', 'main_help', 'detect_help', 'attack_help', 'defend_help', 'analyze_help', 'demo_help'],
            'Threat Detection': ['detect_basic', 'detect_detailed'],
            'Adversarial Attacks': ['attack_enhanced_pgd', 'attack_fgsm'],
            'Defense Evaluation': ['defend_evaluate', 'defend_robustness'],
            'Analysis & Reporting': ['analyze_status', 'analyze_security'],
            'Demonstrations': ['demo_quick', 'demo_attack'],
            'Pipeline Integration': ['pipeline_full'],
            'Error Handling': ['error_invalid_command', 'error_invalid_file']
        }
        
        for category, tests in categories.items():
            category_passed = sum(1 for test in tests if test in self.test_results and self.test_results[test]['success'])
            category_total = len([test for test in tests if test in self.test_results])
            if category_total > 0:
                print(f"{category}: {category_passed}/{category_total} ✅")
        
        print("\n" + "=" * 80)
        print("🔍 DETAILED TEST RESULTS")
        print("=" * 80)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"{test_name}: {status}")
            
            if not result['success']:
                print(f"  Error: {result['details']['stderr'][:100]}...")
                if result['details']['returncode'] != 0:
                    print(f"  Exit Code: {result['details']['returncode']}")
        
        # Overall system health
        critical_tests = ['detect_basic', 'attack_enhanced_pgd', 'defend_evaluate', 'demo_quick']
        critical_passed = sum(1 for test in critical_tests if test in self.test_results and self.test_results[test]['success'])
        
        print("\n" + "=" * 80)
        print("🎯 SYSTEM HEALTH ASSESSMENT")
        print("=" * 80)
        
        if critical_passed == len(critical_tests):
            print("🟢 SYSTEM STATUS: EXCELLENT - All critical functionality working")
        elif critical_passed >= len(critical_tests) * 0.75:
            print("🟡 SYSTEM STATUS: GOOD - Minor issues detected")
        else:
            print("🔴 SYSTEM STATUS: NEEDS ATTENTION - Critical issues detected")
        
        print(f"Critical Tests Passed: {critical_passed}/{len(critical_tests)}")
        print(f"Overall System Health: {(passed_tests/total_tests)*100:.1f}%")
        
        # Save detailed results
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests/total_tests)*100
            },
            'results': self.test_results
        }
        
        report_file = os.path.join(self.project_root, 'test_results.json')
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📁 Detailed results saved to: {report_file}")
        
        return passed_tests == total_tests

def main():
    """Main test execution"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure we're in the right directory
    if not os.path.exists(os.path.join(project_root, 'src', 'cli', 'adv5g_cli.py')):
        print("❌ Error: Could not find CLI script. Please run from project root.")
        sys.exit(1)
    
    tester = CLIIntegrationTester(project_root)
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
        sys.exit(1)

if __name__ == "__main__":
    main()
