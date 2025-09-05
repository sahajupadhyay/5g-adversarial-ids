#!/usr/bin/env python3
"""
Adversarial 5G IDS - Command Line Interface

A comprehensive CLI tool for 5G PFCP intrusion detection with adversarial ML capabilities.
Provides unified access to baseline models, attack simulation, defense evaluation, and analytics.

Usage:
    python adv5g_cli.py detect --data <file> [--model baseline|robust]
    python adv5g_cli.py attack --method <method> [--epsilon <value>] [--target <model>]
    python adv5g_cli.py defend --evaluate [--attacks] [--report]
    python adv5g_cli.py analyze --generate-report [--output <file>]
    python adv5g_cli.py demo [--full-pipeline] [--interactive]

Author: 5G IDS Research Team
Version: v0.3-defenses
Date: September 2025
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cli.commands.detect import DetectCommand
from src.cli.commands.attack import AttackCommand
from src.cli.commands.defend import DefendCommand
from src.cli.commands.analyze import AnalyzeCommand
from src.cli.commands.demo import DemoCommand
from src.cli.utils.config import CLIConfig
from src.cli.utils.output import CLIOutput, Colors

class Adv5GCLI:
    """Main CLI application class."""
    
    def __init__(self):
        self.config = CLIConfig()
        self.output = CLIOutput()
        self.commands = {
            'detect': DetectCommand(),
            'attack': AttackCommand(),
            'defend': DefendCommand(),
            'analyze': AnalyzeCommand(),
            'demo': DemoCommand()
        }
    
    def create_parser(self):
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog='adv5g_cli',
            description='Adversarial 5G IDS - Professional Security Analysis Tool',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s detect --data sample_traffic.csv --model robust
  %(prog)s attack --method pgd --epsilon 0.3 --target baseline
  %(prog)s defend --evaluate --attacks --report
  %(prog)s analyze --generate-report --output security_report.pdf
  %(prog)s demo --full-pipeline --interactive

For more information about each command, use:
  %(prog)s <command> --help
            """
        )
        
        # Global options
        parser.add_argument(
            '--version', 
            action='version', 
            version='%(prog)s v0.3-defenses'
        )
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='<command>'
        )
        
        # Register command parsers
        for name, command in self.commands.items():
            command.add_parser(subparsers)
        
        return parser
    
    def run(self, args=None):
        """Main entry point for the CLI."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # Configure output verbosity
        if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
            self.output.set_verbose(True)
        
        # Load custom config if provided
        if hasattr(parsed_args, 'config') and parsed_args.config:
            self.config.load_config(parsed_args.config)
        
        # Show header
        self.show_header()
        
        # Handle no command case
        if not parsed_args.command:
            self.output.error("No command specified. Use --help for usage information.")
            return 1
        
        # Execute command
        try:
            command = self.commands[parsed_args.command]
            return command.execute(parsed_args, self.config, self.output)
        
        except KeyboardInterrupt:
            self.output.warning("\nOperation cancelled by user.")
            return 130
        
        except Exception as e:
            self.output.error(f"Unexpected error: {str(e)}")
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def show_header(self):
        """Display the application header."""
        header = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                     Adversarial 5G IDS Security Suite                       ║
║                                                                              ║
║  🛡️  Complete attack-defense pipeline for 5G PFCP intrusion detection      ║
║  🎯  Baseline Model: 66.5% accuracy | Robust Defense: Excellent rating     ║
║  ⚡  Real-time threat detection with adversarial robustness                ║
║                                                                              ║
║  Version: v0.3-defenses | Status: Production Ready                         ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
        print(header)

def main():
    """Entry point for the CLI application."""
    cli = Adv5GCLI()
    sys.exit(cli.run())

if __name__ == '__main__':
    main()
