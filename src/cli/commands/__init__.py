"""
CLI commands package for the Adversarial 5G IDS system.
"""

from .detect import DetectCommand
from .attack import AttackCommand
from .defend import DefendCommand
from .analyze import AnalyzeCommand
from .demo import DemoCommand

__all__ = ['DetectCommand', 'AttackCommand', 'DefendCommand', 'AnalyzeCommand', 'DemoCommand']
