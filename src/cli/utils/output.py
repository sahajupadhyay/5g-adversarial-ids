"""
CLI Output utilities for pretty printing and colored output.

Provides consistent formatting, colors, and styling for the CLI interface.
"""

import sys
from datetime import datetime
from typing import Optional

class Colors:
    """ANSI color codes for terminal output."""
    
    # Basic colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    
    # Reset
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        """Disable all colors (for non-terminal output)."""
        for attr in dir(cls):
            if not attr.startswith('_') and attr != 'disable':
                setattr(cls, attr, '')

class Icons:
    """Unicode icons for CLI output."""
    
    SUCCESS = '✅'
    ERROR = '❌'
    WARNING = '⚠️ '
    INFO = 'ℹ️ '
    SECURITY = '🛡️'
    ATTACK = '⚔️ '
    DEFENSE = '🔰'
    ANALYSIS = '📊'
    LOADING = '⏳'
    ROCKET = '🚀'
    TARGET = '🎯'
    CHECKMARK = '✓'
    CROSS = '✗'
    ARROW_RIGHT = '→'
    ARROW_UP = '↑'
    ARROW_DOWN = '↓'

class CLIOutput:
    """Handles all CLI output with consistent formatting."""
    
    def __init__(self, verbose: bool = False, colored: bool = None):
        self._verbose_enabled = verbose
        
        # Auto-detect color support
        if colored is None:
            self.colored = sys.stdout.isatty()
        else:
            self.colored = colored
        
        if not self.colored:
            Colors.disable()
    
    def set_verbose(self, verbose: bool):
        """Enable or disable verbose output."""
        self._verbose_enabled = verbose
    
    def _timestamp(self) -> str:
        """Get current timestamp for verbose output."""
        return datetime.now().strftime('%H:%M:%S')
    
    def _print(self, message: str, file=None, end='\n'):
        """Print message to specified file (stdout by default)."""
        if file is None:
            file = sys.stdout
        print(message, file=file, end=end)
    
    def success(self, message: str, details: Optional[str] = None):
        """Print success message."""
        prefix = f"{Icons.SUCCESS} {Colors.GREEN}SUCCESS{Colors.RESET}"
        self._print(f"{prefix}: {message}")
        
        if details and self.verbose:
            self._print(f"  {Colors.DIM}{details}{Colors.RESET}")
    
    def error(self, message: str, details: Optional[str] = None):
        """Print error message."""
        prefix = f"{Icons.ERROR} {Colors.RED}ERROR{Colors.RESET}"
        self._print(f"{prefix}: {message}", file=sys.stderr)
        
        if details and self._verbose_enabled:
            self._print(f"  {Colors.DIM}{details}{Colors.RESET}", file=sys.stderr)
    
    def warning(self, message: str, details: Optional[str] = None):
        """Print warning message."""
        prefix = f"{Icons.WARNING}{Colors.YELLOW}WARNING{Colors.RESET}"
        self._print(f"{prefix}: {message}")
        
        if details and self._verbose_enabled:
            self._print(f"  {Colors.DIM}{details}{Colors.RESET}")
    
    def info(self, message: str, details: Optional[str] = None):
        """Print info message."""
        prefix = f"{Icons.INFO}{Colors.CYAN}INFO{Colors.RESET}"
        self._print(f"{prefix}: {message}")
        
        if details and self._verbose_enabled:
            self._print(f"  {Colors.DIM}{details}{Colors.RESET}")
    
    def verbose(self, message: str):
        """Print verbose message (only if verbose mode is enabled)."""
        if self._verbose_enabled:
            timestamp = self._timestamp()
            self._print(f"{Colors.DIM}[{timestamp}] {message}{Colors.RESET}")
    
    def header(self, title: str, subtitle: Optional[str] = None):
        """Print section header."""
        line = '═' * 60
        self._print(f"\n{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
        self._print(f"{Colors.BOLD}{Colors.CYAN} {title}{Colors.RESET}")
        
        if subtitle:
            self._print(f"{Colors.DIM} {subtitle}{Colors.RESET}")
        
        self._print(f"{Colors.BOLD}{Colors.CYAN}{line}{Colors.RESET}")
    
    def subheader(self, title: str):
        """Print subsection header."""
        self._print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
        self._print(f"{Colors.DIM}{'─' * len(title)}{Colors.RESET}")
    
    def table_row(self, columns: list, widths: list = None):
        """Print a table row with aligned columns."""
        if widths is None:
            widths = [15] * len(columns)
        
        row = ""
        for i, (col, width) in enumerate(zip(columns, widths)):
            if i == 0:
                row += f"{str(col):<{width}}"
            else:
                row += f" │ {str(col):<{width}}"
        
        self._print(row)
    
    def table_separator(self, widths: list):
        """Print table separator line."""
        sep = ""
        for i, width in enumerate(widths):
            if i == 0:
                sep += "─" * width
            else:
                sep += "─┼─" + "─" * width
        
        self._print(sep)
    
    def progress(self, message: str, current: int = None, total: int = None):
        """Print progress message."""
        if current is not None and total is not None:
            percentage = (current / total) * 100
            bar_length = 30
            filled_length = int(bar_length * current // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            self._print(f"\r{Icons.LOADING} {message} [{bar}] {percentage:.1f}% ({current}/{total})", end='')
            
            if current == total:
                self._print("")  # New line when complete
        else:
            self._print(f"{Icons.LOADING} {message}...")
    
    def metric(self, name: str, value: str, unit: str = "", color: str = None):
        """Print a metric with formatting."""
        if color:
            value_str = f"{color}{value}{Colors.RESET}"
        else:
            value_str = f"{Colors.BOLD}{value}{Colors.RESET}"
        
        self._print(f"  {name}: {value_str}{unit}")
    
    def bullet(self, message: str, icon: str = None):
        """Print bullet point."""
        bullet_icon = icon or f"{Colors.CYAN}•{Colors.RESET}"
        self._print(f"  {bullet_icon} {message}")
    
    def json_pretty(self, data: dict, title: Optional[str] = None):
        """Print JSON data in a pretty format."""
        import json
        
        if title:
            self.subheader(title)
        
        json_str = json.dumps(data, indent=2, sort_keys=True)
        for line in json_str.split('\n'):
            self._print(f"  {Colors.DIM}{line}{Colors.RESET}")
    
    def separator(self, char: str = "─", length: int = 60):
        """Print a separator line."""
        self._print(f"{Colors.DIM}{char * length}{Colors.RESET}")
    
    def blank_line(self):
        """Print a blank line."""
        self._print("")

class ProgressBar:
    """Simple progress bar for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress", width: int = 30):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.output = CLIOutput()
    
    def update(self, increment: int = 1):
        """Update progress bar."""
        self.current = min(self.current + increment, self.total)
        self.output.progress(self.description, self.current, self.total)
    
    def finish(self, message: str = "Complete"):
        """Finish progress bar with success message."""
        self.current = self.total
        self.output.progress(self.description, self.current, self.total)
        self.output.success(message)
