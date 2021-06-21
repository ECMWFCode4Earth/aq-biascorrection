import os
from pathlib import Path

str2agg = {
    'daily': 'D',
    'monthly': 'M'
}


ROOT_DIR = Path(os.path.dirname(os.path.abspath("setup.py")))

log_fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'