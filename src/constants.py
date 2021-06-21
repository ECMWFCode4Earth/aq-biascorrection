import os
from pathlib import Path

str2agg = {
    'daily': 'D',
    'monthly': 'M'
}


ROOT_DIR = Path(__file__).parent.parent

log_fmt = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'