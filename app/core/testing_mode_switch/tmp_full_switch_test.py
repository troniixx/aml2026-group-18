from pathlib import Path
import sys
sys.path.insert(0, '.')
from core.config import load
from core.detector import Detector

cfg = load(Path('config.yaml'))
d = Detector(cfg)
print('initial', d._active_key, d._model_type, type(d._model).__name__)
for key in ['model_2', 'model_3', 'model_2', 'model_1']:
    print('switching to', key)
    d.switch_model(key)
    print('after switch', d._active_key, d._model_type, type(d._model).__name__, 'model is None?', d._model is None)
    if d._model is None:
        print('model None after switch', key)
        break
