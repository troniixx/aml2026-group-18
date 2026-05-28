from pathlib import Path
import sys
sys.path.insert(0, '.')
from core.config import load
from core.detector import Detector

cfg = load(Path('config.yaml'))
print('Active model', cfg.active_model)
print('Model 3 type', cfg.models['model_3'].model_type)
print('Model 3 script', cfg.models['model_3'].script_path)
print('Model 3 weights', cfg.models['model_3'].weights_path)

d = Detector(cfg)
print('initial model type', d._model_type, type(d._model))
print('switching to model_3')
d.switch_model('model_3')
print('after switch type', d._model_type, type(d._model))
print('model methods', [m for m in ['predict', 'predict_proba', 'decision_function'] if hasattr(d._model, m)])
print('model repr', repr(d._model))
