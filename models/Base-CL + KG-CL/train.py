import os,sys,runpy
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'shared'));os.environ['KAML_MODEL_KEY']='15'
os.environ.setdefault('KAML_RUN_DIR',str(Path(__file__).resolve().parent/'runs'));runpy.run_path(str(r/'shared/train_neural.py'),run_name='__main__')
