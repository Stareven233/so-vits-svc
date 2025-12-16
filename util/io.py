import json
import pickle
from pathlib import Path
import yaml


def check_path_exist(func):

  def inner(*args, **kwargs):
    p = args[0]
    if not isinstance(p, Path):
      p = Path(p)
      args = (p, *args[1:])
    if not p.is_file():
      raise FileNotFoundError(p)
    res = func(*args, **kwargs)
    return res

  return inner


@check_path_exist
def load_json(path, encoding='utf-8', **kwargs) -> dict:
  with path.open('r', encoding=encoding) as f:
    return json.load(f, **kwargs)


def write_json(path, obj, encoding='utf-8', **kwargs) -> None:
  with open(path, 'w', encoding=encoding) as f:
    json.dump(obj, f, ensure_ascii=False, **kwargs)


@check_path_exist
def load_pickle(path, **kwargs):
  with path.open('rb') as f:
    return pickle.load(f, **kwargs)


def write_pickle(path, obj, **kwargs) -> None:
  with open(path, 'wb') as f:
    pickle.dump(obj, f, **kwargs)


@check_path_exist
def load_yaml(path) -> dict:
  with path.open('r') as f:
    return yaml.safe_load(f)


def write_yaml(path, obj, encoding='utf-8', **kwargs) -> None:
  with open(path, 'w', encoding=encoding) as f:
    yaml.dump(obj, f, encoding=encoding, **kwargs)
