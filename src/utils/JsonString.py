import json
def is_json(josn_str: str):
  try:
    json.loads(josn_str)
  except ValueError as e:
    return False
  return True