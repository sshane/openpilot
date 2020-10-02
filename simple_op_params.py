import json


class customParams:
  def __init__(self):
    self.file_path = '/data/whatever.json'

  def get(self, param):
    # be aware this reads the file every iteration,
    # it constantly reads which might cause issues if a whole bunch of files are reading the same file while writing to it
    data = self.read_json()
    return data[param]

  def read_json(self):
    with open(self.file_path, "r") as f:
      data = json.loads(f.read())  # in dictionary format
    return data
