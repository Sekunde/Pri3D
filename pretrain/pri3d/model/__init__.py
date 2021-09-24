from model.model import Pri3D
MODELS = [Pri3D]

def load_model(name):
  '''Creates and returns an instance of the datasets given its name.
  '''
  # Find the model class from its name
  mdict = {model.__name__: model for model in MODELS}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid dataset names
    for model in MODELS:
      print('\t* {}'.format(model.__name__))
    raise ValueError(f'Model {name} not defined')
  ModelClass = mdict[name]

  return ModelClass
