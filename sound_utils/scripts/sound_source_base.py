class SoundSourceBase:
    def __init__(self, params):
        self.file = params['file']
        self.type = params['type']
        self.x = params['x']
        self.y = params['y']
        self.z = params['z']