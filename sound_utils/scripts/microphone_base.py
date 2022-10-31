class MicrophoneBase:
    def __init__(self, params):
        self.id = params['id']
        self.x = params['x']
        self.y = params['y']
        self.z = params['z']
        self.rx = params['rx']
        self.ry = params['ry']
        self.rz = params['rz']
        self.rw = params['rw']