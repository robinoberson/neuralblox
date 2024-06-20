import numpy as np


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        occ = data['occ']
        #normals = data['normals']

        points_out = np.zeros((points.shape[0], self.N, 3), dtype=np.float32)
        occ_out = np.zeros((points.shape[0], self.N), dtype=np.float32)
        
        for i in range(points.shape[0]):
            points_i = points[i, :]
            occ_i = occ[i, :]
            
            indices = np.random.randint(points_i.shape[0], size=self.N)
            
            points_out[i, :] = points_i[indices, :]
            occ_out[i, :] = occ_i[indices]
            
        data_out[None] = points_out
        data_out['occ'] = occ_out
        #data_out['normals'] = normals[indices, :]

        return data_out

class SubsamplePointcloudValidation(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data['points']
        occ = data['points.occ']
        
        inputs = data['inputs']
        inputs_occ = data['inputs.occ']
        #normals = data['normals']

        indices_points = np.random.randint(points.shape[0], size=self.N)
        indices_inputs = np.random.randint(inputs.shape[0], size=self.N)
        
        data_out['points'] = points[indices, :]
        data_out['points.occ'] = occ[indices]
        data_out['inputs'] = inputs[indices_inputs, :]
        data_out['inputs.occ'] = inputs_occ[indices_inputs]
        #data_out['normals'] = normals[indices, :]

        return data_out

class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        # points = data[None]
        # occ = data['occ']

        # data_out = data.copy()
        #     idx = np.random.randint(points.shape[0], size=self.N)
        #     data_out.update({
        #         None: points[idx, :],
        #         'occ':  occ[idx],
        #     })
        data_out = data.copy()
        points = data[None]
        occ = data['occ']
        #normals = data['normals']

        points_out = np.zeros((points.shape[0], self.N, 3), dtype=np.float32)
        occ_out = np.zeros((points.shape[0], self.N), dtype=np.float32)
        
        for i in range(points.shape[0]):
            points_i = points[i, :]
            occ_i = occ[i, :]
            
            indices = np.random.randint(points_i.shape[0], size=self.N)
            
            points_out[i, :] = points_i[indices, :]
            occ_out[i, :] = occ_i[indices]
            
        data_out[None] = points_out
        data_out['occ'] = occ_out
        #data_out['normals'] = normals[indices, :]

        return data_out