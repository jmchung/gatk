import numpy as np
import theano
import theano.tensor as tt

# the following dtype will be used for all float numpy ndarrays
floatX = theano.config.floatX

# big uint dtype
big_uint = np.uint64

# small uint type
small_uint = np.uint16

TheanoVector = tt.TensorType(floatX, (False,))
TheanoMatrix = tt.TensorType(floatX, (False, False))
TheanoTensor3 = tt.TensorType(floatX, (False, False, False))
TensorSharedVariable = theano.tensor.sharedvar.TensorSharedVariable
TheanoScalar = tt.TensorType(floatX, ())
