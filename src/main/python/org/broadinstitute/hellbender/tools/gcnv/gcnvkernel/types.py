import theano
import theano.tensor as tt

# the following dtype will be used for all float numpy ndarrays
floatX = theano.config.floatX

TheanoVector = tt.TensorType(floatX, (False,))
TheanoMatrix = tt.TensorType(floatX, (False, False))
TheanoTensor3 = tt.TensorType(floatX, (False, False, False))
TensorSharedVariable = theano.tensor.sharedvar.TensorSharedVariable
TheanoScalar = tt.TensorType(floatX, ())
