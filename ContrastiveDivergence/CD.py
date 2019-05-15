import tensorflow as tf
base = tf.train.Optimizer


# CDOptimizer for K-way softmax
class CDOptimizer(base):
    
    def __init__(self):
        pass
    
    # v0 is visible layer states for a batch of training data;
    # v0, v1 are 3d tensors; h0, h1 are 2d tensors (matrices);
    def compute_gradients(self, v0, h0, v1, h1):
                  
        cross1 = tf.einsum('sik,sj->sijk', v0, h0)
        w_pos_grad = tf.reduce_mean(cross1, axis=0)
        
        cross2 = tf.einsum('sik,sj->sijk', v1, h1)
        w_neg_grad = tf.reduce_mean(cross2, axis=0)
                  
        CD = w_pos_grad - w_neg_grad
                  
        g_bv = tf.reduce_mean(v0, axis=0) - tf.reduce_mean(v1, axis=0)
        g_bh = tf.reduce_mean(h0, axis=0) - tf.reduce_mean(h1, axis=0)                  
        
        return [(CD, 'w'), (g_bv, 'bv'), (g_bh, 'bh')]
        