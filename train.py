import sys

# append the rbm-recommendation directory path to PYTHONPATH
sys.path.append('/mnt/output/home/rbm-recommendation')


import pandas as pd
import numpy as np
import argparse

import time

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse, sys
from ContrastiveDivergence.CD import CDOptimizer
import horovod.tensorflow as hvd

parser = argparse.ArgumentParser()
parser.add_argument('--gbz', help='global batch size')
parser.add_argument('--epochs', help='No. of epochs to train the model')
parser.add_argument('--hidden', help='number of neurons in hidden layer')
parser.add_argument('--data_dir',help='path to the data file')
parser.add_argument('--output_dir',help='directory where the models weights, bias hidden and bias visible files will be stored')

args = parser.parse_args()

if not (args.gbz and args.epochs and args.hidden):
    os.exit('usage: python train.py --hidden 20 --epochs 10 --gbz 200')

def get_batch(start, end, group):
    trX = []
    count = 0
    for userID, curUser in group:
        count += 1
        if count<start:
            continue
        else:
            if count>=end:
                break
                
            temp = [0]*(max(item_id_unique)+1)
            for num, review in curUser.iterrows():
                temp[int(review['item_id'])] = int(review['rating_train'])
            trX.append(temp)            
            
    return trX

def broadcast_to(tensor, shape):
  return tensor + tf.zeros(dtype=tensor.dtype, shape=shape)

def one_hot_coding(rating, scale):
    
    result = [0]*scale
    
    if rating == 0:
        return result
    else:
        result[rating-1] = 1
        return result
    
def coding_a_batch(aBatch, scale):
    dim0, dim1 = aBatch.shape
    dim3 = scale
    
    temp1 = []
    for i in range(0, dim0):
        temp2 = []
        for j in range(0, dim1):
            rating = aBatch[i][j]
            rating_coded = one_hot_coding(rating, scale)
            temp2.append(rating_coded)
        temp1.append(temp2)
        
    result = np.array(temp1)
    return result


data_raw = pd.read_csv(args.data_dir)


item_id_unique = sorted(data_raw.item_id.unique())
print('****************************************************************')
print('Total number of items: ', len(item_id_unique), max(item_id_unique)+1)
user_id_unique = sorted(data_raw.user_id.unique())
print('Total number of users: ', len(user_id_unique), max(user_id_unique)+1, '\n')



begin_all = time.time()

user_group = data_raw.groupby('user_id')


hvd.init()
print('hvd size: ', hvd.size())


scale = 5
visibleUnits = max(item_id_unique) + 1

hiddenUnits = int(args.hidden)

M = visibleUnits
F = hiddenUnits
K = scale

print('M, F, K: ', M, F, K)



##### Draw tensor graph #####

bv = tf.placeholder(tf.float32, [M, K])

bh = tf.placeholder(tf.float32, [F])

bv_auxiliary = tf.placeholder(tf.float32, [None, M, K])

bh_auxiliary = tf.placeholder(tf.float32, [None, F])

W = tf.placeholder(tf.float32, [M, F, K])


v0 = tf.placeholder(tf.float32, [None, M, K], name='v0')


# Conditional probability for sampling hidden layer
probability_forward = tf.nn.sigmoid(tf.einsum('sik,ijk->sj', v0, W) + bh_auxiliary)

h0 = tf.nn.relu( tf.sign( probability_forward - tf.random_uniform( tf.shape(probability_forward) ) ) )


# Conditional probability for sampling visiable layer
numerator = tf.exp( bv_auxiliary + tf.einsum('sj,ijk->sik', h0, W) )

temp = tf.reduce_sum(numerator, axis=2)

denominator1 = tf.expand_dims(temp, axis=2)

denominator2 = broadcast_to(denominator1, [tf.shape(numerator)[0], tf.shape(numerator)[1], K])

probability_backword = tf.truediv(numerator, denominator2)

v1 = tf.nn.relu( tf.sign( probability_backword - tf.random_uniform(tf.shape(probability_backword)) ) )


# Conditional probability for sampling hidden layer
probability_forward2 = tf.nn.sigmoid(tf.einsum('sik,ijk->sj', v1, W) + bh_auxiliary)

h1 = tf.nn.relu( tf.sign( probability_forward2 - tf.random_uniform( tf.shape(probability_forward2) ) ) )



sess = tf.Session()
# Learning rate
alpha = 0.001
opt = CDOptimizer()
opt = hvd.DistributedOptimizer(opt)

grads = opt.compute_gradients(v0, h0, v1, h1)

CD = grads[0][0]
g_bv = grads[1][0]
g_bh = grads[2][0]


# Create methods to update the weights and biases
update_w = W + alpha * CD

bv_auxiliary_extraction = tf.reduce_mean(bv_auxiliary, axis=0)

update_bv = bv_auxiliary_extraction + alpha * g_bv

bh_auxiliary_extraction = tf.reduce_mean(bh_auxiliary, axis=0)

update_bh = bh_auxiliary_extraction + alpha * g_bh



""" Initialize our Variables with Zeroes using Numpy Library """
# Current weight
cur_w = np.zeros([M, F, K], np.float32)
# Current visible unit biases
cur_bv = np.zeros([M, K], np.float32)
# Current hidden unit biases
cur_bh = np.zeros([F], np.float32)
# Previous weight
prv_w = np.zeros([M, F, K], np.float32)
# Previous visible unit biases
prv_bv = np.zeros([M, K], np.float32)
# Previous hidden unit biases
prv_bh = np.zeros([F], np.float32)


sess.run(tf.global_variables_initializer())


# Training RBM 
epochs = int(args.epochs)
gbz = int(args.gbz)
batchsize = int(gbz/hvd.size())
print('\nTraining........................ batch=%d, hvd '%(int(batchsize)), hvd.rank(), '\n')


numberOfRecords = len(user_id_unique)
stepsPerEpoch = int(numberOfRecords/gbz)+1

def get_data_index_template(gbz, batchsize):
    data_index = {}
    for i in range(0, hvd.size(), 1):
        if i<hvd.size()-1:
            begin = 0+i*batchsize
            end = begin+batchsize
        else:
            begin = 0+i*batchsize
            end = gbz
        
        data_index[i]=[begin, end]
        
    return data_index




########## Training ##########
for i in range(epochs):
    
    for k in range(0, stepsPerEpoch, 1):
        
        if k<stepsPerEpoch-1:
            data_index = get_data_index_template(gbz, batchsize)
            begin = data_index[hvd.rank()][0]+gbz*k
            end = data_index[hvd.rank()][1]+gbz*k 
        else:
            gbz_last = numberOfRecords - (stepsPerEpoch-1)*gbz
            batchsize_last = int(gbz_last/hvd.size())
            
            if batchsize_last<1:
                print('Last global batch size is less than hvd.size(); so it is dropped.')
                continue
            else:
                data_index = get_data_index_template(gbz_last, batchsize_last)
                begin = data_index[hvd.rank()][0]+gbz*k
                end = data_index[hvd.rank()][1]+gbz*k
                
        ###########################
        # Training
        ###########################
        t1 = time.time()
        batch = get_batch(begin, end, user_group)
        t2 = time.time()
#         print('Time for preparing data: ', t2-t1)
        
        batch_coded = coding_a_batch(np.array(batch), K)
        batch_coded = batch_coded.astype(float)
        batch_coded = np.float32(batch_coded)
        
        if k == 1:
            prv_bh_auxiliary = np.zeros([batch_coded.shape[0], bh.shape[0]], np.float32)
            prv_bv_auxiliary = np.zeros([batch_coded.shape[0], bv.shape[0], bv.shape[1]], np.float32)
        else:
            prv_bh_auxiliary = np.zeros([batch_coded.shape[0], bh.shape[0]], np.float32)
            prv_bv_auxiliary = np.zeros([batch_coded.shape[0], bv.shape[0], bv.shape[1]], np.float32)
            
            temp = []
            for kk in range(0, batch_coded.shape[0], 1):
                temp.append(list(prv_bh))            
            prv_bh_auxiliary = np.array(temp)
            
            temp = []
            for kk in range(0, batch_coded.shape[0], 1):
                temp.append(list(prv_bv))
            prv_bv_auxiliary = np.array(temp)
            
            
        cur_w, cur_bh, cur_bv = sess.run([update_w, update_bh, update_bv], feed_dict={v0: batch_coded, W: prv_w, bh_auxiliary: prv_bh_auxiliary, bv_auxiliary: prv_bv_auxiliary})
        
        prv_w = cur_w
        prv_bh = cur_bh
        prv_bv = cur_bv
        
        
        
    print("Epoch %d done!"%(int(i)+1))
    
print('\nTraining finished.\n')





# Check if we get 0 solutions:
print(np.sum(prv_w))
print(np.sum(prv_bv))
print(np.sum(prv_bh))

# Hidden layer biases:
print(prv_bh)





W_learned = cur_w.ravel()
bv_learned = cur_bv
bh_learned = cur_bh
# Save learned weights and biases:
if hvd.rank() == 0:
    np.savetxt(args.output_dir+'/rbm_w_bz_%d_epochs_%d_proc_%d_shape_%d_%d_%d_F_%d.txt'%(int(gbz), int(epochs), int(hvd.size()), int(M), int(F), int(K), int(F)), W_learned, fmt='%.10f')
    np.savetxt(args.output_dir+'/rbm_bv_bz_%d_epochs_%d_proc_%d_F_%d.txt'%(int(gbz), int(epochs), int(hvd.size()), int(F)), bv_learned, fmt='%.10f')
    np.savetxt(args.output_dir+'/rbm_bh_bz_%d_epochs_%d_proc_%d_F_%d.txt'%(int(gbz), int(epochs), int(hvd.size()), int(F)), bh_learned, fmt='%.10f')
    

end_all = time.time()

print('Job on process %d done! Over all time cost: %.4f.'%(hvd.rank(), end_all - begin_all))
print('****************************************************************')
print('****************************************************************\n')
