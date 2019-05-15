import numpy as np
import horovod.tensorflow as hvd
import pandas as pd
from sklearn.metrics import mean_squared_error
import sys
import argparse

# append the rbm-recommendation directory path to PYTHONPATH
sys.path.append('/mnt/output/home/rbm-recommendation')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',help='path to the data file')
parser.add_argument('--weights_file',help='path to the weights file')
parser.add_argument('--bias_hidden',help='path to the hidden unit biases file')
parser.add_argument('--bias_visible',help='path to the visible unit biases file')
parser.add_argument('--output_dir',help='directory to store evaluation output')

args = parser.parse_args()


def dot_product(v, W):
    
    return np.sum(v*W)

##############################################################################
def one_hot_coding(rating, scale):
    
    result = [0]*scale
    
    if rating == 0:
        return result
    else:
        result[rating-1] = 1
        return result
    
def get_rating_vector(user_id, user_group, q, k):
    for user, user_info in user_group:
        if user!=user_id:
            continue
        else:
            temp = [0]*(max(item_id_unique)+1)
            for num, review in user_info.iterrows():
                temp[int(review['item_id'])] = int(review['rating_train'])
                
        temp[q] = k
                
        return temp
    
def coding_a_rating_vector(rating_vector, scale):
  
    temp = []
    for j in range(0, len(rating_vector)):
        rating = rating_vector[j]
        rating_coded = one_hot_coding(rating, scale)
        temp.append(rating_coded)        
        
    result = np.array(temp)
    return result   

def compute_log_score(q, k, W, bv, bh, V):
    v_qk = 1
    term0 = v_qk*bv[q][k]
    
    temp = np.einsum('il,ijl->j', V, W)
    temp2 = 1 + np.exp(temp + bh)
    temp3 = np.log(temp2)
    
    term1 = np.sum(temp3)
    
    log_score = term0 + term1
    
    return log_score    

def get_predicted_rating(odds):
    return np.argmax(odds)+1


#####################################################################################
def evaluate_error(rating_test, rating_test_predicted):
    
    RMSE = np.sqrt(mean_squared_error(rating_test, rating_test_predicted))
    
    percentage = 1.0 * sum(rating_test==rating_test_predicted)/len(rating_test)
    
    return RMSE, percentage


hvd.init()


data_raw = pd.read_csv(args.data_dir)


item_id_unique = sorted(data_raw.item_id.unique())
print('****************************************************************')
print('Total number of items: ', len(item_id_unique), max(item_id_unique)+1)
user_id_unique = sorted(data_raw.user_id.unique())
print('Totle number of users: ', len(user_id_unique), max(user_id_unique)+1, '\n')


user_group = data_raw.groupby('user_id')


W = np.loadtxt(args.weights_file, dtype=float)
bv = np.loadtxt(args.bias_visible, dtype=float)
bh = np.loadtxt(args.bias_hidden, dtype=float)



W = W.reshape((bv.shape[0], int(W.shape[0]/bv.shape[0]/bv.shape[1]), bv.shape[1]))



W = np.float32(W)
bv = np.float32(bv)
bh = np.float32(bh)



print(W.shape, bv.shape, bh.shape)



data_test = data_raw[data_raw['flag_test']==1]
print(data_test.head())
print(data_test.shape)



scale = 5
p1 = []
p2 = []
p3 = []
p4 = []
p5 = []

delta = int( len(user_id_unique)/hvd.size() )
delta = 1

user_id_min = 190 + hvd.rank()*delta
rmse_rank = str(user_id_min)
user_id_max = user_id_min + delta

print('debug: ', user_id_min, user_id_max)


for record_id, record in data_test.iterrows():
    if user_id_min<=record['user_id'] and record['user_id']<user_id_max:
        
        user_id = int(record['user_id'])
        item_id = int(record['item_id'])
        
        
        expected_rating = 0
        rating_vector = get_rating_vector(user_id, user_group, item_id, expected_rating)
        rating_matrix = coding_a_rating_vector(rating_vector=rating_vector, scale=scale)
        log_score = compute_log_score(item_id, expected_rating, W, bv, bh, rating_matrix)
        p1.append(log_score)                         
            
            
            
        expected_rating = 1
        rating_vector = get_rating_vector(user_id, user_group, item_id, expected_rating)
        rating_matrix = coding_a_rating_vector(rating_vector=rating_vector, scale=scale)
        log_score = compute_log_score(item_id, expected_rating, W, bv, bh, rating_matrix)
        p2.append(log_score)                         
            
            
            
        expected_rating = 2
        rating_vector = get_rating_vector(user_id, user_group, item_id, expected_rating)
        rating_matrix = coding_a_rating_vector(rating_vector=rating_vector, scale=scale)
        log_score = compute_log_score(item_id, expected_rating, W, bv, bh, rating_matrix)
        p3.append(log_score)                         
            
            
            
        expected_rating = 3
        rating_vector = get_rating_vector(user_id, user_group, item_id, expected_rating)
        rating_matrix = coding_a_rating_vector(rating_vector=rating_vector, scale=scale)
        log_score = compute_log_score(item_id, expected_rating, W, bv, bh, rating_matrix)
        p4.append(log_score)                         
            
            
            
        expected_rating = 4
        rating_vector = get_rating_vector(user_id, user_group, item_id, expected_rating)
        rating_matrix = coding_a_rating_vector(rating_vector=rating_vector, scale=scale)
        log_score = compute_log_score(item_id, expected_rating, W, bv, bh, rating_matrix)
        p5.append(log_score)    
        
        
        
        
    else:
        p1.append(-1)
        p2.append(-1)
        p3.append(-1)
        p4.append(-1)
        p5.append(-1)
        continue



data_test['probability_1'] = np.array(p1)
data_test['probability_2'] = np.array(p2)
data_test['probability_3'] = np.array(p3)
data_test['probability_4'] = np.array(p4)
data_test['probability_5'] = np.array(p5)



predicted_rating = []
for record_id, record in data_test.iterrows():
    if user_id_min<=record['user_id'] and record['user_id']<user_id_max:
        odds = np.array([record['probability_1'], record['probability_2'], record['probability_3'], record['probability_4'], record['probability_5']])
        predicted_rating.append(get_predicted_rating(odds))
    else:
        predicted_rating.append(-1)



data_test['rating_test_predicted'] = predicted_rating



data_test = data_test[data_test.rating_test_predicted != -1]
print('debug: data_test size: ', data_test.shape[0]) 


print(data_test.head())


data_test = data_test.loc[:, ['user_id', 'item_id', 'rating_test', 'rating_test_predicted']]



print(data_test.head())



rmse, percentage = evaluate_error(data_test.rating_test.values, data_test.rating_test_predicted.values)
print('For process %d,   rmse=%f, accuracy=%f' %(hvd.rank(), rmse, percentage))




file = args.output_dir+"/rmse_" + rmse_rank + ".csv"

temp = pd.DataFrame({'test_size':np.array([data_test.shape[0]]), 'rmse': np.array([rmse])})
temp.to_csv(file, index=False)

