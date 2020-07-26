import os
import warnings
import tensorflow as tf
from models.hgate import HGATE_tranductive
from utils import process
from utils.Classification_Clustering import my_LogisticRegression
import scipy.sparse as sp

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


dataset = 'weibo'
direct=os.getcwd()
nb_epochs = 200
lr = 0.0001  # learning rate
hid_units = [512,512]  # numbers of hidden units per each attention head in each node-level layer
model = HGATE_tranductive # transductive learning
lam=1 # proportion parameter of the graph structure reconstruction loss.

print('model: ' + str(model))
print('dataset: ' + dataset)
print('----- hyperparams -----')
print('learning rate: ' + str(lr))
print('node-level layer numbers: ' + str(len(hid_units)))
print('hidden units per node-level layer: ' + str(hid_units))
print('lambda：'+str(lam))

print('---------- data --------')
truefeatures,adj_list, fea_list, Y,idx_train_random_20,idx_test_random_20,idx_train_random_50,idx_test_random_50,idx_train_random_80,idx_test_random_80= process.load_transductive_data_weibo(path=direct)

nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = Y.shape[1]

adj_listcoo = [sp.coo_matrix(adj) for adj in adj_list]
edgecount=[len(adj.row) for adj in adj_listcoo]


with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,shape=(nb_nodes, ft_size),name='ftr_in_{}'.format(i))for i in range(len(fea_list))]
        adj_in_list = [tf.compat.v1.placeholder(dtype=tf.float32,shape=( nb_nodes, nb_nodes),name='adj_in_{}'.format(i))for i in range(len(adj_list))]

    final_embedding, nodefeature, att_val = model.inference(ftr_in_list,adj_in_list,hidden_dims=hid_units,feature_dim=ft_size)

    attributes_loss = tf.sqrt(tf.reduce_sum(tf.reduce_sum(tf.pow(truefeatures - nodefeature, 2))))
    structure_loss=model.compute__structure_loss(final_embedding, adj_in_list, len(adj_list), edgecount)
    loss = attributes_loss + structure_loss * lam

    train_op = model.optimize(loss, lr)
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),tf.compat.v1.local_variables_initializer())

    print('*************** session begin *****************')
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init_op)
        for epoch in range(nb_epochs):
            fd = {i: d for i, d in zip(ftr_in_list, fea_list)}
            fd1 = {i: d for i, d in zip(adj_in_list, adj_list)}
            fd.update(fd1)

            _, loss_value, final_embedding_, str_loss_value, attribute_loss_value = sess.run([train_op, loss, final_embedding, structure_loss, attributes_loss], feed_dict=fd)
            print('epoch : %d,  loss = %.5f,  str_loss=%.5f,  att_loss=%.5f' % (epoch, loss_value, str_loss_value,attribute_loss_value))

            for ii in range(len(final_embedding_)):
                if ii==0:
                    node_embedding=final_embedding_[ii]
                else:
                    node_embedding+=final_embedding_[ii]

            my_LogisticRegression(node_embedding, idx_train_random_80, idx_test_random_80, Y)
            print('**********************************************************************************')

        sess.close()