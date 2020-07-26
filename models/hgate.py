# -*- coding: utf-8 -*-
from models import layers
import tensorflow as tf

#tranductive
class HGATE_tranductive():
    def inference(input_list, adj_list, hidden_dims, feature_dim):
        H_list = []

        hidden_dims = [feature_dim] + hidden_dims

        W = {}
        VS = {}
        VK = {}
        node_att = {}
        logits={}

        for adj_index in range(len(adj_list)):
            W[adj_index] = {}
            VS[adj_index] = {}
            VK[adj_index] = {}
            node_att[adj_index] = {}
            logits[adj_index] = {}

            for i in range(len(hidden_dims) - 1):
                W[adj_index][i] = tf.compat.v1.get_variable("W%s_%s" % (adj_index,i),dtype=tf.float32, shape=(hidden_dims[i], hidden_dims[i + 1]))
                VS[adj_index][i] = tf.compat.v1.get_variable("VS%s_%s" % (adj_index,i),dtype=tf.float32, shape=(hidden_dims[i + 1], 1))
                VK[adj_index][i] = tf.compat.v1.get_variable("VK%s_%s" % (adj_index,i), dtype=tf.float32,shape=(hidden_dims[i + 1], 1))

        # node-level Encoder
        for adj_index in range(len(adj_list)):
            for layer in range(len(hidden_dims) - 1):
                if layer==0:
                    H_list.append(tf.matmul(tf.cast(input_list[adj_index],tf.float32), W[adj_index][layer]))
                else:
                    H_list[adj_index]=tf.matmul(tf.cast(H_list[adj_index], tf.float32), W[adj_index][layer])

                #node-level attention
                node_att[adj_index][layer], logits[adj_index][layer] = layers.node_level_attention(adj_list[adj_index], H_list[adj_index], VS[adj_index][layer], VK[adj_index][layer], layer)

                H_list[adj_index] = tf.matmul(node_att[adj_index][layer], H_list[adj_index])


        #semantic-level encoder
        H_list, semantic_att = layers.semantic_level_attention(H_list)

        # final node representations
        node_embedding=H_list.copy()


        #node-level decoder
        for adj_index in range(len(adj_list)):
            for layer in range(len(hidden_dims) - 2, -1, -1):
                H_list[adj_index] = tf.matmul(H_list[adj_index], W[adj_index][layer], transpose_b=True)
                H_list[adj_index] = tf.matmul(node_att[adj_index][layer], H_list[adj_index])

        #semantic-level decoder
        H_list = tf.transpose(H_list, [1, 0, 2])
        H_list = tf.matmul(semantic_att, H_list)
        H_list = tf.transpose(H_list, [1, 0, 2])

        nodefeature = tf.reduce_sum(H_list, 0)

        return node_embedding, nodefeature, node_att

    def compute__structure_loss(final_embeddings, adj_list,metapathcount, edgecount,pianduna=50000):
        loss=0.0

        for index in range(metapathcount):

            nodeA_nodeBlist = tf.where(tf.not_equal(adj_list[index], 0))
            nodeA_nodeBlist = tf.transpose(nodeA_nodeBlist,[1,0])
            last = 0
            lista = tf.constant([], dtype=tf.float32)
            for i in range(pianduna, edgecount[index], pianduna):
                last = i
                nodeA_emb = tf.nn.embedding_lookup(final_embeddings[index], nodeA_nodeBlist[0][i - pianduna:i])
                nodeB_emb = tf.nn.embedding_lookup(final_embeddings[index], nodeA_nodeBlist[1][i - pianduna:i])
                lista = tf.concat([lista, tf.reduce_sum(nodeA_emb * nodeB_emb, axis=-1)], axis=-1)

            if last<edgecount[index]:
                nodeA_emb = tf.nn.embedding_lookup(final_embeddings[index], nodeA_nodeBlist[0][last:edgecount[index]])
                nodeB_emb = tf.nn.embedding_lookup(final_embeddings[index], nodeA_nodeBlist[1][last:edgecount[index]])
                lista = tf.concat([lista, tf.reduce_sum(nodeA_emb * nodeB_emb, axis=-1)], axis=-1)

            structure_loss = -tf.math.log(tf.sigmoid(lista))
            structure_loss = tf.reduce_sum(structure_loss)
            loss += structure_loss

        return loss

    def optimize(loss, lr):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        gradients, variables = zip(*optimizer.compute_gradients(loss))

        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


#inductive
class HGATE_inductive():
    def inference(input_list, adj_list, hidden_dims, feature_dim,nodesize):
        H_list = []

        hidden_dims = [feature_dim] + hidden_dims

        W = {}
        VS = {}
        VK = {}
        node_att = {}
        logits={}

        for adj_index in range(len(adj_list)):
            W[adj_index] = {}
            VS[adj_index] = {}
            VK[adj_index] = {}
            node_att[adj_index] = {}
            logits[adj_index] = {}

            for i in range(len(hidden_dims) - 1):
                W[adj_index][i] = tf.get_variable("W%s_%s" % (adj_index,i),dtype=tf.float32, shape=(hidden_dims[i], hidden_dims[i + 1]))
                VS[adj_index][i] = tf.get_variable("VS%s_%s" % (adj_index,i),dtype=tf.float32, shape=(hidden_dims[i + 1], 1))
                VK[adj_index][i] = tf.get_variable("VK%s_%s" % (adj_index,i), dtype=tf.float32,shape=(hidden_dims[i + 1], 1))

        # node-level Encoder
        for adj_index in range(len(adj_list)):
            for layer in range(len(hidden_dims) - 1):
                if layer==0:
                    H_list.append(tf.matmul(tf.cast(input_list[adj_index],tf.float32), W[adj_index][layer]))
                else:
                    H_list[adj_index]=tf.matmul(tf.cast(H_list[adj_index], tf.float32), W[adj_index][layer])

                node_att[adj_index][layer], logits[adj_index][layer] = layers.node_level_attention(adj_list[adj_index], H_list[adj_index], VS[adj_index][layer], VK[adj_index][layer], layer)

                H_list[adj_index] = tf.matmul(node_att[adj_index][layer], H_list[adj_index])

        #semantic-level encoder
        H_list, semantic_att = layers.semantic_level_attention_inductive(H_list,hidden_dims[-1],nodesize)


        # final node representations
        node_embedding=H_list.copy()

        #node-level decoder
        for adj_index in range(len(adj_list)):
            for layer in range(len(hidden_dims) - 2, -1, -1):
                H_list[adj_index] = tf.matmul(H_list[adj_index], W[adj_index][layer], transpose_b=True)
                H_list[adj_index] = tf.matmul(node_att[adj_index][layer], H_list[adj_index])

        #semantic-level decoder
        H_list = tf.transpose(H_list, [1, 0, 2])
        H_list = tf.matmul(semantic_att, H_list)
        H_list = tf.transpose(H_list, [1, 0, 2])

        nodefeature = tf.reduce_sum(H_list, 0)

        return node_embedding, nodefeature, node_att

    def compute__structure_loss(final_embeddings, adj_list,metapathcount):
        loss=0.0

        for index in range(metapathcount):
            nodeA_nodeBlist = tf.where(tf.not_equal(adj_list[index], 0))
            nodeA_nodeBlist = tf.transpose(nodeA_nodeBlist,[1,0])
            nodeA_emb = tf.nn.embedding_lookup(final_embeddings[index], nodeA_nodeBlist[0])
            nodeB_emb = tf.nn.embedding_lookup(final_embeddings[index], nodeA_nodeBlist[1])
            lista =  tf.reduce_sum(nodeA_emb * nodeB_emb, axis=-1)

            structure_loss = -tf.math.log(tf.sigmoid(lista))
            structure_loss = tf.reduce_sum(structure_loss)
            loss += structure_loss

        return loss

    def optimize(loss, lr):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op


