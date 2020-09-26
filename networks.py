import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Flatten, Conv2D, Reshape, UpSampling2D, concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.callbacks import TensorBoard

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

latent_dim = 32
action_dim = 1

def CAE_critic() :  
    input_img = Input(shape=(64, 32, 3))
    input_act = Input(shape=(action_dim,),name='input_act')
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    
    encoded = Dense((32), activation='relu', name = 'encoded')(x)
    encoded_with_act = concatenate([encoded, input_act])
    
    x = Dense((64*32*4),activation='relu')(encoded)
    x = Reshape((64,32,4))(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    h_val = Dense((20),activation='relu')(encoded_with_act)
    output_val = Dense((1))(h_val)
    critic = Model(inputs = [input_img,input_act], outputs = output_val)
    encoder = Model(input_img, encoded)
    autoencoder_critic = Model([input_img, input_act], [decoded, output_val])
    adam = Adam(lr = 0.001)
    autoencoder_critic.compile(optimizer=adam, loss=['mse','mse'])
    
    return autoencoder_critic, encoder, critic

def CAE_critic_latent() :  
    input_img = Input(shape=(64, 32, 3))
    input_act = Input(shape=(action_dim,),name='input_act')
    encoded_input = Input(shape=(latent_dim,))
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    
    encoded = Dense((32), activation='relu', name = 'encoded')(x)
    encoded_with_act = concatenate([encoded, input_act], name = 'encoded_with_act')
    
    x = Dense((64*32*4),activation='relu')(encoded)
    x = Reshape((64,32,4))(x)
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    h_val = Dense((20),activation='relu', name = 'h_val')(encoded_with_act)
    output_val = Dense((1), name = 'output_val')(h_val)
    encoder = Model(input_img, encoded)
    autoencoder_critic = Model([input_img, input_act], [decoded, output_val])
    
    deco = autoencoder_critic.get_layer('encoded_with_act')([encoded_input, input_act])
    deco = autoencoder_critic.get_layer('h_val')(deco)
    deco = autoencoder_critic.get_layer('output_val')(deco)
    
    critic = Model(inputs = [encoded_input, input_act], outputs = deco)
    adam = Adam(lr = 0.001)
    autoencoder_critic.compile(optimizer=adam, loss=['mse','mse'])
    critic.compile(optimizer=adam, loss='mse')
    
    return autoencoder_critic, encoder, critic

def actor():
    input_ = Input(shape=(32,))
    x = Dense((20),activation='tanh')(input_)
    output = Dense((action_dim),activation='tanh')(x)
    
    actor = Model(input_,output)
    adam = Adam(lr = 0.0001)
    actor.compile(optimizer=adam, loss='mse')
    
    return actor   

def world():
    input_ = Input(shape=(latent_dim+action_dim,))
    x = Dense((20),activation='tanh',name='hid')(input_)
    output1 = Dense((latent_dim),activation='sigmoid',name='out1')(x) # predicted state
    output2 = Dense((1),activation='tanh',name='out2')(x) # predicted reward
    
    world = Model(input_, [output1, output2])
    world.compile(optimizer='adam', loss=['mse','mse'])
    
    return world 

def rolledout_world(depth):
    inputs = []
    outputs = []
    
    inputs.append(Input(shape=(latent_dim+action_dim,), name='input'+str(1)))
    for i in range(depth-1):
        inputs.append(Input(shape=(action_dim,),name='input'+str(i+2)))
        
    x1 = Dense((20),activation='tanh',name='hid'+str(1))(inputs[0])
    output_s = Dense(latent_dim,activation='sigmoid',name='output_s'+str(1))(x1) # predicted state (t+1)
    outputs.append(Dense(1, activation='tanh',name='output'+str(1))(x1)) # predicted reward (t+1)
    
    for i in range(depth-1):
        x = concatenate([output_s, inputs[i+1]])
        x2 = Dense((20),activation='tanh',name='hid'+str(i+2))(x)
        output_s = Dense(latent_dim, activation='sigmoid',name='output_s'+str(i+2))(x2)
        outputs.append(Dense(1, activation='tanh',name='output'+str(i+2))(x2))
    
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer='adam', loss=['mse' for i in range(depth)])
    
    return model       

def mpc(world_models, rolledout_world, state, initial_plan): # Model Predictive Control
    lr, epochs = 0.001, 10 # learning rate and no. of training epochs (optimization iterations)
    _transfer_weights(world_models, rolledout_world)
    num_outputs = len(world_models)
    err_sum = 0.
    
    input_tensors = [rolledout_world.input[i] for i in range(num_outputs)]
    output_tensors = [rolledout_world.output[i] for i in range(num_outputs)]
    
    for i in range(num_outputs):
        err_sum = err_sum + K.square(1.-output_tensors[i])
    loss = 0.5*err_sum # loss_plan
    
    gradients = K.gradients(loss, rolledout_world.input)
    func = K.function(input_tensors, gradients)
    
    for i in range(epochs):
        inputs = []
        inputs.append(np.array([np.concatenate((state,initial_plan[0]))]))
        for j in range(len(initial_plan)-1):
            inputs.append(np.array([initial_plan[j+1]]))
        grad_val = func(inputs)
        
        grad_wrt_a = np.zeros((len(initial_plan),action_dim))
        grad_wrt_a[0] = np.array([grad_val[0][0][latent_dim+action_dim-1]])
        for j in range(len(initial_plan)-1):
            grad_wrt_a[j+1] = grad_val[j+1][0]
        updated_plan = initial_plan+(grad_wrt_a*-lr)
        
        initial_plan = updated_plan
	
    return initial_plan[0] # reuturns the optimal plan's 1st action

def _transfer_weights(world_models, rolledout_world):
    
    for i in range(len(world_models)):
        params_hid = world_models[i].get_layer('hid').get_weights() 
        params_out1 = world_models[i].get_layer('out1').get_weights()
        params_out2 = world_models[i].get_layer('out2').get_weights()
        
        rolledout_world.get_layer('hid'+str(i+1)).set_weights(params_hid)
        rolledout_world.get_layer('output'+str(i+1)).set_weights(params_out2)
        if i<len(world_models)-1:
            rolledout_world.get_layer('output_s'+str(i+1)).set_weights(params_out1)
