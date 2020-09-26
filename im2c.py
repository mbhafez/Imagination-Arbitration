"""

Intriniscally Motivated Meta-Controller (IM2C) algorithm. 

Link to paper: https://arxiv.org/pdf/2004.08830.pdf

"""

from collections import deque
from parameters import *
from keras.models import load_model, clone_model, save_model
from networks import actor, CAE_critic, rolledout_world, mpc
import random, numpy as np, env
from itm import Map


class IM2C (object):
    
    def __init__(self):
        self.env = env.Env()
        self.actor = actor()
        self.autoencoder_critic, self.encoder, self.critic = CAE_critic()
        self.critic_t = clone_model(self.critic); self.critic_t.set_weights(self.critic.get_weights())
        self.actor_t = clone_model(self.actor); self.actor_t.set_weights(self.actor.get_weights())
        self.memory = deque(maxlen=int(1e+5))
        self.goals = np.loadtxt("goals") # list of random goal postitions
        self.D_MAX = 6
        self.map = Map()

    def compute_initial_plan(self, state, actor): # returns optimistic initial action plan to the MPC to optimize
        depth = 0
        reliable_model = True
        initial_plan = []
        world_models = []
        while depth<self.D_MAX and reliable_model:
            act = actor.predict(np.array([state]), batch_size = 1)[0]
            world = self.map.bmn(state).world
            output = world.predict(np.array([np.concatenate((state, act))]), batch_size = 1)
            next_s, r = output[0][0], output[1][0]
            initial_plan.append(act)
            world_models.append(world)
            state = next_s
            reliable_model = True if self.map.bmn(state).get_learning_progress()>=0 else False
            depth+=1
        
        return depth, initial_plan, world_models

    def remember(self, state, action, total_r, next_state, done):
        self.memory.append((state, action, total_r, next_state, done))
    
    def replay(self):
        batch_size = min(BATCH_SIZE,len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X_c1, X_c2, X_a  = np.zeros((batch_size,64,32,3)), np.zeros((batch_size,1)), []
        Y_c1, Y_c2, Y_a = np.zeros((batch_size,64,32,3)), np.zeros((batch_size,)), []
        
        for i in range(batch_size):
            state, action, total_r, next_state, done = minibatch[i]
            encoded_state = self.encoder.predict(np.array([state]),batch_size=1)[0]
            target0, target1 = state, total_r
            if not done:
                pol_next_t = self.actor_t.predict(np.array([encoded_state]))
                target1 += gamma * self.critic_t.predict([np.array([next_state]), pol_next_t])[0]
            X_c1[i], X_c2[i], Y_c1[i], Y_c2[i] = state, action, target0, target1
            td_err = target1 - self.critic.predict([np.array([state]), np.array([X_c2[i]])], batch_size = 1)[0]
            if td_err>0.:
                X_a.append(encoded_state); Y_a.append(action)
        
        # update critic and actor (off-policy CACLA update)       
        self.autoencoder_critic.fit([X_c1, X_c2], [Y_c1, Y_c2], batch_size=batch_size, epochs=5, verbose=0)
        if len(X_a)>0: self.actor.fit(x=np.asarray(X_a), y=np.asarray(Y_a), batch_size=len(X_a), epochs=15, verbose=0)
        
        # update target networks
        critic_weights, critic_t_weights = self.critic.get_weights(), self.critic_t.get_weights()
        new_critic_target_weights = []
        w_cnt = len(critic_t_weights)
        for i in range(w_cnt):
            new_critic_target_weights.append((tnet_update_rate*critic_weights[i])+((1-tnet_update_rate)*critic_t_weights[i]))
        self.critic_t.set_weights(new_critic_target_weights)
        
        actor_weights, actor_t_weights = self.actor.get_weights(), self.actor_t.get_weights()
        new_actor_target_weights = []
        w_cnt = len(actor_t_weights)
        for i in range(w_cnt):
            new_actor_target_weights.append((tnet_update_rate*actor_weights[i])+((1-tnet_update_rate)*actor_t_weights[i]))
        self.actor_t.set_weights(new_actor_target_weights)

    
    def learn (self, nb_episodes = 10000, steps_per_episode = 50, sim = 1, nb_simulation = 1):
        reliable_planning = False
        steps = np.zeros((nb_episodes,))
        rewards = np.zeros((nb_episodes,)) 
        
        first_obs = self.encoder.predict(np.array([self.env.retrieve(env.getShoulderZ())]), batch_size = 1)[0]
        self.map.nodes[0].refVec = first_obs;
        
        for i_episode in range(nb_episodes):
            total_ext_reward_per_episode = 0.
            goal = self.goals[i_episode%50]
            self.env.reset(goal)
            observation = self.env.retrieve(env.getShoulderZ())
            for t in range(steps_per_episode):
                print "IM2C sim: {}/{} episode: {}/{}. Step: {}/{}".format(sim, nb_simulation, i_episode+1, nb_episodes, t+1, steps_per_episode)
                encoded_obs = self.encoder.predict(np.array([observation]), batch_size = 1)[0]
                reliable_planning =  True if self.map.nodes[self.map.lastVisitedNode].get_learning_progress()>=0 else False
                
                if reliable_planning:
                    depth, initial_plan, world_models = self.compute_initial_plan(encoded_obs, self.actor)
                    action = mpc(world_models, rolledout_world(depth), encoded_obs, initial_plan) # outputs the optimal plan's 1st action
                else:
                    action = self.actor.predict(np.array([encoded_obs]), batch_size = 1)[0]
                    
                action = np.random.normal(action[0], std) # exploration noise
                
                if action>1: 
                    action = 1
                else: 
                    if action<-1: action = -1
                
                clipped_act = action
             
                ready_act = np.multiply(np.array([clipped_act]),[20])
                obs_new, r_ext, done = self.env.step(ready_act)
                encoded_next_obs = self.encoder.predict(np.array([obs_new]), batch_size = 1)[0]
                lp = self.map.adapt(stimulus = encoded_next_obs, next_r = r_ext, prev_state_act = np.concatenate([encoded_obs, ready_act]))
                r_int = -lp
                r_total = r_ext + r_int
                total_ext_reward_per_episode += r_ext 
                
                self.remember(observation, clipped_act, r_total, obs_new, done)                          
                
                observation = obs_new
                if done:
                    break  
            
            print "--updating--"
            self.replay()
                
            steps [i_episode] = t+1
            
            print "--saving networks--"
            try: 
                self.actor.save_weights("actor_W"); self.autoencoder_critic.save_weights("autoencoder_W");
                save_model(self.actor,'actor'); save_model(self.autoencoder_critic, 'autoencoder_critic');
            except IOError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)   

            rewards [i_episode] = total_ext_reward_per_episode
            
            print "--saving results--"
            np.savetxt('rewards',rewards)
            np.savetxt('steps',steps)
        
        self.critic = None; self.critic_t = None; self.actor = None; self.actor_t = None
        
        return rewards, steps 
    
if __name__ == "__main__":
    print('---------Learning started-----------')
    agent = IM2C()
    rewards, steps  = agent.learn() 