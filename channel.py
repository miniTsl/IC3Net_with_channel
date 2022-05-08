import random
import numpy as np
import torch
class Channel:
    '''
    Communication channel class
    '''
    def __init__(self) -> None:
        self.comm_pahse = 60    # length of commu phase, each time slot counts 1
        # self.DIFS = 3
        # self.EIFS = 3
        self.windows = {}
        self.frame_slots = 10
        
    def send(self, agent_mask):
        '''
        input:
            agent_mask: numpy array([10,]), i.e. info['alive_mask']
        output:
            senters: an array like agent_mask, but set failed agents silent
            sent: list of indexes of alive agents in input that will send messages successfully
            failed: list of indexes of alive agents in input that will fail to send messages
        '''
        self.alive_index = (np.nonzero(agent_mask.squeeze())[0]).tolist()  # np.nonzero() returns a tuple of indexes of nonzero elements on each dimension
        # we hope np.array --> list of nonzero element positions
        if len(self.alive_index) == 0:  # all dead
            return agent_mask.copy(), [], []
        else:
            self.windows.clear()    # clear windows
            for x in self.alive_index:
                self.windows[x] = 3     # initial window width: 3
            sent, failed = self.step()
            senters = agent_mask.copy()    # numpy array
            for x in failed:
                senters[x] = 0 # set who_failed slinet so that they are dead
            return senters, sent, failed
    
    def step(self):
        '''
        output:
            who_sent: list of indexes of agents in self.windows that will send messages successfully
            who_failed: list of indexes of agents in self.windows that will fail to send messages
        '''
        phase = self.comm_pahse
        not_sent_yet = self.alive_index.copy()
        waits = {}
        who_sent = []
        for x in not_sent_yet:
            waits[x] = random.randint(0, self.windows[x])    # generate random wait time [low, high]
        while phase > 0:
            if len(not_sent_yet) == 0:  # all sent, done, none failed
                break 
            else:
                min_wait = min(waits.values())
                if phase < (min_wait + self.frame_slots):    # there is not enough time to transmit another message
                    break
                else:
                    phase -= (min_wait + self.frame_slots)
                    if list(waits.values()).count(min_wait) == 1:   # no collision. the message can be sent successfully
                        to_del = 0
                        for key in waits.keys():
                            if waits[key] == min_wait:
                                to_del = key
                                not_sent_yet.remove(key)    # agent key has sent
                                who_sent.append(key)
                            else:
                                waits[key] -= min_wait
                        del waits[to_del]   # agent key has sent
                    
                    else:   # collision happens
                        to_change = []
                        for key in waits.keys():
                            if waits[key] == min_wait:
                                to_change.append(key)
                            else:
                                waits[key] -= min_wait
                        for x in to_change:
                            self.windows[x] = min([15, 2*(self.windows[x]+1) - 1])    # BEB
                            waits[x] = random.randint(0, self.windows[x])    # regenerate random wait time   
        if len(not_sent_yet) == 0:
            return who_sent, []
        else:
            return who_sent, list(set(self.alive_index) - set(who_sent))    # difference set