from itertools import count
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
        self.alive_index = torch.nonzero(agent_mask).squeeze()   # index of alive agents
        for x in self.alive_index:
            self.windows[x] = 3     # initial window width: 3
        who_sent, who_failed = self.step()
        
        agent_mask[self.alive_index[who_failed]] = 0 # set who_failed slinet so that they are dead???
    
    def step(self):
        '''
        input:
            n_agent: number of agents to send info
        output:
            who_sent: which agents succeeded
            who_failed: which agents failed
        '''
        phase = self.comm_pahse
        not_sent_yet = self.alive_index.tolist()
        waits = {}
        who_sent = []
        for x in not_sent_yet:
                waits[x] = torch.randint(0, self.windows[x])    # generate random wait time
        while phase > 0:
            if len(not_sent_yet) == 0:  # all sent, done
                return who_sent, []
            
            min_wait = min(waits.values())
            if phase < (min_wait+ self.frame_slots):    # if there is not enough time to transmit another message
                return who_sent, list(set(self.alive_index.tolist()) - set(who_sent))
            else:
                if list(waits.values()).count(min_wait) == 1:   # no collision. the message can be sent successfully
                    to_del = 0
                    for key in waits.keys():
                        if waits[key] == min_wait:
                            to_del = key
                            not_sent_yet.remove(key)    # agent key is sent
                            who_sent.append(key)
                        else:
                            waits[key] -= min_wait
                    del waits[to_del]   # agent key is sent
                
                else:   # collision happens
                    to_change = []
                    for key in waits.keys():
                        if waits[key] == min_wait:
                            to_change.append(key)
                        else:
                            waits[key] -= min_wait
                    for x in to_change:
                        self.windows[x] = 2*(self.windows[x]+1) - 1     # BEB
                        waits[x] = torch.randint(0, self.windows[x])    # regenerate random wait time   
    