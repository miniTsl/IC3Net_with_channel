from ast import arg
import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action

from channel import Channel

class CommNetMLP(nn.Module):
    """
    MLP based CommNet. Uses communication vector to communicate info between agents
    """
    def __init__(self, args, num_inputs):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            MLP {object} -- Self \n
            args {Namespace} -- Parse args namespace \n
            num_inputs {number} -- Environment observation dimension for per agent: 61 for tf_medium
        """

        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents # 10
        self.hid_size = args.hid_size   # 128
        self.comm_passes = args.comm_passes # 1
        self.recurrent = args.recurrent # true

        self.continuous = args.continuous   # false
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            # 动作函数
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o)
                                        for o in args.naction_heads])   # naction_heads=[2]
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2    # 0.2

        # Mask for communication
        if self.args.comm_mask_zero:    # false
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:   # 除了主对角线全是1, tensor[10,10]
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # Since linear layers in PyTorch now accept * as any number of dimensions
        # between last and first dim, num_agents dimension will be covered.
        # The network below is function r in the paper for encoding
        # initial environment stage
        # 最开始的编码层
        self.encoder = nn.Linear(num_inputs, args.hid_size)
    
        if args.recurrent:  # true
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)

        if args.recurrent:  # true
            self.init_hidden(args.batch_size)   # 500
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)   # LSTM

        else:
            if args.share_weights:  # false
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module
                                                for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                                for _ in range(self.comm_passes)])
        # else:
            # raise RuntimeError("Unsupported RNN type.")

        # Our main function for converting current hidden state to next state
        if args.share_weights:  # false
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module
                                            for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size)
                                            for _ in range(self.comm_passes)])

        # initialise weights as 0
        if args.comm_init == 'zeros':   # uniform
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

        # 值函数
        self.value_head = nn.Linear(self.hid_size, 1)

        # 信道
        self.channel = Channel()
        
    def get_agent_mask(self, batch_size, info):
        n = self.nagents    # 10

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])   # numpy --> tensor
            agent_mask = self.channel.send(agent_mask)  # tensor --> tensor (after channel, few will survive)
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n)   # [1,1,10]
        agent_mask = agent_mask.expand(batch_size, n, n).unsqueeze(-1)  # [1,10,10,1]

        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent: # true
            x, extras = x
            x = self.encoder(x)

            if self.args.rnn_type == 'LSTM':    # true
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
        else:
            x = self.encoder(x)
            x = self.tanh(x)
            hidden_state = x

        return x, hidden_state, cell_state


    def forward(self, x, info={}):
        # TODO: Update dimensions
        """Forward function for CommNet class, expects state, previous hidden
        and communication tensor.

        Arguments:
            x {list}
                0: obs of agents tensor[B x N x num_inputs]
                1: tuple
                    0: hidden_state tensor[ N x hid_size]
                    1: cell_state tensor[N x hid_size]     
            B: Batch Size: Normally 1 in case of episode
            N: number of agents
            num_inputs : 61 for tf_medium
            
            
            prev_hidden_state {tensor} -- Previous hidden state for the networks in
            case of multiple passes (B x N x hid_size) \n
            comm_in {tensor} -- Communication tensor for the network. (B x N x N x hid_size)

        Returns:
            tuple -- Contains
                next_hidden {tensor}: Next hidden state for network
                comm_out {tensor}: Next communication tensor
                action_data: Data needed for taking next action (Discrete values in
                case of discrete, mean and std in case of continuous)
                v: value head
        """

        x, hidden_state, cell_state = self.forward_state_encoder(x) # tensor[1,10,128], tensor[10,128], tensor[10,128]

        batch_size = x.size()[0]    # 1
        n = self.nagents    # 10

        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # Hard Attention - action whether an agent communicates or not
        if self.args.hard_attn: # false
            comm_action = torch.tensor(info['comm_action'])
            comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
            # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
            agent_mask *= comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)   # tensor[1,10,10,1] alive变为列向量

        for i in range(self.comm_passes):
            # Choose current or prev depending on recurrent
            comm = hidden_state.view(batch_size, n, self.hid_size) if self.args.recurrent else hidden_state # [10,128] --> [1,10,128]

            # Get the next communication vector based on next hidden state
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)   # [1,10,128] --> [1,10,1,128] --> [1,10,10,128]

            # Create mask for masking <self communication>
            mask = self.comm_mask.view(1, n, n) # tensor[1,10,10]
            mask = mask.expand(comm.shape[0], n, n) # [1, 10, 10]
            mask = mask.unsqueeze(-1)   # [1,10,10,1]

            mask = mask.expand_as(comm) # [1,10,10,128]
            comm = comm * mask  # [1,10,10,128]
            
            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Mask comm_in
            # Mask communcation <from dead agents>
            comm = comm * agent_mask
            # Mask communication <to dead agents>
            comm = comm * agent_mask_transpose

            # Combine all of C_j for an ith agent which essentially are h_j
            comm_sum = comm.sum(dim=1)  # [1,10,10,128] --> [1,10,128]
            c = self.C_modules[i](comm_sum)


            if self.args.recurrent:
                # skip connection - combine comm. matrix and encoded input for all agents
                inp = x + c

                inp = inp.view(batch_size * n, self.hid_size)
                
                # 估计LSTM自带非线性函数，所以这里就不用手动添加了
                output = self.f_module(inp, (hidden_state, cell_state))

                hidden_state = output[0]
                cell_state = output[1]

            else: # MLP|RNN
                # Get next hidden state from f node
                # and Add skip connection from start and sum them
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # 利用最后一步通信的 h计算值函数
        value_head = self.value_head(hidden_state)
        h = hidden_state.view(batch_size, n, self.hid_size)
        # 计算动作概率
        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            # will be used later to sample
            action = (action_mean, action_log_std, action_std)
        else:
            # discrete actions
            # 返回的是action的概率[1, 10, 2]，选择0或者1
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        if self.args.recurrent:
            return action, value_head, (hidden_state.clone(), cell_state.clone())
        else:
            return action, value_head

    def init_weights(self, m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(0, self.init_std)

    def init_hidden(self, batch_size):
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

