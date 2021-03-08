import torch
import torch.nn as nn

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self,num_state, num_node, bias = False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node,num_node,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 =nn.Conv1d(num_state,num_state,kernel_size=1,bias=bias)
        self.conv3 = nn.Conv1d(num_node, int(num_node/2), kernel_size=1)


    def forward(self, x):
        '''input:(n, num_state, num_node)'''
        h =self.conv1(x.permute(0,2,1).contiguous()).permute(0,2,1)
        h =h+x
        h = self.conv2(self.relu(h))
        h = self.conv3(h.permute(0,2,1).contiguous()).permute(0,2,1)
        return h




class g_GCN(nn.Module):
    """
    Graph-based Global Reasoning Unit

    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 Conv2d=nn.Conv2d,
                 BatchNorm2d=nn.BatchNorm2d,
                 normalize=False):
        super(g_GCN, self).__init__()
        self.normalize = normalize
        self.num_s =int(2*num_mid)
        self.num_n =int(2*num_mid)
        # reduce dim
        self.conv_state = Conv2d(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = Conv2d(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(self.num_s, self.num_n*2)
        # -------
        # extend dimension
        self.conv_extend = Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
        self.blocker = BatchNorm2d(num_in, eps=1e-04) # should be zero initialized
    def forward(self, x,y):
        '''
        :param x: (n, c, h, w)
        '''

        n= x.size(0)
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = (self.conv_state(x)).view(n, self.num_s, -1)

        y_state_reshaped = (self.conv_state(y)).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)
        y_proj_reshaped = self.conv_proj(y).view(n, self.num_n, -1)


        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        y_rproj_reshaped = y_proj_reshaped
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, y_proj_reshaped.permute(0, 2, 1))
        y_n_state = torch.matmul(y_state_reshaped, x_proj_reshaped.permute(0, 2, 1))

        # CROSS EMBEDDING
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
            y_n_state = y_n_state * (1. / y_state_reshaped.size(2))

        x_n_state= torch.cat([x_n_state,y_n_state],2)
        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)
        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, y_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s,*x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = y + self.blocker(self.conv_extend(x_state))

        return out
