#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Title  ：Leach for WSN
@Author ：Kay
@Date   ：2019-09-27
=================================================='''
import numpy as np
import matplotlib.pyplot as plt

#如遇中文显示问题可加入以下代码
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

class WSN(object):
    """ The network architecture with desired parameters """
    xm    = 200  # Length of the yard
    ym    = 200  # Width of the yard
    n     = 100  # total number of nodes
    sink  = None # Sink node
    nodes = None # All sensor nodes set
    # Energy model (all values in Joules)
    # Eelec = ETX = ERX
    ETX = 50 * (10 ** (-9))      # Energy for transferring of each bit:发射单位报文损耗能量:50nJ/bit
    ERX = 50 * (10 ** (-9))      # Energy for receiving of each bit:接收单位报文损耗能量:50nJ/bit
    # Transmit Amplifier types
    Efs = 10 * (10 ** (-12))     # Energy of free space model:自由空间传播模型:10pJ/bit/m2
    Emp = 0.0013 * (10 ** (-12)) # Energy of multi path model:多路径衰减空间能量模型:0.0013pJ/bit/m4
    EDA = 5 * (10 ** (-9))       # Data aggregation energy:聚合能量 5nJ/bit
    f_r = 0.6                    # fusion_rate:融合率 , 0代表完美融合
    # Message
    CM = 32     # 控制信息大小/bit
    DM = 4096   # 数据信息大小/bit
    # computation of do
    do = np.sqrt(Efs / Emp) # 87.70580193070293
    
    # 恶意传感器节点
    m_n = 3 # the number of malicious sensor nodes
    
    # Node State in Network
    n_dead           = 0 # The number of dead nodes
    flag_first_dead  = 0 # Flag tells that the first node died
    flag_all_dead    = 0 # Flag tells that all nodes died
    flag_net_stop    = 0 # Flag tells that network stop working:90% nodes died
    round_first_dead = 0 # The round when the first node died
    round_all_dead   = 0 # The round when all nodes died
    round_net_stop   = 0 # The round when the network stop working
    
    def dist(x, y):
        """ 判断两个节点之间的一维距离 """
        distance = np.sqrt(np.power((x.xm - y.xm), 2) + np.power((x.ym - y.ym), 2))
        return distance
        
    def trans_energy(data, dis):
        if dis > WSN.do:
            energy = WSN.ETX * data + WSN.Emp * data * (dis ** 4)
        else: # min_dis <= do
            energy = WSN.ETX * data + WSN.Efs * data * (dis ** 2)
        return energy
        
    def node_state(r):
        nodes  = WSN.nodes
        n_dead = 0
        for node in nodes:
            if node.energy <= Node.energy_threshold:
                n_dead += 1
                if WSN.flag_first_dead == 0 and n_dead == 1:
                    WSN.flag_first_dead  = 1
                    WSN.round_first_dead = r - Leach.r_empty
        if WSN.flag_net_stop == 0 and n_dead >= (WSN.n * 0.9):
            WSN.flag_net_stop  = 1
            WSN.round_net_stop = r - Leach.r_empty
        if n_dead == WSN.n - 1:
            WSN.flag_all_dead  = 1
            WSN.round_all_dead = r - Leach.r_empty
        WSN.n_dead = n_dead
    
class Node(object):
    """ Sensor Node """
    energy_init      = 0.5 # initial energy of a node
    # After the energy dissipated in a given node reached a set threshold,
    # that node was considered dead for the remainder of the simulation.
    energy_threshold = 0.001
    
    def __init__(self):
        """ Create the node with default attributes """
        self.id      = None # 节点编号
        self.xm      = np.random.random() * WSN.xm
        self.ym      = np.random.random() * WSN.ym
        self.energy  = Node.energy_init
        self.type    = "N" # "N" = Node (Non-CH):点类型为普通节点
        # G is the set of nodes that have not been cluster-heads in the last 1/p rounds.
        self.G       = 0 # the flag determines whether it's a CH or not:每一周期此标志为0表示未被选为簇头，1代表被选为簇头
        self.head_id = None # The id of its CH：隶属的簇, None代表没有加入任何簇
        
    def init_nodes():
        """ Initialize attributes of every node in order """
        nodes = []
        # Initial common node
        for i in range(WSN.n):
            node    = Node()
            node.id = i
            nodes.append(node)
        # Initial sink node
        sink    = Node()
        sink.id = -1
        sink.xm  = 0.5 * WSN.xm # x coordination of base station
        sink.ym  = 50 + WSN.ym # y coordination of base station
        # Add to WSN
        WSN.nodes = nodes
        WSN.sink  = sink
        
    def init_malicious_nodes():
        """ Initialize attributes of every malicious node in order """
        for i in range(WSN.m_n):
            node = Node()
            node.id = WSN.n + i
            WSN.nodes.append(node)
    
    def plot_wsn():
        nodes = WSN.nodes
        n = WSN.n
        m_n = WSN.m_n
        # base station
        sink = WSN.sink
        plt.plot([sink.xm], [sink.ym], 'r^',label="基站")
        # 正常节点
        n_flag = True
        for i in range(n):
            if n_flag:
                plt.plot([nodes[i].xm], [nodes[i].ym], 'b+',label='正常节点')
                n_flag = False
            else:
                plt.plot([nodes[i].xm], [nodes[i].ym], 'b+')
        # 恶意节点
        m_flag = True
        for i in range(m_n):
            j = n + i
            if m_flag:
                plt.plot([nodes[j].xm], [nodes[j].ym], 'kd',label='恶意节点')
                m_flag = False
            else:
                plt.plot([nodes[j].xm], [nodes[j].ym], 'kd')
        plt.legend()
        plt.xlabel('X/m')
        plt.ylabel('Y/m')
        plt.show()
        
class Leach(object):
    """ Leach """
    # Optimal selection probablitity of a node to become cluster head
    p       = 0.1 # 选为簇头概率
    period  = int(1/p) # 周期
    heads   = None # 簇头节点列表
    members = None # 非簇头成员列表
    cluster = None # 簇类字典 :{"簇头1":[簇成员],"簇头2":[簇成员],...}
    r       = 0 # 当前轮数
    rmax    = 5#9999 # default maximum round
    r_empty = 0 # 空轮
    
    def show_cluster():
        fig = plt.figure()
        # 设置标题
        # 设置X轴标签
        plt.xlabel('X/m')
        # 设置Y轴标签
        plt.ylabel('Y/m')
        icon = ['o', '*', '.', 'x', '+', 's']
        color = ['r', 'b', 'g', 'c', 'y', 'm']
        # 对每个簇分类列表进行show
        i = 0
        nodes = WSN.nodes
        for key, value in Leach.cluster.items():
            cluster_head = nodes[int(key)]
            # print("第", i + 1, "类聚类中心为:", cluster_head)
            for index in value:
                plt.plot([cluster_head.xm, nodes[index].xm], [cluster_head.ym, nodes[index].ym], 
                         c=color[i % 6], marker=icon[i % 5], alpha=0.4)
                # 如果是恶意节点
                if index >= WSN.n:
                    plt.plot([nodes[index].xm], [nodes[index].ym], 'dk')
            i += 1
        # 显示所画的图
        plt.show()
        
    def optimum_number_of_clusters():
        """ 完美融合下的最优簇头数量 """
        N = WSN.n - WSN.n_dead
        M = np.sqrt(WSN.xm * WSN.ym)
        d_toBS = np.sqrt((WSN.sink.xm - WSN.xm) ** 2 +
                         (WSN.sink.ym - WSN.ym) ** 2)
        k_opt = (np.sqrt(N) / np.sqrt(2 * np.pi) * 
                 np.sqrt(WSN.Efs / WSN.Emp) *
                 M / (d_toBS ** 2))
        p = int(k_opt) / N
        return p
    
    def cluster_head_selection():
        """ 根据阈值选择簇头节点 """
        nodes   = WSN.nodes
        n       = WSN.n # 非恶意节点
        heads   = Leach.heads = [] # 簇头列表, 每轮初始化为空
        members = Leach.members = [] # 非簇成员成员列表
        p       = Leach.p
        r       = Leach.r
        period  = Leach.period
        Tn      = p / (1 - p * (r % period)) # 阈值Tn
        print(Leach.r, Tn)
        for i in range(n):
            # After the energy dissipated in a given node reached a set threshold, 
            # that node was considered dead for the remainder of the simulation.
            if nodes[i].energy > Node.energy_threshold: # 节点未死亡
                if nodes[i].G == 0: # 此周期内节点未被选为簇头
                    temp_rand = np.random.random()
#                    print(temp_rand)
                    # 随机数低于阈值节点被选为簇头
                    if temp_rand <= Tn:
                        nodes[i].type = "CH" # 此节点为周期本轮簇头
                        nodes[i].G = 1 # G设置为1，此周期不能再被选择为簇头 or (1/p)-1
                        heads.append(nodes[i])
                        # 该节点被选为簇头，广播此消息
                        # Announce cluster-head status, wait for join-request messages
                        max_dis = np.sqrt(WSN.xm ** 2 +  WSN.ym ** 2)
                        nodes[i].energy -= WSN.trans_energy(WSN.CM, max_dis)
                        # 节点有可能死亡
                if nodes[i].type == "N": # 该节点非簇头节点
                    members.append(nodes[i])
        m_n = WSN.m_n
        for i in range(m_n):
            j = n + i
            members.append(nodes[j])
        # 如果本轮未找到簇头
        if not heads:
            Leach.r_empty += 1
            print("---> 本轮未找到簇头！")
            # Leach.cluster_head_selection()
        print("The number of CHs is:", len(heads), (WSN.n - WSN.n_dead))
        return None # heads, members
                
    def cluster_formation():
        """ 进行簇分类 """
        nodes   = WSN.nodes
        heads   = Leach.heads
        members = Leach.members
        cluster = Leach.cluster = {} # 簇类字典初始化
        # 本轮未有簇头，不形成簇
        if not heads:
            return None
        # 如果簇头存在，将簇头id作为cluster字典的key值
        for head in heads:
            cluster[str(head.id)] = [] # 成员为空列表
        # print("只有簇头的分类字典:", cluster)
        # 遍历非簇头节点，建立簇
        for member in members:
            # 选取距离最小的节点
            min_dis = np.sqrt(WSN.xm ** 2 +  WSN.ym ** 2) # 簇头节点区域内的广播半径
            head_id = None
            # 接收所有簇头的信息
            # wait for cluster-head announcements
            member.energy -= WSN.ERX * WSN.CM * len(heads)
            # 判断与每个簇头的距离，加入距离最小的簇头
            for head in heads:
                tmp = WSN.dist(member, head)
                if tmp <= min_dis:
                    min_dis = tmp
                    head_id = head.id
            member.head_id = head_id # 已找到簇头
            # 发送加入信息，通知其簇头成为其成员
            # send join-request messages to chosen cluster-head
            member.energy -= WSN.trans_energy(WSN.CM, min_dis)
            # 簇头接收加入消息
            # wait for join-request messages
            head = nodes[head_id]
            head.energy -= WSN.ERX * WSN.CM
            cluster[str(head_id)].append(member.id) # 添加到出簇类相应的簇头
        # 为簇中每个节点分配向其传递数据的时间点
        # Create a TDMA schedule and this schedule is broadcast back to the nodes in the cluster.
        for key, values in cluster.items():
            head = nodes[int(key)]
            if not values:
                # If there are cluster members, the CH sends schedule by broadcasting
                max_dis = np.sqrt(WSN.xm ** 2 +  WSN.ym ** 2)
                head.energy -= WSN.trans_energy(WSN.CM, max_dis)
                for x in values:
                    member = nodes[int(x)]
                    # wait for schedule from cluster-head
                    member.energy -= WSN.ERX * WSN.CM
#        print(cluster)
        return None # cluster
        
    def set_up_phase():
        Leach.cluster_head_selection()
        Leach.cluster_formation()
        
    def steady_state_phase():
        """ 簇成员向簇头发送数据，簇头汇集数据然后向汇聚节点发送数据 """
        nodes   = WSN.nodes
        cluster = Leach.cluster
        # 如果本轮未形成簇，则退出
        if not cluster:
            return None
        for key, values in cluster.items():
            head     = nodes[int(key)]
            n_member = len(values) # 簇成员数量
            # 簇中成员向簇头节点发送数据
            for x in values:
                member = nodes[int(x)]
                dis    = WSN.dist(member, head)
                member.energy -= WSN.trans_energy(WSN.DM, dis) # 簇成员发送数据
                head.energy   -= WSN.ERX * WSN.DM # 簇头接收数据
            d_h2s = WSN.dist(head, WSN.sink) # The distance of from head to sink
            if n_member == 0: # 如果没有簇成员,只有簇头收集自身信息发送给基站
                energy = WSN.trans_energy(WSN.DM, d_h2s)
            else:
                new_data = WSN.DM * (n_member + 1) # 加上簇头本身收集的数据，进行融合后的新的数据包
                E_DA     = WSN.EDA * new_data # 聚合数据的能量消耗
                if WSN.f_r == 0: # f_r为0代表数据完美融合
                    new_data_ = WSN.DM
                else:
                    new_data_ = new_data * WSN.f_r 
                E_Trans  = WSN.trans_energy(new_data_, d_h2s)
                energy = E_DA + E_Trans
            head.energy -= energy
            
    def leach():
        Leach.set_up_phase()
        Leach.steady_state_phase()
        
    def run_leach():
        for r in range(Leach.rmax):
            Leach.r = r
            nodes   = WSN.nodes
            # 当新周期开始时，G重置为0
            if (r % Leach.period) == 0:
                print("==============================")
                for node in nodes:
                    node.G = 0
            # 当每一轮开始时，节点类型重置为非簇头节点
            for node in nodes:
                node.type = "N"
            Leach.leach()
            WSN.node_state(r)
            if WSN.flag_all_dead:
                print("==============================")
                break
            Leach.show_cluster()

def main():
    Node.init_nodes()
    Node.init_malicious_nodes()
    Node.plot_wsn()
    Leach.run_leach()
    # print("The first node died in Round %d!" % (WSN.round_first_dead))
    # print("The network stop working in Round %d!" % (WSN.round_net_stop))
    # print("All nodes died in Round %d!" % (WSN.round_all_dead))
    
if __name__ == '__main__':
    main()
