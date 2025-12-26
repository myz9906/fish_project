'''
The code is desiged to make virtal fish swim with real fish by copying real fish's motion.
Stimulation is triggerred by virtual fish swimming away of the real fish.
Test virtual fish swimming with different visibilies. The speed of virtual fish gets faster when it is invisiable
'''


# import system libraries
import math
import random
import time
import numpy as np
import pandas as pd
import LLvr as vr

# use special conventions ('roslib.load_manifest(...)'
# to import python ros-packages packages 
import roslib
roslib.load_manifest('rospy')
import rospy

# osg_utils contains utility functions for manipulating osg files in the display server
roslib.load_manifest('flyvr')
import flyvr.osg_utils as osg_utils

# to make writing experiements easier we provide serveral libraries, including a common
# base class from which all experiments should derive
roslib.load_manifest('fishvr')
import fishvr.experiment
import fishvr.rosutil


# import RL model of virtual fish 

from RL_module.fish_deployment_interface import FishDeploymentAgent


class VirtualConspecificExperiment(fishvr.experiment.Experiment):

    # the base class takes care of a number of things for you invisibly including
    # command line parsing, recording experimental metadata, recording experimental
    # configuration for reproducibility, etc. You must override the following functions
    #
    #  * condition_switched()
    #  * loop()
    #

    def __init__(self, args):
        # first chain up to the base class to allow it to parse the command line arguments
        # fishvr.experiment.Experiment.__init__(self, args, state=('osg_node_x', 'osg_node_y', 'hidden'))
        fishvr.experiment.Experiment.__init__(self, args, state=('osg_fish1_x', 'osg_fish1_y', 'osg_fish1_z',\
                                                                 'real_fish_x','real_fish_y','real_fish_z',\
                                                                 'orientation','fish1_ori_vr',\
                                                                 'velocity','Stim_Flag','Flag_Start_Letter',\
                                                                 'time')) #not sure this is how i need to add 'visible'

        # state has a very important meaning, it describes things which should be saved in the
        # resulting experimental log csv file. This log file is a magic object to which you assign
        # values

        # initilaise some variables which will be used later to hold confiuguration state
        self._osg_path = ''
        self._node_fish = ['Fish1']
        self._anim_name = ''
        self._should_animate = False
        self._buff_size        = 100        
        self._period_positions = np.zeros((self._buff_size,3)) 
        self._period_distance  = np.zeros((self._buff_size,3))
        # two virtual fish, 
        # 1D: x; 2D: y; 3D: z; 4D: Orientation; 
        # 5D: flag of the virtual fish moving left or right side
        self._virtual_fish     = np.zeros((1,5))   
           
        self._fish_velocity    = 0       
        self._fish_orientation = 0        
        self._fish_acc         = 0
        self._locked_i      = -1 
        self._Flag_round     = 0
        self._rot_theta     = 0
        self.experiment_done  = False
        self._Flag_Control = 100 * 60 * 0.02  # 20 minus control test 这个指的是 先让鱼熟悉20分钟环境 
        self._Flag_Circle = 100 * 60 * 0.02     # 这个是绕圈的时长
        self._pos_last = np.zeros((1,4)) # this variable is defined for the smooth motion function
        # StimulusOSGController makes it easier to control things in osg files
        self._osg_model = osg_utils.StimulusOSGController()

        """Preserve original data loading logic"""
        self.s_vf = pd.read_csv(
            '/home/lab/fishvr/experiments/LL/LLvr/data/period30.csv',
            header=None
        )[0].values    # 这个 s_vf 应该只是预加载的一个数据，self.s_vf 存储的是虚拟鱼在转圈阶段的线速度 (Linear Velocity) 序列
        
        # Load trajectories of the virtual fish
        trajectory_files = ['M_trajectory.csv', 'P_trajectory.csv', 'I_trajectory.csv']
        self.df_trajectories = [pd.read_csv('/home/lab/fishvr/experiments/LL/LLvr/data/'+file) for file in trajectory_files]     # 这个 df_trajectories 很重要，这个是和 virtual fish有关的


    # this function is called after every condition switch. At this point the self.condition
    # object (dict like) contains all those values specified in the configuration yaml file
    # for the current conditon. If you want to know the current condition name, you can
    # use self.condition.name
    def condition_switched(self):
        # when using a filesystem path, make sure to use expand_ros_path(), this saves you some
        # trouble and lets you use ros-relative paths '($find xxx)'
        path = self.condition['osg_filename']
        self._osg_path = fishvr.rosutil.expand_ros_path(path)
        self._osg_model.load_osg(self._osg_path)

        # take some values you defined in the configuration yaml and store them for later use
        # self._anim_name = self.condition['animation_name']
        # self._node_fish[0] = self.condition['Fishs']
        # self._should_animate = self.condition['should_animate']

    # this function is called when there is no current active object being experiemented on, but
    # the tracking system has detected a new object that might be worthwhile performing a trail on
    def should_lock_on(self, obj):
        # for example only lock onto fish < 15cm (in x,y) from the centre of the arena 
        r2 = obj.position.x**2 + obj.position.y**2
        return r2 < (0.15**2)

    # this is the main function that is called after the node is constructed. you can do anything
    # you wish in here, but typically this is where you would dynamically change the virtual
    # environment, for example in response to the tracked object position


    # def CircleMotion(self,Stim_r,Stim_d,i_vf,Angle_difference):
    #     #################################################  main control  #######################################
    #     for i in range(0, 1):
    #         # self._virtual_fish[i,3] = self._virtual_fish[i,3] + np.linalg.norm([vf_x[i_vf+1]-vf_x[i_vf],vf_y[i_vf+1]-vf_y[i_vf]],ord = 2)/Stim_r
    #         self._virtual_fish[i,3] = self._virtual_fish[i,3] + self.s_vf[i_vf]*0.01/Stim_r
    #         if self._virtual_fish[i,3] > 2*np.pi:
    #             self._virtual_fish[i,3] = self._virtual_fish[i,3] - 2*np.pi
    #         self._virtual_fish[i,0] = Stim_r * np.cos(self._virtual_fish[i,3])   # 计算 x, y 的位置
    #         self._virtual_fish[i,1] = Stim_r * np.sin(self._virtual_fish[i,3])
    #         self._osg_model.move_node(self._node_fish[i], x=self._virtual_fish[i,0], y=self._virtual_fish[i,1],
    #                     z=Stim_d, orientation_z = self._virtual_fish[i,3] - Angle_difference,hidden=False)
    #     if len(self._node_fish)>1: # 如果有多条鱼，隐藏其他的
    #         for ii in range(1,len(self._node_fish)+1):
    #             self._osg_model.move_node(self._node_fish[ii], hidden=True)
    #         # print "self._virtual_fish" + `self._virtual_fish`


    '''
    This circle motion is clock-wise, to adapt the shape of lettter. 
    '''

    def CircleMotion(self, Stim_r, Stim_d, i_vf, Angle_difference):
        #################################################  main control  #######################################
        for i in range(0, 1):
            # 1. 【修改点一】角度递减：将 + 改为 -
            # 顺时针运动意味着角度（Theta）随时间减小
            self._virtual_fish[i,3] = self._virtual_fish[i,3] - self.s_vf[i_vf]*0.01/Stim_r
            
            # 2. 【修改点二】边界重置：处理小于 0 的情况
            # 当角度减小到负数时，加 2pi 将其转回正数范围
            if self._virtual_fish[i,3] < 0:
                self._virtual_fish[i,3] = self._virtual_fish[i,3] + 2*np.pi
            
            # 3. 位置计算保持不变 (cos/sin 会自动处理负角度或递减角度)
            self._virtual_fish[i,0] = Stim_r * np.cos(self._virtual_fish[i,3])
            self._virtual_fish[i,1] = Stim_r * np.sin(self._virtual_fish[i,3])
            
            # 4. 【修改点三】鱼头朝向翻转：增加 np.pi (180度)
            # 因为圆周运动反向了，如果不加这个，鱼会变成“倒着游”（尾巴向前）
            # 原本切线方向是 +90度，现在是 -90度，两者相差 180度

            raw_orientation = self._virtual_fish[i,3] - Angle_difference - np.pi

            # 【防御性代码】将其归一化到 [0, 2*pi] 之间
            # Python 的 % 对浮点数也有效，且处理负数结果符合直觉（结果符号与除数相同）
            norm_orientation = raw_orientation % (2 * np.pi)

            self._osg_model.move_node(self._node_fish[i], 
                                    x=self._virtual_fish[i,0], 
                                    y=self._virtual_fish[i,1],
                                    z=Stim_d, 
                                    orientation_z=norm_orientation,  # 传入处理后的安全角度
                                    hidden=False)
                                      
        if len(self._node_fish)>1: 
            for ii in range(1,len(self._node_fish)+1):
                self._osg_model.move_node(self._node_fish[ii], hidden=True)



    def loop(self):
        PA  = vr.ClsLLPhaseANA.PhaseAna()
        vfc = vr.vfcontrol.vfcontrol()

        TempVF = np.zeros_like(self._virtual_fish) 

        i = 0
        Break_range = [5,10]  # break 5~10 seconds

        
        Test_duration  = 100*60*60*2  # as the length of the CSV file generated
        Flag_round_end = self._Flag_round
        ThreDis           = 0.08 # real fish is together with the virutal fish within 0.03 for around 1 second
        Flag_Start_Letter = 0 # at biginning, the virtual fish swim in a circle, 0表示转圈，1表示走轨迹
        Break_duration = np.random.randint(Break_range[0], Break_range[1]) 
        Flag_rotation = True  # A flag to set the rotation of the fish stimulus
        Stim_Flag = -10.0
        Angle_difference = 0
        Stim_Flag_dir = 1   #  If this flag is 1, it means the virtual fish swim along the initial random direction, if it is -1, it means the virtual fish swims backtrack.

        Letters = np.array([0,1,2]) # as ['M','P',"I"]
        iltetter = 0
        C_Letter = 0
        All_Letters = np.random.choice(Letters,size=len(Letters),replace=False) # 将三个字母 打乱顺序 进行洗牌
        diemeter_letter = 0.1 # the diameter of the letter
        
        # Parameters for circling
        Stim_r_circle = 0.08
        Stim_r  = 0.12  # determine the half length of the line path
        Stim_d  = -0.03 # The depth of the virtual fish
        self._virtual_fish[:,0] = [Stim_r]
        self._virtual_fish[:,1] = [0.]
        self._virtual_fish[:,2] = [Stim_d]
        self._virtual_fish[:,3] = [0.5*np.pi]   # 这部分内容好像是用来定义虚拟鱼的初始化位置和状态的
        i_t_vf = 0 # index of trajectory
        i_s_vf = 0 # index of speed     # 这个是用在 circle motion 函数里面的，和我们没啥关系


        # 加载我们的RL 模型
        MODEL_PATH_PROCESS = "checkpoints/letter_M_100_Hz_size_vr_ReachableSet_001/rl_model_9900000_steps.zip"
        # MODEL_PATH_PROCESS = "checkpoints/letter_M_100_Hz_size_vr_ReachableSet_002/rl_model_9900000_steps.zip"
        agent = FishDeploymentAgent(model_path=MODEL_PATH_PROCESS, device='cpu')

        Experiment_Configs = [
            {'rx': 0.01,  'ry': 0.01,  'type': 'box'},    # Config for Round 1
            {'rx': 0.02, 'ry': 0.02, 'type': 'box'}, # Config for Round 2
            {'rx': 0.01,  'ry': 0.01,  'type': 'circle'},    # Config for Round 3
        ]

        config = Experiment_Configs[iltetter]
        agent.update_constraints(config['rx'], config['ry'], config['type'])


        # we can manipulate the VR at whatever rate we wish, but doing so faster than the
        # projector refresh rate (120Hz) would yield little benefit. We use rospy.Rate() to
        # spin the loop at this rate
        r = rospy.Rate(100)
        hidden = True
        while not rospy.is_shutdown():

            x = float(self.object_position.x)         # 真实鱼的位置坐标
            y = float(self.object_position.y)
            z = float(self.object_position.z)

            if self.object_is_locked:
                # place the virtual con specific in a specific place
                # get the orientation of the fish

                self._locked_i += 1
                if self._locked_i > self._buff_size-1:
                    self._period_positions[0:self._buff_size-1] = self._period_positions[1:self._buff_size]     # 这个 buff_size 就是连续 100帧的位置
                    self._period_positions[-1,:] = [x,y,z]  # 把_period_positions 中最后一行的位置赋给 得到的真实鱼位置

                    self._period_distance[0:self._buff_size-1] = self._period_distance[1:self._buff_size]
                    self._period_distance[-1,:] = np.linalg.norm([x-self._virtual_fish[0,0],
                                                                  y-self._virtual_fish[0,1],
                                                                  z-self._virtual_fish[0,2]],ord=2) # 这个是计算虚拟鱼与真实鱼之间的相对距离

                    #################################################  main control  #######################################
                    if self._locked_i < self._Flag_Control: # taking the control test
                        self._virtual_fish = np.zeros((2,5))
                        self._osg_model.move_node(self._node_fish[0], hidden=True) # hide all fish
                        hidden = True

                    #################################################  fish circles  #######################################
                    elif self._locked_i - self._Flag_Control < self._Flag_Circle:
                        self.CircleMotion(Stim_r_circle,Stim_d,i_s_vf,Angle_difference - 0.5*np.pi)
                        i_s_vf += 1
                        if i_s_vf == len(self.s_vf) - 1:   
                            i_s_vf = 0
                        Stim_Flag = -1
                        hidden = False
                        # print "Stim_Flag" + `Stim_Flag`
                    #################################################  main trigger  #######################################
                    elif not self.experiment_done:  # Flag to mark if experiment is done

                        # get speed and acc of real fish in real time
                        # C_Acc = PA.getAcc(self._period_positions[:,0],self._period_positions[:,1])

                        if self._Flag_round < Test_duration: # Test  Test_duration 应该是测试时长，具体数值应该是要和 csv 里的时长对齐
                            
                            if Flag_Start_Letter == 0: # at biginning, the virtual fish swim in a circle
                                self.CircleMotion(diemeter_letter,Stim_d,i_s_vf,-0.5*np.pi)
                                i_s_vf += 1
                                if i_s_vf == len(self.s_vf) - 1:
                                    i_s_vf = 0


                                tmpdis = self._period_distance # _period_distance 存储了buffersize 个时间段的 距离
                                if np.all(tmpdis<ThreDis) :    # ThreDis 说的就是那个判断跟随的条件，之前说的是0.03，但是代码里写的是0.08
                                    Flag_Start_Letter = 1
                                    i_t_vf = 0
                                    Test_duration = self._Flag_round + agent.max_steps  # as the length of the CSV file generated
                                    Stim_Flag           = round(1  + 0.1*C_Letter+0.001*Flag_Start_Letter,12) # 这个 flag 没什么用，暂时不用管他


                                    # 准备进入 RL 控鱼的阶段
                                    init_real_pos = (x, y) # 真实鱼初始坐标
                                    init_virtual_pos = (self._virtual_fish[0,0], self._virtual_fish[0,1]) # 虚拟鱼初始坐标

                                    rf_position_ang = np.arctan2(y,x) # 把真实鱼的朝向作为字母角度变换的数据传进去

                                    letter_angle = (rf_position_ang+np.pi) % (2 * np.pi) 

                                    # 准备执行 字母引导过程
                                    agent.reset(init_real_pos, init_virtual_pos, init_angle=letter_angle)


                                    
                            elif Flag_Start_Letter == 1: # if Start Letter is 1, then the virtual fish swim along the letter trajectory

                                vx_next, vy_next, orientation = agent.step(x, y)     # 根据真实鱼的位置计算虚拟鱼的下一时刻位置

                                self._virtual_fish[0,0] = vx_next  
                                self._virtual_fish[0,1] = vy_next
                                self._virtual_fish[0,2] = Stim_d
                                self._virtual_fish[0,3] = orientation  # 我们的朝向要改成virtual fish的速度方向

    

                                for i in range(1):
                                    self._osg_model.move_node(self._node_fish[i], \
                                    x=self._virtual_fish[i,0],\
                                    y=self._virtual_fish[i,1], \
                                    z = self._virtual_fish[i,2],\
                                    orientation_z = self._virtual_fish[i,3])



                            self._Flag_round += 1

                        elif self._Flag_round == Test_duration: # finish Test_duration, then update the paramters  就是切换字母，换下一个字母看效果
                            Flag_Start_Letter = 0  # reset back to circling mode and waiting for the real fish to follow
                            i_s_vf = 0
                            
                            iltetter += 1

                            if iltetter < len(Experiment_Configs):
                                config = Experiment_Configs[iltetter]
                                
                                # 解包参数并更新
                                agent.update_constraints(
                                    new_rx=config['rx'], 
                                    new_ry=config['ry'], 
                                    constraint_type=config['type']
                                )
                                print(f">>> [Next Round Setup] Applying Config {iltetter}: {config}")
                            else:
                                # 如果配置用完了，可以保持最后的状态，或者重置为默认
                                iltetter = 0

                                config = Experiment_Configs[iltetter]

                                agent.update_constraints(
                                    new_rx=config['rx'], 
                                    new_ry=config['ry'], 
                                    constraint_type=config['type']
                                )
                                print(">>> [Loop Reset] Restarting with Config Index 0")

                            # -----------------------------------------------------------
                            

                            

                            Stim_Flag           = round(1  + 0.1*C_Letter+0.001*Flag_Start_Letter,12)
                            self._Flag_round   += 1
                            

                            # 把计算好的虚拟鱼位置，更新到 3D 显示界面（屏幕/VR）
                            for i in range(1):
                                self._osg_model.move_node(self._node_fish[i], \
                                x=self._virtual_fish[i,0],\
                                y=self._virtual_fish[i,1], \
                                z = self._virtual_fish[i,2],\
                                orientation_z = self._virtual_fish[i,3])

                        elif self._Flag_round < Test_duration+Break_duration: # Break
                                self._osg_model.move_node(self._node_fish[0], hidden=True)
                                hidden = True
                                Stim_Flag = 2
                                self._Flag_round += 1
                                

                        else: # End of the round
                            Break_duration = np.random.randint(Break_range[0], Break_range[1])  # break duration between 1~5 s
                            Test_duration = 100*60*60*2 
                            Flag_round_end = Test_duration + Break_duration
                            self._Flag_round = 0
                            Stim_Flag = 0
                            print('end')

                else:
                    self._osg_model.move_node(self._node_fish[0], hidden=True) # hide all fish
                    self._period_positions[self._locked_i,:] = [x,y,z] # use kalman filter results
                    Stim_Flag = 6
                    # print "Stim_Flag" + `Stim_Flag`    

            else:  # if self.object_is_locked:
                # hide the fish
                self._osg_model.move_node(self._node_fish[0], hidden=True)
                hidden = True

            # save any should_lock_on state which we changed (for later analysis)

            self._pos_last = self._virtual_fish[:,0:4]
            self.log.orientation = self._fish_orientation
            self.log.osg_fish1_x = self._virtual_fish[0,0]
            self.log.osg_fish1_y = self._virtual_fish[0,1]
            self.log.osg_fish1_z = self._virtual_fish[0,2]
            self.log.real_fish_x = self.object_position.x
            self.log.real_fish_y = self.object_position.y
            self.log.real_fish_z = self.object_position.z
            self.log.fish1_ori_vr = self._virtual_fish[0,3]
            self.log.velocity = self._fish_velocity
            self.log.Flag_Start_Letter = Flag_Start_Letter
            self.log.Stim_Flag = Stim_Flag
            self.log.hidden = hidden
            self.log.time = time.time()
            self.log.update()

            r.sleep()


def main():
    rospy.init_node("experiment")
    parser, args = fishvr.experiment.get_and_parse_commandline()
    node = VirtualConspecificExperiment(args)
    return node.run()

if __name__=='__main__':
    main()
