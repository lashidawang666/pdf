<!-- Page 1 -->

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 

1341 

# Deep Reinforcement Learning-Based Resource Allocation for RSMA in LEO Satellite-Terrestrial Networks 

Jingfei Huang , Yang Yang , _Member, IEEE_ , Jemin Lee , _Member, IEEE_ , Dazhong He , _Member, IEEE_ , and Yonghui Li , _Fellow, IEEE_ 

**_Abstract_ — This paper considers the joint optimization of resource allocation and power control for rate-splitting multiple access (RSMA) based low earth orbits (LEO) satellite-terrestrial networks, where resource sharing between terrestrial and LEO satellite communications is optimized and the LEO satellite serves multiple ground stations (GSs) simultaneously through RSMA technique. Particularly, to make full use of RSMA technique, the LEO satellite needs to appropriately schedule transmitting power to common and private streams. Therefore, the key issue is to jointly optimize the resource allocation and power control to fully utilize the benefits of resource sharing and RSMA. However, the combination of continuous power control and discrete resource allocation becomes the bottleneck for providing an effective solution with limited system information. To deal with this problem, we propose a deep reinforcement learning (DRL)based framework which jointly employs deep Q-network (DQN) algorithm for discrete resource allocation and proximal policy optimization (PPO) algorithm for continuous power control to maximize a joint objective. Simulation experiments evaluate the performance of the proposed scheme compared with several baseline schemes and the results show the advantages of the proposed scheme.** 

**_Index Terms_ — LEO satellite-terrestrial networks, rate-splitting multiple access (RSMA), deep reinforcement learning, proximal policy optimization, joint optimization.** 

## I. INTRODUCTION 

**W** ITHand high-throughputthe increasing demandservices, oftheseamlessfuture 6Gconnectionnetworks aim to develop a flexible, resilient and adaptable framework 

Manuscript received 14 November 2022; revised 16 June 2023 and 27 September 2023; accepted 1 November 2023. Date of publication 8 November 2023; date of current version 19 March 2024. This work is supported by National Key Technologies R&D Program (2022YFB4300400), China Scholarship Council (202106475002), Belt and Road Initiative Innovative Talent Exchange Foreign Expert Project (DL2023107003L), and in partly by National Natural Science Foundation of China (61801035). The associate editor coordinating the review of this article and approving it for publication was M. C. Gursoy. _(Corresponding author: Yang Yang.)_ 

Jingfei Huang, Yang Yang, and Dazhong He are with the School of Artificial Intelligence, Beijing University of Posts and Telecommunications (BUPT), Beijing 100876, China (e-mail: huangjf@bupt.edu.cn; buptyy1015@gmail.com; hdz@bupt.edu.cn). 

Jemin Lee is with the School of Electrical and Electronic Engineering, Yonsei University, Seoul 03722, South Korea (e-mail: jemin.lee@yonsei.ac.kr). 

Yonghui Li is with the School of Electrical and Information Engineering, The University of Sydney, Sydney NSW 2006, Australia (e-mail: yonghui.li@sydney.edu.au). 

Color versions of one or more figures in this article are available at https://doi.org/10.1109/TCOMM.2023.3331021. 

Digital Object Identifier 10.1109/TCOMM.2023.3331021 

which can provide services for heterogeneous devices with high-throughput and extend coverage to unserved areas such as aero, maritime, and rural areas [1], [2]. It is generally understood that only terrestrial networks cannot meet such a demand. Integrating satellite communications with conventional terrestrial networks has been widely regarded as an inevitable trend for future 6G networks to meet the massive connectivity and ubiquitous coverage demands [3]. In this direction, low earth orbits (LEO) satellite communication has attracted great attention for supporting lower latency and higher service density. 

Moreover, considering the overlap of frequency between satellite and terrestrial communications in future 6G networks, the resource sharing in satellite-terrestrial networks is a popular research trend [4], [5]. Meanwhile, a novel multiple access method is needed for supporting higher spectral efficiency and more robust interference management is required to improve the communication security in satellite-terrestrial networks [6]. Mao et al. [7] put forward a powerful and promising multiple access scheme named rate-splitting multiple access (RSMA) for future 6G networks and gave sufficient mathematical analysis to prove the superiority of RSMA over two stateof-the-art multiple access methods, namely non-orthogonal multiple access (NOMA) and space division multiple access (SDMA) [7], [8]. Inspired by aforementioned benefits, this work investigates the application of RSMA in LEO satelliteterrestrial communications. 

To fully utilize the benefits from resource sharing and RSMA, it is important to jointly optimize the resource allocation between satellite and terrestrial networks and the power control for RSMA. Specifically, the basic idea of RSMA is to split each user’s message into two parts, namely common message and private message. All users’ common messages are encoded into one common stream which will be decoded by multiple users, while the private messages are encoded independently which will be decoded by their corresponding users [7]. Relying on the superimposed transmission manner, the transmitter transmits signals superimposing all streams. At the receiving end, each user decodes common stream firstly through successive interference cancellation (SIC) technique and then decodes its corresponding private stream by treating other private streams as noise. The key issue is to find an optimal solution for power control among common and private streams to mitigate interference and achieve optimal performance of RSMA. Inspired by the advantages of RSMA, many 

0090-6778 © 2023 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information. 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 2 -->

1342 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 

works studied the implementation of RSMA in various wireless networks and proposed some heuristic and sub-optimal approaches for solving such problems to achieve better performance of RSMA [9], [10], [11], [12], [13]. Specifically, the authors in [9] explored the joint energy efficiency and spectral efficiency optimization problem of RSMA in general downlink multiple input single output (MISO) system. In [10], a joint sub-carrier and power assignment problem was investigated for RSMA based multi-carrier MISO systems, where a successive convex approximation (SCA) method was introduced to obtain the local optimal solution for maximizing the minimum group rate. Meanwhile, in [11], a joint optimization problem for power control and rate allocation was proposed, where the authors solved the problem using a SCA method to maximize the sum rate of all users. Further, the authors in [12] and [13] explored the application of RSMA in satellite networks. In particular, a max-min fairness (MMF) optimization problem was formulated in [12], which aims to optimize the beamforming vector to achieve MMF rate objective in RSMA enabled multibeam satellite system. The authors in [13] considered an integrated satellite and aerial network where the unmanned aerial vehicle (UAV) applies RSMA scheme to provide access for numerous devices. 

The above literature makes great contributions to improving the performance for RSMA assisted systems. Nevertheless, most existing works are model-dependent and require perfect knowledge of the instantaneous channel state information (CSI). The instantaneous CSI at the transmitter is quite difficult to obtain in practice. Especially for the satellite communication system, the CSI can usually be out-of-date due to the fast time-varying channel and long propagation delay. Fortunately, in recent years, deep reinforcement learning (DRL) has shown its powerful ability in intelligent decision-making without prior knowledge of mathematical model of the system [14], [15], [16], [17]. Hence, we propose to employ DRL to solve such joint optimization problems without requirements on the instantaneous CSI at the transmitter. Moreover, the combined optimization of discrete resource allocation and continuous power control also poses a challenge for conventional methods, and it is hard for conventional methods to discover the implicit relationship among users due to the lack of learning ability and the complicated channel characteristics of different users. Therefore, considering the dynamics and uncertainty of the system, conventional methods are unable to effectively optimize resource allocation and power control simultaneously with limited information of the channel distribution. In particular, DRL has the ability to capture the implicit association among users through the interactions with the unknown environment and learn an optimal solution to achieve a pre-specified objective. Also, DRL technique has been adopted to deal with such combined problems in many existing works [18], [19], [20], [21], [22], [23], [24] and has gained outstanding performance. 

Specifically, both [18] and [19] apply DRL to jointly optimize power control and channel allocation in NOMA based systems. The authors in [18] first obtained the optimal power allocation solution and then utilized a DRL framework to solve the channel assignment problem, while the 

authors in [19] introduced deep deterministic policy gradient (DDPG) algorithm to optimize the sub-carrier and power assignment simultaneously, aiming to achieve the weighted sum rate maximization objective. Further, in [20], a combined deep Q-network (DQN) and DDPG framework for resource allocation in vehicle-to-everything (V2X) networks was proposed, which used DQN and DDPG algorithms to optimize the sub-channel allocation and continuous power control problems, respectively. In particular, the authors in [21] and [22] explored the application of DRL for resource allocation in satellite networks. Furthermore, there are several works focusing on the implementation of DRL in RSMA enabled networks [23], [24]. Specifically, the authors in [23] exploited RL technology to deal with the user clustering optimization problem in a downlink fog radio access network and jointly utilized alternating optimization and fractional programming approaches for resource allocation optimization. Further, the authors in [24] put forward a proximal policy optimization (PPO) based power control method and conducted simulations to prove the benefits of the proposed scheme compared with several baseline schemes under different power and QoS requirements. However, the joint optimization of resource allocation and power control for RSMA based LEO satellite-terrestrial communications using DRL technique has not been studied in existing literature. And such problems with joint discrete and continuous action space pose challenges to the application of DRL algorithms. 

## _A. Contributions and Organization_ 

This work considers a RSMA based LEO satellite-terrestrial communications network where the time-frequency resources of terrestrial networks are allowed to be shared with LEO satellite communications. Specifically, there are multiple resource blocks (RBs) that are shared by terrestrial networks and LEO satellite communications. The LEO satellite serves multiple ground devices sharing the same RB through a RSMA technique. The key issue is to jointly optimize the RB allocation between satellite and terrestrial communications and the power control for RSMA to improve the system performance. Inspired by the power of DRL technique, we propose to exploit DRL to solve the joint optimization problem. To the best of our knowledge, this is the first work to exploit DRL technique to jointly optimize RB allocation and power control for RSMA based LEO satellite-terrestrial communications networks. The main contributions of this work are summarized below. 

- We investigate a joint resource allocation and power control problem for RSMA based LEO satellite- terrestrial communications networks. To maximize the sum rate of the system, we formulate the proposed problem as a sum rate joint optimization problem and propose to exploit DRL technique to find optimal solutions for the first time. 

- To improve the decision-making efficiency, the formulated problem is modeled as a multi-agent problem with distributed decision-making, where we set up an agent for each RB in charge of decisions on resource allocation and power control for users sharing this RB. Furthermore, we adopt a centralized training method to enhance the training efficiency of the algorithm. 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 3 -->

HUANG et al.: DRL-BASED RESOURCE ALLOCATION FOR RSMA IN LEO SATELLITE-TERRESTRIAL NETWORKS 

1343 

- Since a joint discrete and continuous action space is designed for the proposed problem, we put forward a combined DQN and PPO resource allocation (DPRA) framework, where the DQN unit and PPO unit are in charge of decision-making on resource allocation and power control, respectively. 

- The advantages of the proposed DPRA scheme over several baseline schemes is demonstrated via simulations under different cases, and the results prove the effectiveness of the proposed scheme. 

The organization of the paper is as follows. Section II introduces the considered RSMA based hybrid LEO satellite and terrestrial communications network, and Section III describes the mathematical formulation for the proposed problem. After that, the proposed DPRA framework is demonstrated in Section IV and Section V analyzes simulation experiments and results. Section VI concludes the paper. 

## II. SYSTEM MODEL 

As illustrated in Fig. 1, considering the limited time-frequency resources in future 6G networks, we investigate a resource sharing scheme for hybrid satellite and terrestrial communications networks, where the LEO satellite communications are allowed to share the time-frequency resources with terrestrial networks. Specifically, in the system, there are _M_ terrestrial cellular cells (T-cells) orthogonally occupying _M_ RBs to serve multiple cellular users in the T-cell. Considering downlink communications, the LEO satellite provides services to _N_ ground stations (GSs) by sharing _M_ T-cells’ RBs, where _K_ ( _K < N_ ) out of _N_ GSs combine into one GS group and will be served through the same RB simultaneously to improve spectral efficiency. Each GS belongs to multiple groups and can be served by LEO satellite on multiple RBs. For example, in Fig. 1, the downlink communications in T-cell1 and T-cell2 are carried out through RB1 and RB2, respectively. The LEO satellite serves multiple GSs by sharing the RBs of terrestrial networks, where group1 shares RB2 with T-cell2 and group2 shares RB1 with T-cell1. Note that the RB assignment among GS groups should be optimized to improve the system performance. As for the interference, compared with GSs, users in T-cell usually locate in areas that are heavily obscured by buildings and their antenna gain can usually be much lower than that of GSs. Hence, the interference from the LEO satellite can be ignored while the interference from the T-cell base station (BS) is considered. Furthermore, as a promising interference management and multiple access technique for future 6G networks, RSMA technique is expected to be applied in LEO satellite communications, where the LEO satellite serves multiple GSs in the same group, to mitigate the co-channel interference and improve the spectral efficiency. 

## _A. Signal Model_ 

As illustrated in Fig. 1, the LEO satellite provides downlink services to _K_ GSs in the same group by the RSMA technique. Let _Gm_ ( _m_ = 1 _, · · · , M_ ) denote the group composed by _K_ 

GSs sharing _m_ -th RB. For group _Gm_ , the LEO satellite has _K_ messages denoted by _{W_ 1 _,m, · · · , Wk,m, · · · , WK,m}_ that need to be transmitted to _K_ GSs in group _Gm_ , respectively. The message _Wk,m_ transmitted to _k_ -th GS in group _Gm_ is divided into two parts, namely common message and private message, i.e., _Wk,m →{Wk,m_<sup>_c, W_</sup> _k,m_<sup>_p},_(</sup><sup>_k_=1</sup><sup>_,_2</sup><sup>_, · · ·, K_).</sup> All common messages of _K_ GSs in group _Gm_ are encoded into a common stream that will be decoded by all GSs in group _Gm_ , while each private message _Wk,m_<sup>_p_isencodedinto</sup> a private stream _s_<sup>_k_</sup> _m_<sup>thatwillbedecodedbythe</sup><sup>_k_-thGS</sup> independently. Relying on superimposed transmission manner, the signal transmitted by the LEO satellite for group _Gm_ is written as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0003-10.png)


where _pc,m_ is the transmitting power assigned to the common stream and _pk,m_ is the transmitting power assigned to the private stream intended for _k_ -th GS. The power control has 

_K_ 

to meet the constraint _pc,m_ + � _pk,m ≤ Pt_ , in which _Pt_ is _k_ =1 the maximum transmitting power of the LEO satellite on each RB. 

The total signal received by the _k_ -th GS in group _Gm_ can be written as: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0003-15.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0003-16.png)


where _gk,m_ is the link gain between the _k_ -th GS in group _Gm_ and the LEO satellite, and _nk,m_ includes the additive white Gaussian noise at _k_ -th GS in group _Gm_ and the interference from the T-cell sharing _m_ -th RB. Applying SIC technique, each GS will decode the common stream firstly by treating other private streams as interference and then subtract the common stream from the overall received signal. After that, the private stream _s_<sup>_k_</sup> _m_<sup>isdecodedonlybythe</sup><sup>_k_-thGSwhile</sup> other private streams are treated as noise at the _k_ -th GS. As a result, the _k_ -th GS in group _Gm_ can calculate the signal to interference plus noise ratio (SINR) for two streams after performing SIC, which are written as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0003-18.png)


in which _PT_ is the transmitting power of the T-cell BS, and _gk,m_<sup>_T_isthelinkgainbetweentheT-cellBSoccupying</sup><sup>_m_-th</sup> RB and the _k_ -th GS in group _Gm_ . (3) is the SINR of common stream and thus the achievable rate for common stream at _k_ -th GS is given as _Ck,m_<sup>_c_=log</sup> 2<sup>(1 + Γ</sup><sup>_c_</sup> _k,m_<sup>).Toensurethat</sup> the common stream can be successfully demodulated by all GSs on the _m_ -th RB, RSMA regards the minimum value of all common streams’ rate as the achievable common rate on _m_ -th 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 4 -->

1344 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-02.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-03.png)


Fig. 1. RSMA based LEO satellite-terrestrial communications networks. 

RB, i.e., _Cm_<sup>_c_=</sup> _k_ =1min _,··· ,K_ �log2(1 + Γ<sup>_c_</sup> _k,m_<sup>)</sup> �. (4) is the SINR of private stream and the achievable rate for private stream at _k_ -th GS is given as _Ck,m_<sup>_p_=log</sup> 2<sup>(1 + Γ</sup><sup>_p_</sup> _k,m_<sup>).Sincethereis</sup> only a part of the common stream belonging to a single GS, it is reasonable to share the common rate _Cm_<sup>_c_among</sup><sup>_K_GSs</sup> in group _Gm_ [7]. For convenience, considering a fixed rate allocation by evenly distributing the common rate to each GS in group _Gm_ , the total rate of _k_ -th GS in group _Gm_ is given as _Ck,m_<sup>_tot_=</sup><sup>_C_</sup> _m_<sup>_c/K_+</sup><sup>_C_</sup> _k,m_<sup>_p_andthesumgrouprateofallGSs</sup> _K_ in group _Gm_ is given as _Cm_<sup>_sum_</sup> = _Cm_<sup>_c_+</sup> � _Ck,m_<sup>_p_.</sup> _k_ =1 

## _B. Channel Model_ 

_1) Terrestrial channel model:_ The terrestrial interference link gain between the T-cell BS occupying _m_ -th RB and the _n_ -th ( _n_ = 1 _,_ 2 _, . . . , N_ ) GS can be modeled as 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-08.png)


where _βn,m_ is the large-scale fading component and _h_<sup>_T_</sup> _n,m_ is the small-scale fading with Rician distribution. The large-scale fading _βn,m_ is modeled as _βn,m_ ( _dB_ ) = _−_ 120 _._ 9 _−_ 37 _._ 6log10( _dn,m_ ) + 10log10( _z_ ) and _βn,m_ = 10<sup>_βn,m_(</sup><sup>_dB_)</sup><sup>_/_10</sup> according to [25], where _dn,m_ is the distance in km between the T-cell BS and the GS, and _z_ is a log-normal random variable following ln _z ∼ N_ (0 _, σz_<sup>2)with</sup><sup>_σ_</sup> _z_<sup>2= 8dB.</sup> 

_2) LEO satellite channel model:_ Here, we adopt the satellite channel model described in [26] and [27], where the link gain between the _n_ -th GS and the LEO satellite on the _m_ -th RB can be modeled as 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-11.png)


where _δn,m_ is the small-scale fading with Rayleigh distributed envelope, _Gk_ , _Gs_ are the gains of GS antenna and satellite antenna, respectively. _fc_ is the carrier frequency, _c_ is the light speed, and _ds_ is the distance between the GS and the LEO satellite. 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-13.png)


As mentioned above, this paper jointly considers the RB allocation and power control optimization problems. Thus, we first represent the decisions for the RB allocation and power control by the sets **_α_** = [ **_α_** 1 _, · · · ,_ **_α_** _M_ ] and **_p_** = [ **_p_** 1 _, · · · ,_ **_p_** _M_ ], respectively, where **_α_** _m_ = = [ _α_ 1 _,m, · · · , αN,m_ ]( _m_ 1 _,_ 2 _, · · · , M_ ) is a _N_ dimension vector denoting the assignment of _m_ -th RB, and **_p_** _m_ = [ _pc,m, p_ 1 _,m, · · · , pN,m_ ]( _m_ = 1 _,_ 2 _, · · · , M_ ) is a _N_ + 1 dimension vector denoting the power control among group _Gm_ sharing the _m_ -th RB. For the RB allocation, _αn,m_ = 1 indicates that the LEO satellite serves the _n_ -th GS through the _m_ -th RB, while _αn,m_ = 0 means that the _n_ -th GS does not occupy the _m_ -th RB. Since each RB can support _K_ GSs to access at the same time, the assignment of each RB 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-15.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-16.png)


The SINR and achievable rates defined in Section II-A are only for the _K_ GSs sharing the _m_ -th RB. Considering the resource allocation, the SINR for common and private streams at the _n_ -th ( _n_ = 1 _, . . . , N_ ) GS on _m_ -th RB can be calculated by the following equations: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0004-18.png)


The corresponding rate for the private stream at the _n_ -th GS on _m_ -th RB is written as _Cn,m_<sup>_p_=log</sup> 2<sup>(1+Γ</sup><sup>_p_</sup> _n,m_<sup>).The</sup> achievable rate of common stream on _m_ -th RB is defined as _Cm_<sup>_c_=</sup> min �log2(1 + Γ<sup>_c_</sup> _n,m_<sup>)</sup><sup>_|αn,m_= 1</sup> �, and the total _n_ =1 _,··· ,N_ achievable rate of _n_ -th GS on _m_ -th RB is given as _Cn,m_<sup>_tot_=</sup> _αn,m_ ( _Cm_<sup>_c/K_+</sup><sup>_C_</sup> _n,m_<sup>_p_).</sup> 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 5 -->

1345 

HUANG et al.: DRL-BASED RESOURCE ALLOCATION FOR RSMA IN LEO SATELLITE-TERRESTRIAL NETWORKS 

Furthermore, to successfully implement SIC while decoding the common stream at the receiver, for each GS sharing _m_ -th RB, i.e., _αn,m_ = 1, the transmitting power must satisfy the following constraint [28]: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-03.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-04.png)


where _θth_ denotes the minimum power difference between the common stream and the other private streams plus interference and noise. In general, _θth_ is regarded as the SIC detection threshold and we name the left part of inequality (9) as the power difference. In (9), the first term is the received common stream power of GSs on the _m_ -th RB, while the remaining terms are the power of all non-decoded private streams plus interference and noise. To successfully implement SIC for the common stream decoding in RSMA, each user should have the received SINR of the private stream smaller than that of the common stream [11]. Just as illustrated in the constraint (9), the common stream should be a stronger signal compared with the private stream plus interference and noise. 

In this work, with the objective of maximizing the system sum rate, the key issue is to solve the joint optimization problem for RB allocation and power control under some power and SIC detection constraints. Mathematically, the joint optimization problem with the sum-rate maximization is formulated as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-07.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-08.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-09.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-10.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-11.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-12.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-13.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-14.png)


where _Cn,m_<sup>_th_denotestheminimumachievablerateofthe</sup><sup>_n_-</sup> th GS on _m_ -th RB. Constraint (10a) suggests that each RB should be either assigned or not and Constraint (10b) indicates that each RB can support _K_ GSs at the same time. Constraints (10c) and (10d) indicate that the power value must be larger than zero and cannot exceed the maximum transmitting power _Pt_ . Constraint (10e) is the minimum rate requirement for each GS. Constraint (10f) is the requirement to successfully implement SIC for the common stream. 

The optimization problem in (10) is non-convex and it is hard to find the optimal solution with the exhaustive searching method due to the continuous value of power control and huge solution space. Moreover, considering the dynamic and uncertainty of the environment, the channel state distribution 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0005-17.png)


Fig. 2. RL framework for the proposed optimization problem. 

is unfamiliar to the LEO satellite, which makes it more challenging for conventional methods to adaptively derive the optimal solution. Therefore, we adopt DRL based approaches in the following sections to learn an optimal solution for the proposed problem, which has the ability to intelligently make decisions on RB allocation and power control without any mathematical knowledge of the environment in advance. 

## IV. DRL FRAMEWORK FOR JOINT RESOURCE ALLOCATION AND POWER CONTROL 

In this section, we intend to design a DRL-based optimization framework to jointly optimize RB allocation and power control problem in (10). As a promising branch of machine learning mechanisms, DRL combines RL and deep learning to improve the learning efficiency of RL [15], which aims to learn an optimal policy to make adaptive decisions by interacting with the time-varying environment frequently. In general, the problem modeling in DRL is based on a Markov decision process (MDP), which is characterized by state space, action space, transition probability, and immediate reward. As shown in Fig. 2, let _at_ and _st_ denote the action chosen by the agent and the state observed by the agent at the time step _t_ , respectively. A reward fed back from the environment, denoted by _rt_ +1, is defined to reflect the performance of the agent taking action _at_ . Then, we have the cumulative discount reward, written as _Rt_ =<sup>�</sup><sup>_∞_</sup> _τ_ =0<sup>_γτrt_+1+</sup><sup>_τ_,</sup> where _γ ∈_ [0 _,_ 1) is the discounted factor to balance the influence of the immediate reward and future rewards. Since the state transition probability is usually hard to know in realistic environment, model-free DRL algorithms gain more attention. In other words, any prior knowledge or mathematical model of the environment is no longer necessary for DRL. By interacting with the unknown environment in a trial-anderror manner, the agent autonomously learns an optimal action policy which can instruct the agent to take a suitable action based on the monitored environment state to maximize a predefined objective function [14]. 

In this work, the LEO satellite has to make decisions on the assignment for each RB and the following power control for GSs sharing this RB. Therefore, we have to set 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 6 -->

1346 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 

an agent responsible for each RB and thus a multi-agent DRL problem is proposed. Two main schemes are available for such problems. In one scheme, all the agents are treated as a single integrated agent which collects global state information from all agents and then makes centralized decisions for all agents [29]. On the one hand, the action and state space increase linearly with the number of agents, which will decrease the convergence of DRL. On the other hand, the integrated agent makes decisions after collecting state information of all RBs, but the information feedback of each RB is not synchronous, thus causing the decision-making delay. Hence, we adopt the other distributed decision-making scheme to solve the proposed multi-agent problem, where each agent only collects local state information related to one RB and makes decisions for this RB independently. 

## _A. Key Elements of DRL_ 

Inspired by the core idea of DRL, we model the problem in (10) as an MDP and design a DRL-based optimization framework to jointly optimize RB allocation and power control. Without loss of generality, time for data transmission can be divided into a series of sub-frames denoted by positive integer _{_ 1 _, . . . , t, . . .}_ . The time step for the interaction between the agent and the environment can be set as one sub-frame. According to the above analysis of DRL, three key elements, i.e., action, state, and reward are very important for the interaction process between the agent and the environment, so we first give the detailed definitions of three key elements for each agent as follows. 

_1) Action:_ Relying on the monitored local state information, each agent takes its own actions to decide the assignment of corresponding RB and the subsequent power control. At time step _t_ , the action executed by the agent for the _m_ - th RB is given as _am_ ( _t_ ) = [ **_α_** _m_ ( _t_ ) _,_ **_p_** _m_ ( _t_ )]. As mentioned in Section III, **_α_** _m_ ( _t_ ) = [ _α_ 1 _,m_ ( _t_ ) _, · · · , αN,m_ ( _t_ )] is a _N_ dimension vector to denote the assignment of _m_ -th RB at time step _t_ , where _αn,m_ ( _t_ ) = 1 means that the _n_ -th GS occupies the _m_ -th RB, otherwise _αn,m_ ( _t_ ) = 0. Similarly, **_p_** _m_ ( _t_ ) = [ _pc,m_ ( _t_ ) _, p_ 1 _,m_ ( _t_ ) _, · · · , pN,m_ ( _t_ )] is a _N_ +1 dimension vector to denote the power control of GSs on the _m_ -th RB. 

_2) State:_ As illustrated in Fig. 2, at time _t_ , the agent takes the action _at_ relying on the observed state _st_ , and then the environment will transit to the next state _st_ +1 and feed back a reward _rt_ +1. Therefore, the state has to contain useful information that can enable the agent to make suitable decisions. Here, we first define the historical SINRs of the common and private streams from GSs as the channel observation for the _m_ -th RB, i.e., **_o_** _m_ ( _t_ ) = _{_ [Γ<sup>_c_</sup> _n,m_<sup>(</sup><sup>_t−_1)</sup><sup>_,_Γ</sup><sup>_p_</sup> _n,m_<sup>(</sup><sup>_t−_1)]</sup><sup>_, k_=</sup> 1 _, . . . , K}_ . Then, the state observed by the agent for deciding the assignment of _m_ -th RB at time _t_ is written as _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)=</sup> **_o_** _m_ ( _t_ ), and the state for deciding power control of GSs on _m_ -th RB is defined as _s_<sup>_p_</sup> _m_<sup>(</sup><sup>_t_) = [</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_o_**</sup><sup>_m_(</sup><sup>_t_)].</sup> 

_3) Reward:_ The agent receives an immediate reward _rt_ +1 after performing the action _at_ in state _st_ , which acts as an incentive for the agent to train an optimal strategy that can select proper action to achieve a pre-specified objective. Considering the proposed optimization issue in (10), the reward function should be formulated to maximize the sum rate of 

all GSs sharing the same RB. Considering the constraint (10f) that indicates the requirement to successfully implement SIC for the common stream decoding, we first reformulate the achievable rate of common stream for GSs in group _Gm_ (i.e., _αn,m_ = 1 _, ∀n_ ) as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0006-09.png)


where _diff_ is the power difference defined as the left part of inequality (9). The achievable common rate on the _m_ -th RB is still defined as _Cm_<sup>_c_=</sup> min �log2(1 + Γ<sup>_c_</sup> _n,m_<sup>)</sup><sup>_|αn,m_= 1</sup> �. _n_ =1 _,··· ,N_ The immediate reward _rm_ ( _t_ + 1) received by the agent responsible for the _m_ -th RB is defined as: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0006-11.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0006-12.png)


where _χm_ ( _t_ ) denotes the penalty for not satisfying the minimum rate requirement in constraint (10e). Specifically, at time step _t_ , the penalty related to the _m_ -th RB is written as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0006-14.png)


where the function _f_ ( _Cn,m_<sup>_tot_(</sup><sup>_t_)</sup><sup>_, C_</sup> _n,m_<sup>_th_)isequalto0if</sup> _Cn,m_<sup>_tot_(</sup><sup>_t_)</sup><sup>_≥C_</sup> _n,m_<sup>_th_,andotherwise</sup><sup>_f_(</sup><sup>_C_</sup> _n,m_<sup>_tot_(</sup><sup>_t_)</sup><sup>_, C_</sup> _n,m_<sup>_th_) = 1.</sup> Based on the reward designed in (12), the agent will receive a higher reward value if it chooses actions according to a policy that can achieve a higher sum rate, where the punishment for not satisfying constraints is also considered. Therefore, through such an incentive mechanism, the agent updates its policy to achieve a higher sum rate by interacting with the unknown environment. 

## _B. Combined DQN and PPO Framework Design_ 

After giving the definitions of three key elements, we intend to introduce the framework design combining DQN and PPO algorithms in this section. As mentioned above, each agent has to make decisions on RB allocation and power control for corresponding RB. The action for RB allocation is discrete, while the action for power control is continuous. For such problems with joint action space, most studies proposed to discretize the continuous action values into several levels first, and then solved the problem with conventional DRL algorithms such as DQN. However, the discretization of the action space usually loses some useful information, thus causing the quantization error. To address this issue, we intend to design a joint optimization framework by exploiting the DRL technique, where the agent uses DQN algorithm for RB allocation and a highly-effective algorithm suitable for continuous action space named PPO is employed for power control. Moreover, to improve the training efficiency, a centralized training mechanism is adopted, in which the experience data from all agents will be collected to train a public agent and all the agents share the same structure and parameters with the public agent. Fig. 3 shows the proposed DRL framework, 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 7 -->

HUANG et al.: DRL-BASED RESOURCE ALLOCATION FOR RSMA IN LEO SATELLITE-TERRESTRIAL NETWORKS 

1347 

which illustrates both the DQN unit for RB allocation and the PPO unit for power control as well as the training process for both units. 

_1) DQN for RB Allocation:_ DQN algorithm is the combination of deep learning and Q-learning, which is a popular value-based DRL algorithm suitable for problems with discrete actions and large state space. Specifically, given the observed state _s_ , the agent generates an action _a_ according to a policy _π_ ( _a|s_ ) which can be think of as the probability of all possible actions taken in the current state. Q-learning algorithm defines a Q-function to evaluate the policy, and the Q-value for an action-state pair ( _s, a_ ) under a specific policy is expressed as _Qπ_ ( _s, a_ ) = E _π_ [ _Rt | st_ = _s, at_ = _a_ ]. The agent seeks to learn an optimal policy _π_<sup>_∗_</sup> ( _a|s_ ) that can maximize the Q-value, i.e., _Q_<sup>_∗_</sup> ( _s, a_ ) = max _Qπ_ ( _s, a_ ). Relying on the Bellman optimality _π_ equation, we can get the following equation: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-04.png)


As mentioned above, through the constant interactions with the environment, the agent obtains the experience data denoted by a four-tuple, i.e., ( _st, at, rt_ +1 _, st_ +1). Then, the update of the Q-value is given as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-06.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-07.png)


in which _η_ 1 _∈_ [0 _,_ 1) is the learning rate. 

In DQN algorithm, a neural network named Q neural network (QNN) is introduced to approximate the Q-value denoted by _q_ ( _s, a_ ; **_ϕ_** ), where **_ϕ_** is the weights of the QNN. Furthermore, DQN employs two mechanisms to enhance its stability and accelerate its convergence. The first one is _experience replay_ , which maintains an memory pool to storage the historical experience data in a first-in-first-out (FIFO) manner for QNN training. Specifically, for each training step, a batch of experience data will be sampled from the memory pool to update the weights of the QNN. The second one is _quasi-static target network_ , which introduces another target QNN with weights **_ϕ_**<sup>**_′_**</sup> to approximate the target Q-value in the training procedure. 

As shown in the right part of Fig. 3, taking RB allocation into account, the QNN takes the state _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)asinputand</sup> outputs Q-values _q_ ( _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_);</sup><sup>**_ϕ_**)forallpossibleactions.</sup> Then, the agent selects action **_α_** _m_ ( _t_ ) based on _ε_ -greedy policy which allows the agent to execute a random action from action space with a probability _ε_ or to select an action that can maximize _q_ ( _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_);</sup><sup>**_ϕ_**)withaprobability1</sup><sup>_−ε_.</sup> After that, the environment will transit to the next state _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_+ 1)andfeedbackareward</sup><sup>_rm_(</sup><sup>_t_+ 1)thatisrelated</sup> to the sum group rate. Based on such interaction process, the agent constantly obtains the experience data in the form of ( _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_)</sup><sup>_, rm_(</sup><sup>_t_+ 1)</sup><sup>_, ss_</sup> _m_<sup>(</sup><sup>_t_+ 1))totraintheQNN.The</sup> purpose of QNN training is to optimize the fitting of the Q-function by minimizing the loss between the target value 

and the predicted value. The loss function used here is defined as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-12.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-13.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-14.png)


To minimize the loss function, the gradient descent method is utilized to update the weights of QNN, which is written as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0007-16.png)


_2) PPO for Power Control:_ As the action value of power control is continuous, proximal policy optimization (PPO) algorithm [30] is a perfect choice for solving such problems, which is a new highly-effective algorithm originated from classic actor-critic (AC) algorithm and combines the advantages of value-based and policy-based DRL algorithms. The conventional policy-based algorithms, such as policy gradient (PG) algorithm, intend to directly update the policy _π_ **_θ_** to maximize the cumulative reward, which can solve the problems with continuous action and the stochastic policy problems. The policy _π_ **_θ_** can be regarded as the probability distribution of possible actions in a specific state, i.e., _π_ **_θ_** = **_Pr_** ( _a|s_ ). However, PG algorithm has low sample utilization and is sensitive to the update trajectories. AC algorithm contains two neural networks to estimate the value function and the policy, respectively, which compensates for the shortcomings of the above two algorithms. Further, PPO algorithm allows to train neural networks multiple times with a batch of experience data to improve sample utilization. And it raises a novel loss function which adds the restriction on the difference between the new and old policies to enhance the learning efficiency. 

Generally, there are three neural networks in PPO algorithm, namely new actor network _π_ **_θ_** with weights **_θ_** , old actor network _π_ **_θ_** _old_ with weights **_θ_** _old_ , and critic network _V_ **_φ_** with weights **_φ_** . The main function of the new actor network is to interact with the environment and generate the new policy (i.e., the probability distribution of actions in a given state). The old actor network has the same structure as the new actor network and records the old policy output by the new actor network, which serves to limit the gap between the old and new policies. The critic network has the role of evaluating the actor networks by fitting the state value function under the given policy. To be specific, the new actor network is responsible for interacting with the environment constantly and generates a batch of experience data stored in the memory pool with the form of ( _s_<sup>_p_</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_p_**</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_, rm_(</sup><sup>_t_+1)).Then,theweights</sup> of critic network and new actor network will be firstly updated multiple times using collected experience data, and the value of weights **_θ_** _old_ needs to be replaced by weights **_θ_** at last, as shown in the left part of Fig. 3. 

Taking power control into account, the right part of Fig. 2 describes the interaction process where the new actor network obtains the observed state _s_<sup>_p_</sup> _m_<sup>(</sup><sup>_t_)asinputandpredictsthe</sup> action probability distribution for power control in current 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 8 -->

1348 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0008-02.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0008-03.png)


Fig. 3. Detailed architecture of the proposed DPRA framework. 

state. The agent executes an action **_p_** _m_ ( _t_ ) based on this probability distribution and receives a reward _rm_ ( _t_ +1). After multiple time steps, the agent collects a batch of experience data and updates the weights **_θ_** and **_φ_** in a gradient ascent manner, i.e., **_θ_** = **_θ_** + _η_ 2 _∇_ **_θ_** _L_ ( **_θ_** _,_ **_φ_** ) and **_φ_** = **_φ_** + _η_ 2 _∇_ **_φ_** _L_ ( **_θ_** _,_ **_φ_** ), where the learning rate _η_ 2 _∈_ [0 _,_ 1) and _L_ ( **_θ_** _,_ **_φ_** ) = _L_ ( **_θ_** ) _−L_ ( **_φ_** ) is the combined loss function. _L_ ( **_φ_** ) is the loss function for the critic network which is still a square error loss written as: 

Algorithm 1 describes the overall process of the proposed DPRA scheme in detail in the form of pseudo code. 

## V. SIMULATION RESULTS 

In this section, we evaluate the performance of the proposed DPRA scheme through multiple simulations. For comparison, we also conduct simulations to study the performance of several baseline methods as follows: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0008-09.png)


- DQN based resource allocation (DQRA) method: This method utilizes DQN algorithm to learn optimal strategies for both RB allocation and power control. As DQN algorithm can only optimize problems with discrete action space, the action value of power control is uniformly discretized into ten levels from 0dBm to _Pt_ here. 

where _V_ **_φ_** ( _sm_<sup>_p_(</sup><sup>_t_))isthestatevaluepredictedbythecritic</sup> network. Before introducing the loss function _L_ ( **_θ_** ) for actor network, we first give the definition of the advantage function at time step _t_ as follows: 

- Joint DQN and WMMSE based resource allocation (DWRA) method: This method uses DQN algorithm for RB allocation and adopts a modified WMMSE method for solving power control problem in RS [31] to get the optimal power control strategy. 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0008-13.png)


For problems with continuous action space, the policy variation is sensitive to the update of the weights [14] and the large difference between the new policy and the old policy during training process will also affect the learning efficiency. To solve these problems, PPO adopts a clipped surrogate advantage loss defined as follows: 

- Random RB allocation and PPO based power control (RPRA) method. 

- Random RB allocation and DQN based power control (RDRA) method. 

- No-RS system (NoRS) and Orthogonal multiple access (OMA) system: In these cases, we still consider resource sharing between terrestrial communications and LEO satellite communications, but RSMA technique is not adopted. In the first case, the LEO satellite serves _K_ GSs sharing the same RB simultaneously through a single stream. In the second case, the LEO satellite serves _K_ GSs sharing the same RB through an orthogonal manner such as FDM. 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0008-18.png)


where _rm_<sup>**_θ_**(</sup><sup>_t_)=</sup> _π_ **_θ_** _πold_ **_θ_** <u>(</u> **_<u>p</u>_** ( _<u>m</u>_ **_p_** _m_ <u>(</u> _t_ () _t|_ ) _s|_<sup>_p_</sup> _<u>msm</u>_<sup>_~~p~~_</sup><sup><u>(</u></sup><sup>_t_())</sup><sup>_t_))istheprobabilityratio</sup> between the new and old policies. The clip operation _clip_ ( _·_ ) is short for _clip_ � _rm_<sup>**_θ_**(</sup><sup>_t_)</sup><sup>_,_1</sup><sup>_−ε,_1 +</sup><sup>_ε_</sup> � to restrict the probability ratio into the interval [1 _− ε,_ 1 + _ε_ ], which can prevent the large changes in policy updates. Furthermore, since the agent is prone to be stuck in sub-optimal situations, the entropy regularization method is adopted in the loss function to encourage the agent’s exploration of actions. Thus, the loss function defined in (21) for the new actor network is replaced by following equation: 

## _A. Simulation Setup_ 

In the system, we consider one LEO satellite coexisting with two T-cells, i.e., _M_ = 2. For most cases, we assume that there are three GSs in total and each RB can support two GSs to access at the same time, i.e. _N_ = 3 and _K_ = 2. Unless otherwise specified, the transmitting power of the T-cell 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0008-22.png)


where _Sπ_ **_θ_** denotes the information entropy of the new policy. Unless otherwise specified, the transmitting power of the T-cell Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 9 -->

HUANG et al.: DRL-BASED RESOURCE ALLOCATION FOR RSMA IN LEO SATELLITE-TERRESTRIAL NETWORKS 

1349 

**Algorithm 1** The Training Process of the Proposed DPRA <u>Scheme</u> 

- 1: Initialize the memory pools _D_ 1 and _D_ 2 2: Initialize the weights **_ϕ_** in DQN unit, the weights **_θ_** and **_φ_** in PPO unit 

- 3: Initialize the weights as **_ϕ_**<sup>_′_</sup> = **_ϕ_** , **_θ_** _old_ = **_θ_** 4: **for** each time step _t_ **do** 5: **for** each agent _m_ **do** 6: Get the channel observation **_o_** _m_ ( _t_ ) and set _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_) =</sup><sup>**_o_**</sup><sup>_m_(</sup><sup>_t_)</sup> 

7: Input _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)totheQNNandoutputQ-values</sup> _q_ ( _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_);</sup><sup>**_ϕ_**)forallpossibleactions</sup> 8: Select RB allocation action **_α_** _m_ ( _t_ ) according to _ϵ_ -greedy policy and Q-values 

9: Set state _s_<sup>_p_</sup> _m_<sup>(</sup><sup>_t_) = [</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_o_**</sup><sup>_m_(</sup><sup>_t_)]</sup> 10: Input _s_<sup>_s_</sup> _p_<sup>(</sup><sup>_t_)tothenewpolicynetworkandoutput</sup> the probability distribution for power control action 

11: Generate power control action **_p_** _m_ ( _t_ ) based on the probability distribution 

12: Obtain the reward _rm_ ( _t_ + 1) in (12) and next state _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_+1)=</sup><sup>**_o_**</sup><sup>_m_(</sup><sup>_t_+1)afterconductingactions</sup> ( **_α_** _m_ ( _t_ ) _,_ **_p_** _m_ ( _t_ )) 

13: Collect experience data ( _s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_α_**</sup><sup>_m_(</sup><sup>_t_)</sup><sup>_, rm_(</sup><sup>_t_+</sup> 1) _, s_<sup>_s_</sup> _m_<sup>(</sup><sup>_t_+ 1))into</sup><sup>_D_1,and(</sup><sup>_ss_</sup> _p_<sup>(</sup><sup>_t_)</sup><sup>_,_</sup><sup>**_p_**</sup> _m_<sup>(</sup><sup>_t_)</sup><sup>_, rm_(</sup><sup>_t_+ 1))into</sup> _D_ 2 

14: **end for** 15: Randomly select a batch of experience samples from _D_ 1 to update weights **_ϕ_** according to (18) 

16: Update **_ϕ_**<sup>_′_</sup> as **_ϕ_**<sup>_′_</sup> = **_ϕ_** every 20 time steps 

17: **if** _D_ 2 is full **then** 18: **for** _i ∈{_ 1 _, · · · , F }_ **do** 

### TABLE I 

PARAMETERS IN THE SYSTEM 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0009-15.png)


### TABLE II 

HYPER-PARAMETERS IN THE TRAINING PROCESS 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0009-18.png)


19: Compute the loss function based on experi- 

ences in _D_ 2: 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0009-21.png)


where _L_ ( **_θ_** ) is defined in (22). 

20: Update the weights **_θ_** and **_φ_** in a gradient ascent manner w.r.t. _L_ ( **_θ_** _,_ **_φ_** ) 

21: **end for** 

22: Replace the value of **_θ_** _old_ with **_θ_** , i.e., **_θ_** _old_ = **_θ_** 23: Clear the memory pool _D_ 2 24: **end if** 25: **end for** 

BS and the LEO satellite is 35dBm and 50dBm, respectively, and the SIC detection threshold _θth_ is set as -94dBm [11]. Other system parameters are listed in Table I. In particular, the small-scale fading in LEO satellite channel model denoted by _δk,m_ is modeled as _δk,m_ ( _t_ ) = _ρδk,m_ ( _t−_ 1)+ _nk,m_ ( _t_ ) according to the Jakes model. The correlation _ρ_ is calculated by _ρ_ = _J_ 0 (2 _πfdTs_ ), in which _J_ 0 ( _·_ ) is the first kind zero-order Bessel function, _Ts_ is the time interval between adjacent instants, and _fd_ is the maximum Doppler frequency. 

The basic structure of the neural networks used in the designed DPRA framework includes one input layer taking 

state as input, two hidden layers to extract feature information, and one output layer to generate value function or action. The hidden layer is fully-connected layer containing 64 neurons with ReLU activation layer at last. Note that there is an additional layer at the end of actor networks in PPO unit to limit the output to the power boundary. For the network training, the Adam optimizer [32] is adopted for updating the weights of each neural network. Most of the hyper-parameters are determined relying on the past experience and we will discuss the effects of several parameters on the performance of the proposed algorithm in the following part. Table II summarizes the hyper-parameters used here. In particular, the discounted factor _γ_ is 0.1 for both units, and the learning rate _η_ 1 and _η_ 2 are set to 0.005 and 0.01, respectively. The batch size for DQN unit and PPO unit are 128 and 64, respectively. The exploration _ε_ in DQN unit is initialized to 1 and then drops to 0.01 at a rate of 0.995. The parameter _ϵ_ in clip operation defaults to 0.2 in PPO algorithm and the number of update times for new actor network in PPO unit is 20. 

## _B. Parameter Analysis_ 

In this section, we analyze the effects of different hyper-parameters on the performance of the proposed DPRA scheme, including the discounted factor _γ_ , learning rate _η_ 2 and batch size for PPO training. We first compare the influence of different _γ_ values. We increase _γ_ value from 0.1 to 0.9, and 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 10 -->

1350 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0010-02.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0010-03.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0010-04.png)


Fig. 4. The recorded sum rate during training process under different training parameters (curves smoothed by averaged window). (a) the discounted factor _γ_ . (b) the learning rate _η_ 2. (c) the batch size in PPO unit. 

the sum rate over the training process is presented in Fig. 4(a). We can see that the algorithm with _γ_ = 0.1 achieves the highest sum rate with great convergence while the algorithm with highest _γ_ value performs the worst performance. Fig. 4(a) showcases that a large discounted factor raises a negative effect on the learning ability of the proposed DPRA, which implies that future rewards provide little guidance for the agent to select action at current time. Furthermore, Fig. 4(b) shows the influence of learning rate _η_ 2 in PPO unit, form which we can see that the performance will decrease when the learning rate is too large or too small. One convincing explanation is that the learning rate with large value such as 0.1 here will cause the algorithm to linger at the optimal point, while the learning rate with small value such as 0.0001 here will lead to terrible convergence performance. Fig. 4(b) shows the best value of learning rate here is 0.01. Besides, the influence of batch size in PPO unit is presented in Fig. 4(c), which shows that as the training batch size increases, the algorithm converges slowly and the performance deteriorates. This is in line with the characteristics of neural network training. Considering that training with large batch consumes less time, we set the batch size to 64 here. 

## _C. Performance Analysis_ 

In this section, we compare the performance of the proposed DPRA with several baseline schemes mentioned above. We first present the sum rate results over the training time steps in Fig. 5, which shows that all schemes can converge after training a few time steps and the proposed DPRA scheme can achieve the best performance. Notice that the simulation results are averaged over 100 time steps to smooth the curve for clearer presentation. In particular, the comparison between DPRA and RPRA shows that the DRL-based method can learn the situation of channel quality and interference from historical information, so as to properly allocate RBs to improve the transmission rate. From the performance comparison among DPRA, DWRA and DQRA, we can prove the superiority of the proposed PPO-based algorithm in optimizing power. Besides, the results show that the proposed framework with RSMA technique outperforms NoRS and OMA systems in terms of transmission rate and spectral efficiency. 

Further, Fig. 6 presents the results of sum rate with respect to the maximum transmitting power of the LEO satellite. From 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0010-10.png)


Fig. 5. The recorded sum rate during training process (curves smoothed by averaged window). 

this figure, we can see that the sum rate increases with the transmitting power increasing, where the sum rate is linearly related to the maximum transmitting power in OMA system since there is no interference among users. The increment of sum rate decreases in RSMA system as the maximum transmitting power increases. Such a phenomenon can be explained by the fact that according to (3) and (4), the benefit of increasing the transmitting power will saturate when the power supply gets higher. Since the DQN algorithm needs to discretize the continuous action space into several discrete levels, the low performance is achieved in DQRA scheme, which is caused by the inadequate discretization of the action space. This figure also illustrates that the proposed DPRA scheme always achieves the best performance compared with other schemes, especially in the case of low transmitting power. 

Moreover, we investigate the performance of the proposed scheme with respect to the number of GSs sharing the same RB _K_ . Specifically, there are five GSs in total in the system and the number of GSs sharing the same RB varies from 2 to 4, i.e., _N_ = 5, _K_ = 2 _,_ 3 _,_ 4. The results of sum rate versus _K_ are shown in Fig. 7, from which we can see that the sum rate achieved by the RSMA based schemes increases with the number of GSs, while the sum rate achieved by NoRS 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 11 -->

1351 

HUANG et al.: DRL-BASED RESOURCE ALLOCATION FOR RSMA IN LEO SATELLITE-TERRESTRIAL NETWORKS 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0011-02.png)


Fig. 6. Sum rate vs. maximum transmitting power of LEO satellite. 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0011-04.png)


Fig. 7. Sum rate vs. number of GSs on each RB. 

and OMA schemes decreases. To explain this phenomenon, we have to know that the optimal strategy for RB allocation is to preferentially allocate the current RB to GSs with better channel quality. When we increase the number of GSs that can be accommodated on each RB, the channel quality of the newly joined GS is usually worse than that of the previous GSs. Thus, with the same transmitting power, the achievable rate of the newly joined GS is usually lower than that of the previous GSs and the bandwidth allocated to each GS also becomes smaller. This may cause the decrease of sum rate for OMA scheme. By contrast, the schemes with RSMA can mitigate the interference between GSs through proper power control, which enables the sum rate to increase, but the advantage will saturate when the number of GSs is large. This also proves the advantage of RSMA in terms of spectrum efficiency and achievable rate. Besides, the proposed DPRA scheme is superior to other RSMA based schemes. 

Fig. 8 presents the results of sum rate versus SIC detection threshold _θth_ . Since SIC technique is not used in NoRS and OMA systems, the sum rate does not vary with SIC detection threshold and we do not draw the curves for these 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0011-08.png)


Fig. 8. Sum rate vs. SIC detection threshold. 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0011-10.png)


Fig. 9. CDF of the power difference. 

two schemes. From this figure, we can see that the sum rate decreases in all schemes with the SIC detection threshold increasing. This is partly due to the fact that the probability of the GS successfully detecting common stream declines as the SIC detection threshold gets higher. In other words, only if the power difference between the common stream and the remaining signals is greater than the detection threshold, the common stream can be successfully demodulated and removed. As a result, the sum rate decreases with the SIC detection threshold increasing. But the proposed DPRA scheme still achieves a higher sum rate compared with other schemes. To further explain this phenomenon, we analyze the power difference defined in (9) for these three schemes. Specifically, we present the cumulative distribution function (CDF) of the power difference for DPRA, DQRA, and DWRA in Fig. 9. This figure shows that the CDF of the proposed DPRA scheme is better than that of other schemes, which illustrates that the proposed DPRA scheme has a higher tolerance for the power difference and is more suitable for receivers with high SIC detection threshold. 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 12 -->

1352 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0012-02.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0012-03.png)



![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0012-04.png)


Fig. 10. The distribution of power values for three schemes. (a) DPRA scheme. (b) DQRA scheme. (c) DWRA scheme. 

## _D. Analysis of Power Control Results_ 

This sub-section discusses the results of power control decisions made by the agent and thus we visualize the action values of power control after the algorithm converged. Specifically, the distribution diagrams of power values for three schemes are shown in Fig. 10. Here, we still assume that there are two RBs and three GSs, where each RB can accommodate two GSs at the same time. In Fig. 10, the horizontal axis represents the action set of power control, i.e., **_p_** = [ _pc, p_ 1 _, · · · , pN_ ], where _pc_ is the transmitting power assigned to the common stream and _pi_ ( _i_ = 1 _, · · · , N_ ) is the transmitting power assigned to the _i_ -th GS’s private stream. The vertical axis is the power value in dBm. And the decision results of the power control for two groups are represented by green and orange distribution curves, respectively, where the curve will bulge to both sides in the position where power values are dense. 

In particular, considering the results of the RB allocation, the first and third GSs share the first RB and form the group _G_ 1, while the first and second GSs share the second RB and 

form the group _G_ 2. Thus, from Fig. 10, we can see that the power value of _p_ 2 in group _G_ 1 and the power value of _p_ 3 in group _G_ 2 are very small in the three schemes. Besides, since the common stream needs decoding first according to the SIC technique, Fig. 10 also shows that the common stream has a stronger signal than private streams, which is consistent with the analysis of SIC detection in Section III. Moreover, in Fig. 10, comparing the proposed DPRA scheme with the other two schemes, we find that the remaining power is almost evenly distributed to each private stream in DQRA and DWRA schemes, while the power distribution for private streams is not fixed in one position in proposed DPRA scheme. Specifically, the agent sometimes evenly allocates remaining power to private streams, and sometimes allocates most of the remaining power to one private stream, which not only ensures the communication requirements of each user but also maximizes the sum rate as much as possible. Such a phenomenon also illustrates that the proposed DPRA scheme is more exploratory and flexible, which makes it easier to jump out of the suboptimal state and find a better solution. 

## VI. CONCLUSION 

This paper proposed a DRL-based resource allocation and power control approach for the considered RSMA based LEO satellite-terrestrial communication networks, which aims to maximize the sum rate of the system. Specifically, we modeled the considered joint optimization problem as an MDP and proposed a combined DRL framework, which employs DQN algorithm to optimize discrete RB allocation and PPO algorithm to optimize continuous power control. Finally, we evaluated the performance of the proposed scheme compared with several schemes. The simulation results showed the superiority of the proposed scheme in terms of transmission rate and spectral efficiency, which proves the advantages of DRL technique for resource optimization in RSMA based LEO satellite-terrestrial communications. For the future work, we put forward several interesting directions for further investigation. One is to extend the proposed framework to the LEO satellite communications with multiple antennas in which the advantage of RSMA mechanism is more significant. Another direction is to consider the scenario where multiple groups are allowed to share the same RB and thus the interference among groups cannot be ignored. 

## REFERENCES 

- [1] A. Vanelli-Coralli, A. Guidotti, T. Foggi, G. Colavolpe, and G. Montorsi, “5G and beyond 5G non-terrestrial networks: Trends and research challenges,” in _Proc. IEEE 3rd 5G World Forum (5GWF)_ , Sep. 2020, pp. 163–169. 

- [2] O. Kodheli et al., “Satellite communications in the new space era: A survey and future challenges,” _IEEE Commun. Surveys Tuts._ , vol. 23, no. 1, pp. 70–109, 1st Quart., 2021. 

- [3] S. Liu et al., “LEO satellite constellations for 5G and beyond: How will they reshape vertical domains?” _IEEE Commun. Mag._ , vol. 59, no. 7, pp. 30–36, Jul. 2021. 

- [4] M. Jia, X. Zhang, J. Sun, X. Gu, and Q. Guo, “Intelligent resource management for satellite and terrestrial spectrum shared networking toward B5G,” _IEEE Wireless Commun._ , vol. 27, no. 1, pp. 54–61, Feb. 2020. 

- [5] X. Wen, Y. Ruan, Y. Li, and R. Zhang, “Cognitive region design for overlay cognitive satellite terrestrial networks,” _IEEE Commun. Lett._ , vol. 25, no. 1, pp. 244–248, Jan. 2021. 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 13 -->

1353 

HUANG et al.: DRL-BASED RESOURCE ALLOCATION FOR RSMA IN LEO SATELLITE-TERRESTRIAL NETWORKS 

- [6] S. Han, J. Li, W. Meng, M. Guizani, and S. Sun, “Challenges of physical layer security in a satellite-terrestrial network,” _IEEE Netw._ , vol. 36, no. 3, pp. 98–104, May 2022. 

- [7] Y. Mao, B. Clerckx, and V. O. K. Li, “Rate-splitting multiple access for downlink communication systems: Bridging, generalizing, and outperforming SDMA and NOMA,” _EURASIP J. Wireless Commun. Netw._ , vol. 2018, no. 1, pp. 1–54, May 2018. 

- [8] B. Clerckx, Y. Mao, R. Schober, and H. V. Poor, “Rate-splitting unifying SDMA, OMA, NOMA, and multicasting in MISO broadcast channel: A simple two-user rate analysis,” _IEEE Wireless Commun. Lett._ , vol. 9, no. 3, pp. 349–353, Mar. 2020. 

- [9] G. Zhou, Y. Mao, and B. Clerckx, “Rate-splitting multiple access for multi-antenna downlink communication systems: Spectral and energy efficiency tradeoff,” _IEEE Trans. Wireless Commun._ , vol. 21, no. 7, pp. 4816–4828, Jul. 2022. 

- [10] H. Chen, D. Mi, B. Clerckx, Z. Chu, J. Shi, and P. Xiao, “Joint power and subcarrier allocation optimization for multigroup multicast systems with rate splitting,” _IEEE Trans. Veh. Technol._ , vol. 69, no. 2, pp. 2306–2310, Feb. 2020. 

- [11] Z. Yang, M. Chen, W. Saad, and M. Shikh-Bahaei, “Optimization of rate allocation and power control for rate splitting multiple access (RSMA),” _IEEE Trans. Commun._ , vol. 69, no. 9, pp. 5988–6002, Sep. 2021. 

- [12] L. Yin and B. Clerckx, “Rate-splitting multiple access for multigroup multicast and multibeam satellite systems,” _IEEE Trans. Commun._ , vol. 69, no. 2, pp. 976–990, Feb. 2021. 

- [13] Z. Lin, M. Lin, T. de Cola, J.-B. Wang, W.-P. Zhu, and J. Cheng, “Supporting IoT with rate-splitting multiple access in satellite and aerial-integrated networks,” _IEEE Internet Things J._ , vol. 8, no. 14, pp. 11123–11134, Jul. 2021. 

- [14] R. S. Sutton and A. G. Barto, _Reinforcement Learning: An Introduction_ . Cambridge, MA, USA: MIT Press, 2018. 

- [15] V. Mnih et al., “Human-level control through deep reinforcement learning,” _Nature_ , vol. 518, pp. 529–533, Feb. 2015. 

- [16] Y. Huang, C. Xu, C. Zhang, M. Hua, and Z. Zhang, “An overview of intelligent wireless communications using deep reinforcement learning,” _J. Commun. Inf. Netw._ , vol. 4, no. 2, pp. 15–29, Jun. 2019. 

- [17] N. C. Luong et al., “Applications of deep reinforcement learning in communications and networking: A survey,” _IEEE Commun. Surveys Tuts._ , vol. 21, no. 4, pp. 3133–3174, 4th Quart., 2019. 

- [18] C. He, Y. Hu, Y. Chen, and B. Zeng, “Joint power allocation and channel assignment for NOMA with deep reinforcement learning,” _IEEE J. Sel. Areas Commun._ , vol. 37, no. 10, pp. 2200–2210, Oct. 2019. 

- [19] S. Wang, T. Lv, W. Ni, N. C. Beaulieu, and Y. J. Guo, “Joint resource management for MC-NOMA: A deep reinforcement learning approach,” _IEEE Trans. Wireless Commun._ , vol. 20, no. 9, pp. 5672–5688, Sep. 2021. 

- [20] Y. Yuan, G. Zheng, K.-K. Wong, and K. B. Letaief, “Meta-reinforcement learning based resource allocation for dynamic V2X communications,” _IEEE Trans. Veh. Technol._ , vol. 70, no. 9, pp. 8964–8977, Sep. 2021. 

- [21] D. Zhou, M. Sheng, Y. Wang, J. Li, and Z. Han, “Machine learningbased resource allocation in satellite networks supporting Internet of Remote Things,” _IEEE Trans. Wireless Commun._ , vol. 20, no. 10, pp. 6606–6621, Oct. 2021. 

- [22] X. Hu, S. Liu, R. Chen, W. Wang, and C. Wang, “A deep reinforcement learning-based framework for dynamic resource allocation in multibeam satellite systems,” _IEEE Commun. Lett._ , vol. 22, no. 8, pp. 1612–1615, Aug. 2018. 

- [23] Md. Z. Hassan, Md. J. Hossain, J. Cheng, and V. C. M. Leung, “Joint throughput-power optimization of fog-RAN using rate-splitting multiple access and reinforcement-learning based user clustering,” _IEEE Trans. Veh. Technol._ , vol. 70, no. 8, pp. 8019–8036, Aug. 2021. 

- [24] N. Q. Hieu, D. T. Hoang, D. Niyato, and D. I. Kim, “Optimal power allocation for rate splitting communications with deep reinforcement learning,” _IEEE Wireless Commun. Lett._ , vol. 10, no. 12, pp. 2820–2823, Dec. 2021. 

- [25] F. Meng, P. Chen, L. Wu, and J. Cheng, “Power allocation in multiuser cellular networks: Deep reinforcement learning approaches,” _IEEE Trans. Wireless Commun._ , vol. 19, no. 10, pp. 6255–6267, Oct. 2020. 

- [26] Z. Gao, A. Liu, and X. Liang, “Data transmission time minimization for LEO satellite-terrestrial integrated networks,” in _Proc. Int. Conf. Wireless Commun. Signal Process. (WCSP)_ , Oct. 2020, pp. 642–647. 

- [27] C. Han, A. Liu, H. Wang, L. Huo, and X. Liang, “Dynamic anti-jamming coalition for satellite-enabled army IoT: A distributed game approach,” _IEEE Internet Things J._ , vol. 7, no. 11, pp. 10932–10944, Nov. 2020. 

- [28] M. S. Ali, E. Hossain, A. Al-Dweik, and D. I. Kim, “Downlink power allocation for CoMP-NOMA in multi-cell networks,” _IEEE Trans. Commun._ , vol. 66, no. 9, pp. 3982–3998, Sep. 2018. 

- [29] J. Ge, Y.-C. Liang, J. Joung, and S. Sun, “Deep reinforcement learning for distributed dynamic MISO downlink-beamforming coordination,” _IEEE Trans. Commun._ , vol. 68, no. 10, pp. 6070–6085, Oct. 2020. 

- [30] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017, _arXiv:1707.06347_ . 

- [31] L. Yin and B. Clerckx, “Rate-splitting multiple access for multibeam satellite communications,” in _Proc. IEEE Int. Conf. Commun. Workshops (ICC Workshops)_ , Jun. 2020, pp. 1–6. 

- [32] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” 2014, _arXiv:1412.6980_ . 

**Jingfei Huang** received the B.Eng. degree in communication engineering from the Beijing University of Posts and Telecommunications, Beijing, China, in 2021, where she is currently pursuing the master’s degree. 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0013-30.png)


Her research interests include wireless resource allocation, machine learning, and AI-based wireless networks. 

**Yang Yang** (Member, IEEE) received the master’s and Ph.D. degrees from the School of Telecommunications Engineering, Beijing University of Posts and Telecommunications (BUPT), Beijing, China, in 2012 and 2015, respectively. From 2015 to 2017, he was a Post-Doctoral Researcher with the Tsinghua National Laboratory for Information Science and Technology (TNList), Department of Electronic Engineering, Tsinghua University, Beijing. He is currently an Associate Professor with the School of Artificial Intelligence, BUPT. His research interests include AI-based wireless networks, machine learning, and ultra-dense networks. 

He was a recipient of the Exemplary Reviewer of IEEE TRANSACTIONS ON COMMUNICATIONS in 2018. 

**Jemin Lee** (Member, IEEE) received the B.S. (Hons.), M.S., and Ph.D. degrees in electrical and electronic engineering from Yonsei University, Seoul, South Korea, in 2004, 2007, and 2010, respectively. She was a Post-Doctoral Fellow with the Massachusetts Institute of Technology (MIT), Cambridge, MA, USA, from 2010 to 2013, and a Temasek Research Fellow with the iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD), Singapore, from 2014 to 2016. She was an Associate Profes- 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0013-35.png)


sor with the Department of Electrical Engineering and Computer Science, Daegu Gyeongbuk Institute of Science and Technology (DGIST), Daegu, South Korea, from 2016 to 2021, and an Associate Professor with the Department of Electronic and Electrical Engineering, Sungkyunkwan University (SKKU), Suwon, South Korea, from 2021 to 2023. Currently, she is an Associate Professor with the School of Electrical and Electronic Engineering, Yonsei University. Her current research interests include wireless communications, communication security, intelligent networking, and blockchain technology. 

She received the Haedong Young Engineering Researcher Award in 2020, the IEEE ComSoc Young Author Best Paper Award in 2020, the IEEE ComSoc AP Outstanding Paper Award in 2017, the IEEE ComSoc AP Outstanding Young Researcher Award in 2014, the Temasek Research Fellowship in 2013, and the Chun-Gang Outstanding Research Award in 2011. She served as the Chair for the IEEE Communication Society (ComSoc) Radio Communications Technical Committee (RCC) from 2021 to 2022. She is currently an Area Editor of IEEE TRANSACTIONS ON MACHINE LEARNING IN COMMUNICATIONS AND NETWORKING and an Editor for IEEE TRANSACTIONS ON COMMUNICATIONS, the IEEE WIRELESS COMMUNICATIONS LETTERS, and the IEEE OPEN JOURNAL OF VEHICULAR TECHNOLOGY. She was an Editor of IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS and the IEEE COMMUNICATION LETTERS from 2014 to 2019. 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.

<!-- Page 14 -->

1354 

IEEE TRANSACTIONS ON COMMUNICATIONS, VOL. 72, NO. 3, MARCH 2024 


![](RSMA_DRL_Resource_Allocation_LEO_STN_images/RSMA_Deep_Reinforcement_Learning-Based_Resource_Allocation_for_RSMA_in_LEO_Satellite-Terrestrial_Networks.pdf-0014-02.png)


**Dazhong He** (Member, IEEE) received the B.E. and M.E. degrees in information engineering and the Ph.D. degree in information and communication engineering from the Beijing University of Posts and Telecommunications (BUPT) in 1994, 1997, and 2020, respectively. 

He is currently a Senior Engineer with the School of Artificial Intelligence, BUPT. His current research interests include network traffic research, big data analysis, and hardware-based deep learning. 

**Yonghui Li** (Fellow, IEEE) received the Ph.D. degree in communication engineering from the Beijing University of Aeronautics and Astronautics, Beijing, China, in 2002. 

Since 2003, he has been with the Centre of Excellence in Telecommunications, The University of Sydney, Australia, where he is currently a Professor and the Director of the Wireless Engineering Laboratory, School of Electrical and Information Engineering. He was a recipient of the Australian Research Council (ARC) Queen Elizabeth II Fellowship in 2008 and ARC Future Fellowship in 2012. He is a fellow of IEEE for contributions to cooperative communications technologies. His current research interests are in the area of wireless communications, with a particular focus on the IoT, machine to machine communications, MIMO, millimeter wave communications, channel coding techniques, game theory, machine learning, and signal processing. He holds a number of patents granted and pending in these fields. 

Prof. Li is an Editor of IEEE TRANSACTIONS ON COMMUNICATIONS and IEEE TRANSACTIONS ON VEHICULAR TECHNOLOGY and a Guest Editor of several special issues of IEEE journals, such as IEEE JOURNAL ON SELECTED AREAS IN COMMUNICATIONS, IEEE INTERNET OF THINGS JOURNAL, IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS, and _IEEE Communications Magazine_ . He currently serves as the Specialty Chief Editor for _Frontiers Signal Processing_ journal. He received the best paper awards from IEEE International Conference on Communications (ICC) in 2014, IEEE PIMRC 2017, and IEEE Wireless Days Conferences (WD) in 2014. 

Authorized licensed use limited to: CHONGQING UNIV OF POST AND TELECOM. Downloaded on April 18,2024 at 02:13:56 UTC from IEEE Xplore.  Restrictions apply.
