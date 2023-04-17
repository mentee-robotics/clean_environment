import torch
import pytorch_kinematics as pk

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

chain = pk.build_serial_chain_from_urdf(open("/home/raphael/PycharmProjects/arms_w_gripper/assets/urdf/happybot_arms_v1_0/urdf/happy_arms_gym_no_encoding.urdf").read(), "right_gripper_ee")
chain = chain.to(dtype=dtype, device=d)

N = 1000
th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)
inp = torch.rand(1, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)
# order of magnitudes faster when doing FK in parallel
# elapsed 0.008678913116455078s for N=1000 when parallel
# (N,4,4) transform matrix; only the one for the end effector is returned since end_only=True by default
tg_batch = chain.forward_kinematics(inp)
print(tg_batch)

inp[:, -1] = 0
tg_batch = chain.forward_kinematics([0., 0., 0., 0., 0])
print(tg_batch)