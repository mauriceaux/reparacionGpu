import cumpleRestricciones as cumpleGPU

problema = SCPProblem(f'scp/instances/mscpnrg2.txt')

pondRestricciones = 1/np.sum(problema.instance.get_r(), axis=1)

reparadasGpu = cumpleGPU.reparaSoluciones(sols, problema.instance.get_r(), problema.instance.get_c(), pondRestricciones)
