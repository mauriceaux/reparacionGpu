from scp.SCPProblem import SCPProblem
from scp.repair.ReparaStrategy import ReparaStrategy
import cumpleRestricciones as cumpleGPU
import numpy as np
import datetime 


#problema = SCPProblem(f'scp/instances/off/scp0.txt')
problema = SCPProblem(f'scp/instances/mscp42.txt')
#problema = SCPProblem(f'scp/instances/mscpnrg2.txt')
#problema = SCPProblem(f'scp/instances/off/scpnrh5.txt')


repara = ReparaStrategy(problema.instance.get_r()
                            ,problema.instance.get_c()
                            ,problema.instance.get_rows()
                            ,problema.instance.get_columns())

pondRestricciones = 1/np.sum(problema.instance.get_r(), axis=1)
#print(pondRestricciones)
#exit()
#print(f"columnas {problema.instance.get_columns()}")
#exit()
#sols = np.zeros((50, problema.getNumDim()), dtype=np.float)
#sols = [ [1.,1.,1.,0.,1.,1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,0.,0.,0.,1.,1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]]
#sols = [ [1.,1.,1.,0.,1.,1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.,1.,0.,0.,0.,1.,0.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,0.,1.,0.,1.,1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,1.,0.,0.,1.,0.,0.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]]
for i in range(5):
    nsol = 1
    print(f"creando {nsol} soluciones al azar")
    #sols = problema.generarSolsAlAzar(nsol)
    #sols = np.random.randint(0,2,(nsol,problema.instance.get_columns()), dtype=np.int8)
    sols = np.zeros((nsol,problema.instance.get_columns()), dtype=np.int8)
    #sols = np.ones((nsol,problema.instance.get_columns()), dtype=np.int8)
    #print(f"{np.random.randint(0,problema.instance.get_columns(),(nsol,3), dtype=np.int16)}")
    #print(f"{sols[np.arange(sols.shape[0]),np.random.randint(0,problema.instance.get_columns(),(3,nsol), dtype=np.int16)]}")
    sols[np.arange(sols.shape[0]),np.random.randint(0,sols.shape[1],(3,nsol), dtype=np.int16)]=1
    #print(sols)
    #exit()
    #sols = np.ones((nsol,problema.instance.get_columns()), dtype=np.int8)
    #print(sols)
    npsols = np.array(sols)
    #print(f"fin creando soluciones")
    #cumple=[]
    reparadasCpu = []
    #print(f"revisando factibilidad")
    
    inicio = datetime.datetime.now()

    for i in range(npsols.shape[0]):
        #print(npsols[i,:])
        #print(repara.cumple(npsols[i,:]))
        #cumple.append(repara.cumple(npsols[i,:]))
        reparadasCpu.append(repara.repara(npsols[i,:])[0])
    reparadasCpu = np.array(reparadasCpu)
    
    #print(reparadasCpu)
    #exit()
    fin = datetime.datetime.now()
    print(f"reparacion cpu \t{fin-inicio}")
    
    #sols = repara.reparaBatch(sols)
    #fitness, decoded, _ = problema.evalDecBatch(sols, None)
    #fitness, de(coded, _ = problema.evalEnc(sols[1])
    #cumple = np.array(cumple)
    #print(cumple)

    inicio = datetime.datetime.now()
    reparadasGpu = cumpleGPU.reparaSoluciones(sols, problema.instance.get_r(), problema.instance.get_c(), pondRestricciones)

    fin = datetime.datetime.now()

    print(f"reparacion gpu \t{fin-inicio}")
    cumplidasCpu = []
    fitnessCpu = []
    cumplidasGpu = []
    fitnessGpu = []
    for i in range(npsols.shape[0]):
        cumplidasCpu.append(repara.cumple(reparadasCpu[i].tolist()))
        cumplidasGpu.append(repara.cumple(reparadasGpu[i].tolist()))
        fitnessCpu.append(problema.evalInstance(reparadasCpu[i].tolist()))
        fitnessGpu.append(problema.evalInstance(reparadasGpu[i].tolist()))
    print(f"repara cpu cumple? {(np.array(cumplidasCpu) == 1).all()}")
    print(f"repara gpu cumple? {(np.array(cumplidasGpu) == 1).all()}")
    print(f"fitness promedio cpu cumple? {np.average(np.array(fitnessCpu))}")
    print(f"fitness promedio gpu cumple? {np.average(np.array(fitnessGpu))}")
    print(f"fitness devstd cpu {np.std(np.array(fitnessCpu))}")
    print(f"fitness devstd gpu {np.std(np.array(fitnessGpu))}")
    #print(np.prod(cumplidas, axis=1))

    #sumaRes = np.array(problema.instance.get_r()).sum(axis=1)

    #print(f"restricciones \n{np.array(problema.instance.get_r()).sum(axis=1)}")

    #print(f"resultado cpu correcto? {(sumaRes==cumple).all()}")
    #print(f"resultado gpu correcto? {(cumple==np.prod(cumplidas, axis=1)).all()}")