from scp.SCPProblem import SCPProblem
from scp.repair.ReparaStrategy import ReparaStrategy
import cumpleRestricciones as cumpleGPU
import numpy as np
import datetime 


problema = SCPProblem(f'scp/instances/off/scp0.txt')
#problema = SCPProblem(f'scp/instances/mscp42.txt')
#problema = SCPProblem(f'scp/instances/off/scpnrh5.txt')


repara = ReparaStrategy(problema.instance.get_r()
                            ,problema.instance.get_c()
                            ,problema.instance.get_rows()
                            ,problema.instance.get_columns())
#print(f"columnas {problema.instance.get_columns()}")
#exit()
#sols = np.zeros((50, problema.getNumDim()), dtype=np.float)
#sols = [ [1.,1.,1.,0.,1.,1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.,1.,0.,0.,1.,1.,0.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,0.,0.,0.,1.,1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]]
#sols = [ [1.,1.,1.,0.,1.,1.,0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,1.,1.,1.,1.,0.,1.,1.,0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,0.,1.,1.,1.,1.,1.,0.,1.,0.,1.,0.,0.,1.,1.,0.,0.,0.,1.,0.,1.,0.,0.,1.,1.,1.,0.,0.,1.,1.,1.,0.,0.,1.,0.,1.,0.,1.,1.,0.,0.,1.,0.,1.,0.,1.,0.,0.,0.,0.,1.,0.,1.,0.,0.,1.,0.,0.,1.,0.,0.,0.,1.,1.,1.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,1.,1.,0.,1.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]]
for i in range(1):
    nsol = 2
    print(f"creando {nsol} soluciones al azar")
    #sols = problema.generarSolsAlAzar(nsol)
    #sols = np.random.randint(0,2,(nsol,problema.instance.get_columns()), dtype=np.int8)
    sols = np.zeros((nsol,problema.instance.get_columns()), dtype=np.int8)
    #sols = np.ones((nsol,problema.instance.get_columns()), dtype=np.int8)
    #print(sols)
    npsols = np.array(sols)
    #print(f"fin creando soluciones")
    cumple=[]
    #print(f"revisando factibilidad")
    inicio = datetime.datetime.now()
    for i in range(npsols.shape[0]):
        #print(npsols[i,:])
        #print(repara.cumple(npsols[i,:]))
        cumple.append(repara.cumple(npsols[i,:]))
    fin = datetime.datetime.now()
    print(f"factibilidad cpu \t{fin-inicio}")
    #sols = repara.reparaBatch(sols)
    #fitness, decoded, _ = problema.evalDecBatch(sols, None)
    #fitness, de(coded, _ = problema.evalEnc(sols[1])
    cumple = np.array(cumple)
    #print(cumple)

    inicio = datetime.datetime.now()
    cumplidas = cumpleGPU.reparaSoluciones(sols, problema.instance.get_r(), problema.instance.get_c())

    fin = datetime.datetime.now()

    print(f"factibilidad gpu \t{fin-inicio}")
    #print(np.prod(cumplidas, axis=1))

    #sumaRes = np.array(problema.instance.get_r()).sum(axis=1)

    #print(f"restricciones \n{np.array(problema.instance.get_r()).sum(axis=1)}")

    #print(f"resultado cpu correcto? {(sumaRes==cumple).all()}")
    #print(f"resultado gpu correcto? {(cumple==np.prod(cumplidas, axis=1)).all()}")