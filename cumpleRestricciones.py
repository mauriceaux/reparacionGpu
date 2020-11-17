import numpy as np
from numba import cuda, float32, jit, uint8, int8, uint16, int32
import numba
import math
import sys
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

#cumple reparaSoluciones
#recibe (todos numpy array)
#matriz de n soluciones con m columnas 
#matriz de o restricciones con m columnas
#lista de pesos con m columnas
#retorna
#lista binaria de n x o elementos donde un 1 representa que la solucion n cumple las restricciones, 0 si no

NSOL = 10
MRES = 100
COL = 100
DETERMINISTA = 0.8


def reparaSoluciones(soluciones, restricciones, pesos, pondRestricciones):
    soluciones = np.array(soluciones, dtype=np.int8)
    restricciones = np.array(restricciones, dtype=np.int8)
    n,m = soluciones.shape
    assert m == restricciones.shape[1], f"numero de columnas distinto en soluciones {m} y restricciones {restricciones.shape[1]}"
    factibilidad = _procesarFactibilidadGPU(soluciones, restricciones)
    columnas = np.arange(soluciones.shape[0])
    cont = 0
    while (factibilidad == 0).any():        
    
        assert factibilidad.shape[0] == n, f"numero de factibilidades {factibilidad.shape[0]} distinto de numero de soluciones {n}"
        assert factibilidad.shape[1] == restricciones.shape[0], f"numero de restricciones en factibilidades {factibilidad.shape[1]} distinto de numero de restricciones {restricciones.shape[0]}"
    
        # ELEGIR RESTRICCIONES



        infactibles = factibilidad.copy()
        infactibles = infactibles.astype(float)
        infactibles[infactibles > 0] = -np.inf
        infactibles[infactibles == 0] = 1
        infactiblesPonderadas = infactibles*pondRestricciones
        infactiblesPonderadas[factibilidad==1] = -np.inf
        # print(f"infactibles * pondRestricciones {infactiblesPonderadas}")
        # print(f"infactiblesPonderadas {infactiblesPonderadas}")
        nCols = 10
        # infactiblesSel = np.argpartition(infactiblesPonderadas,nCols,axis=1)[:,:nCols]
        #infactiblesSel = np.argsort(infactiblesPonderadas, axis=1)[::-1][:,:nCols]
        infactiblesSel = np.argsort(-infactiblesPonderadas, axis=1)[:,:nCols]
        # print(f"infactibles seleccionadas {infactiblesSel}")
        # exit()
        # print(infactiblesPonderadas[np.arange(infactiblesPonderadas.shape[0]).reshape(-1,1),infactiblesSel])
        iPondInfactibles = infactiblesPonderadas[np.arange(infactiblesPonderadas.shape[0]).reshape(-1,1),infactiblesSel] > 0
        # iPondInfactibles = np.argwhere(infactiblesPonderadas[np.arange(infactiblesPonderadas.shape[0]).reshape(-1,1),infactiblesSel] < np.inf)
        # print(f"infactibles efectivamente infactibles {iPondInfactibles}")
        # exit()
        factibilidadTmp = np.ones(factibilidad.shape)
        for idx in np.argwhere(iPondInfactibles):
            # print(idx)
            factibilidadTmp[idx[0], infactiblesSel[idx[0],idx[1]]] = 0
        factibilidadTmp[factibilidad==1] = 1
        # print(factibilidadTmp)
        # print(f"infactibles seleccionadas {infactiblesSel[iPondInfactibles[:,0].reshape(-1,1), iPondInfactibles[:,1].reshape(-1,1)]}")

        # factibilidadTmp[np.arange(factibilidadTmp.shape[0]).reshape(-1,1), infactiblesSel[iPondInfactibles]] = 0
        # print(factibilidadTmp)

        # print(f"total infactibles: {np.count_nonzero(factibilidadTmp == 0)}")
        # print(f"total infactibles seleccionadas: {np.count_nonzero(iPondInfactibles)}")

        # # factibilidad(np.arange(factibilidad.shape[0]), infactiblesSel)
        # exit()

        ponderaciones = _ponderarColsReparar(restricciones, factibilidadTmp, pesos, pondRestricciones)
        ponderaciones[ponderaciones==0] = np.inf
        
        idxSolsInfactibles = np.any(factibilidadTmp==0, axis=1)
        # print(f"idxSolsInfactibles {idxSolsInfactibles}")
        nCols = 10
        # colsElegidas = np.argpartition(ponderaciones,nCols,axis=1)[:,:nCols]
        # print(f"colsElegidas 1 {colsElegidas}")
        # print(f"ponderaciones {ponderaciones}")
        colsElegidas = np.argsort(ponderaciones,axis=1)[:,:nCols]
        # print(f"colsElegidas 2 {colsElegidas}")
        # exit()
        # INDICES SOLUCIONES A REPARAR DETERMINISTA Y RANDOM
        random = np.random.uniform(size=(idxSolsInfactibles[idxSolsInfactibles==True].shape)) < DETERMINISTA
        idxDeterministas = idxSolsInfactibles.copy()
        idxDeterministas[idxSolsInfactibles] = random
        idxNoDeterministas = idxSolsInfactibles.copy()
        idxNoDeterministas[idxSolsInfactibles] = ~random
        # print(f"idxDeterministas {idxDeterministas}")
        # print(f"idxNoDeterministas {idxNoDeterministas}")
        
        # MEJOR COLUMNA REPARAR
        # mejorColumna = np.argmin(ponderaciones[idxSolsInfactibles,colsElegidas.T[:,idxSolsInfactibles]].T, axis=1)
        mejorColumna = colsElegidas[:,0]
        # print(f"colsElegidas {colsElegidas}")
        # print(f"mejorColumna {mejorColumna}")
        # print(f"colsElegidas[mejorColumna] {colsElegidas[mejorColumna]}")
        # COLUMNAS RANDOM REPARAR
        ponderacionesElegidasRandom = ponderaciones[np.argwhere(idxNoDeterministas),colsElegidas[idxNoDeterministas]]
        idcolNoInf = np.argwhere(ponderacionesElegidasRandom!=np.inf)
        idcolNoInfRandom = [np.random.choice(idcolNoInf[idcolNoInf[:,0]==pos][:,1]) for pos in range(ponderacionesElegidasRandom.shape[0])]
        colsElegidasRandom = colsElegidas[idxNoDeterministas][np.arange(colsElegidas[idxNoDeterministas].shape[0]),idcolNoInfRandom]
        columnas = np.arange(idxNoDeterministas[idxNoDeterministas==True].shape[0]).reshape(-1,1)

        # VALIDACION, NO SE DEBEN REPARAR COLUMNAS QUE CONTIENEN 1
        if (idxDeterministas[idxSolsInfactibles == True] == idxNoDeterministas[idxSolsInfactibles == True]).any():
            raise Exception(f"No se eligieron bien las columnas a reparar")
        if (~idxDeterministas[idxSolsInfactibles == True]).all() and (~idxNoDeterministas[idxSolsInfactibles == True]).all():
            raise Exception(f"No se eligio ninguna columna a reparar")
        if(soluciones[idxDeterministas,mejorColumna[idxDeterministas]] == 1).any():
            # print(f"soluciones {soluciones}")
            # print(f"colsElegidas {colsElegidas}")
            # print(f"mejor columna {mejorColumna}")
            # print(f"idxDeterministas {idxDeterministas}")
            # print(f"idxSolsInfactibles {idxSolsInfactibles}")
            raise Exception(F"Mejores columnas mal elegidas")

        if(soluciones[np.argwhere(idxNoDeterministas),colsElegidasRandom.reshape((-1,1))] == 1).any():
            raise Exception(f"Columnas random mal elegidas")

        # print(np.argwhere(idxNoDeterministas))
        # exit()
        # COLUMNAS ELEGIDAS
        # i=0
        # j = 0
        # print(f"sols infactibles {idxSolsInfactibles}")
        # print(f"random {random}")
        # for idx in range(idxSolsInfactibles.shape[0]):
        #     if not idxSolsInfactibles[idx]: continue
        #     colsInfactibles = [col for col in colsElegidas[idx] if ponderaciones[idx, col] < np.inf]
        #     pondInfactibles = [ponderaciones[idx,col] for col in range(ponderaciones[idx].shape[0]) if ponderaciones[idx, col] < np.inf]
        #     # print(f"solucion infactible {i}")
        #     # print(colsElegidasRandom)
        #     # print(colsElegidas)
        #     # print(mejorColumna)
        #     # print(idx)
        #     colElegida = 0
        #     if random[i]:
        #         colElegida = colsElegidas[idx,mejorColumna[i]]
        #     else:
        #         colElegida = colsElegidasRandom[j]
        #         j+= 1
        #     # print(f"ponderaciones gpu columnas {colsInfactibles} ponderaciones {pondInfactibles} col elegida {colElegida}")
        #     print(f"gpu col elegida {colElegida} ponderacion {ponderaciones[idx]} elegidas {colsElegidas[idx]}")
        #     exit()
        #     # print(f"gpu ponderacion {ponderaciones}")
        #     i += 1
        # PONDERACIONES ELEGIDAS
        # print(f"idxSolsInfactibles {idxSolsInfactibles}")
        # print(f"idxDeterministas {idxDeterministas}")
        # print(f"mejorColumna {mejorColumna}")
        # print(f"colsElegidas {colsElegidas}")
        # print(f"elegidas {mejorColumna[idxDeterministas[idxSolsInfactibles]]}")
        # print(f"fin iteracion")
        soluciones[idxDeterministas,mejorColumna[idxDeterministas]] = 1
        soluciones[np.argwhere(idxNoDeterministas),colsElegidasRandom.reshape((-1,1))] = 1
        factibilidad = _procesarFactibilidadGPU(soluciones, restricciones)
    return soluciones
    

def _procesarFactibilidadGPU(soluciones, restricciones):

    restriccionesCumplidas = np.zeros((soluciones.shape[0], restricciones.shape[0]), dtype=np.uint16)
    #iniciar kernel
    threadsperblock = (NSOL, MRES)
    blockspergrid_x = int(math.ceil(soluciones.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(restricciones.shape[0] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    sol_global_mem = cuda.to_device(soluciones)
    rest_global_mem = cuda.to_device(restricciones)
    resultado_global_mem = cuda.to_device(restriccionesCumplidas)

    #llamar kernel
    kernelFactibilidadGPU[blockspergrid, threadsperblock](sol_global_mem,rest_global_mem,resultado_global_mem)

    return resultado_global_mem.copy_to_host()

def _ponderarColsReparar(restricciones, factibilidad, pesos, pondRestricciones):
    ponderaciones = np.zeros((factibilidad.shape[0], restricciones.shape[1]), dtype=np.float32)
    #iniciar kernel
    threadsperblock = (NSOL, COL)
    blockspergrid_x = int(math.ceil(factibilidad.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(restricciones.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    sol_global_mem = cuda.to_device(restricciones)
    fact_global_mem = cuda.to_device(factibilidad)
    pondRestricciones_mem = cuda.to_device(pondRestricciones)
    pesos_mem = cuda.to_device(pesos)
    resultado_global_mem = cuda.to_device(ponderaciones)
    
    #llamar kernel
    kernelPonderarGPU[blockspergrid, threadsperblock](sol_global_mem,fact_global_mem,pesos_mem, pondRestricciones_mem, resultado_global_mem)

    return resultado_global_mem.copy_to_host()

def _calcularColsReparar(ponderaciones):
    resultado = np.zeros((ponderaciones.shape[0], ponderaciones.shape[1]), dtype=np.uint8)
    #colsCandidatas = np.zeros((ponderaciones.shape[0], ponderaciones.shape[1]), dtype=np.uint8)
    colsCandidatasGlobal = np.ones((ponderaciones.shape[0], 10), dtype=np.int32) * -1
    rng_states = create_xoroshiro128p_states(COL, seed=1)
    ponderacionMaxima = np.array([np.max(ponderaciones)])
    print(f"ponderacion maxima {ponderacionMaxima}")
    #iniciar kernel
    threadsperblock = (NSOL, COL)
    blockspergrid_x = int(math.ceil(ponderaciones.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(ponderaciones.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    ponderaciones_global_mem = cuda.to_device(ponderaciones)
    resultado_global_mem = cuda.to_device(resultado)
    colsCandidatasGlobal_mem = cuda.to_device(colsCandidatasGlobal)
    poderacionMaxima_mem = cuda.to_device(ponderacionMaxima)
    rng_states_mem = cuda.to_device(rng_states)


    #llamar kernel
    kernelColsCandidatasGPU[blockspergrid, threadsperblock](ponderaciones_global_mem, poderacionMaxima_mem, colsCandidatasGlobal,rng_states_mem, resultado_global_mem)

    return colsCandidatasGlobal_mem.copy_to_host()

@cuda.jit
def kernelFactibilidadGPU(soluciones, restricciones, resultado):
    
    #leer soluciones y restricciones a procesar
    solTmp = cuda.shared.array(shape=(NSOL, COL), dtype=uint8)
    restTmp = cuda.shared.array(shape=(MRES, COL), dtype=uint8)
    resultadoTmp = cuda.shared.array(shape=(NSOL, MRES), dtype=uint8)
    solIdx, restIdx = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if solIdx >= soluciones.shape[0]: return
    if restIdx >= restricciones.shape[0]: return
    
    tmp = 0
    numGCols = int(math.ceil(soluciones.shape[1]/COL))
    for gcol in range(numGCols):
        colInicio = gcol*COL

        for c in range(COL):
            col = colInicio+c
            if col >= soluciones.shape[1]: break
            tmp += soluciones[solIdx,col] * restricciones[restIdx, col]
            if tmp > 0: break
        

        cuda.syncthreads()
        if tmp > 0: break

    resultado[solIdx, restIdx] = tmp


@cuda.jit
def kernelPonderarGPU(restricciones, factibilidad, pesos, pondRestricciones, cReparar):
    restTmp = cuda.shared.array(shape=(COL), dtype=float32)
    pesosTmp = cuda.shared.array(shape=(COL), dtype=float32)
    infactTmp = cuda.shared.array(shape=(NSOL), dtype=float32)
    pondRestriccionesTmp = cuda.shared.array(shape=(1), dtype=float32)
    solIdx, colIdx = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    if solIdx >= cReparar.shape[0]: return
    if colIdx >= cReparar.shape[1]: return
    tmp = 0

    if tx == 0:
        pesosTmp[ty] = pesos[colIdx] 
    cuda.syncthreads()

    for res in range(restricciones.shape[0]):
        if tx == 0 and ty == 0:
            pondRestriccionesTmp[0] = pondRestricciones[res]

        if tx == 0:
            restTmp[ty] = restricciones[res,colIdx]
            
        if ty == 0:
            infactTmp[tx] = factibilidad[solIdx,res]
        cuda.syncthreads()

        if infactTmp[tx] == 0:
            # tmp += restTmp[ty] + pondRestriccionesTmp[0]
            tmp += restTmp[ty]
        cuda.syncthreads()
    if tmp > 0:
        cReparar[solIdx,colIdx] = ( pesosTmp[ty] / tmp )


@cuda.jit()
def kernelColsCandidatasGPU(ponderacion, ponderacionMax, colsCandidatasGlobal, rng_states, resultado):
    ponderacionTmp = cuda.shared.array(shape=(NSOL,COL), dtype=float32)
    colsCandidatasBloque = cuda.shared.array(shape=(NSOL, COL), dtype=int32)
    pondCandidatasBloque = cuda.shared.array(shape=(NSOL, COL), dtype=float32)
    colsCandidatasGlobalTmp = cuda.shared.array(shape=(NSOL, 10), dtype=int32)
    pondCandidatasGlobalTmp = cuda.shared.array(shape=(NSOL, 10), dtype=float32)
    solIdx, colIdx = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if solIdx >= ponderacion.shape[0]: return
    if colIdx >= ponderacion.shape[1]: return
    min = -1
    for i in range(int(ponderacion.shape[1]/COL)):
        ponderacionTmp[tx,ty] = ponderacion[solIdx, ty+i*COL]
        if ponderacionTmp[tx,ty] == 0: 
            resultado[solIdx,ty+i*COL] = 0
            continue
        cuda.syncthreads()
        if min == -1 or ponderacionTmp[tx, ty] < ponderacionTmp[tx, min]:
            min = colIdx
    
    colsCandidatasBloque[tx,ty] = min
    pondCandidatasBloque[tx,ty] = ponderacionTmp[tx, min]
    
    if ty < 10:
        colsCandidatasGlobalTmp[tx, ty] = colsCandidatasGlobal[solIdx, ty]
        pondCandidatasGlobalTmp[tx,ty] = ponderacion[solIdx, colsCandidatasGlobalTmp[tx, ty]]

    cuda.syncthreads()

    if ty==0 and tx==0:
        print(ponderacionTmp)
        return
    if ty == 0:
        
        for i in range(COL):
            #para cada minimo del bloque
            for j in range(pondCandidatasGlobalTmp.shape[1]):
                #comparado con cada valor de minimos globales
                if pondCandidatasGlobalTmp[tx,j] == -1 or pondCandidatasBloque[tx,i] < pondCandidatasGlobalTmp[tx,j]:
                    #si el minimo global no esta asignado o es mayor al minimo del bloque se inserta el minimo del bloque en 
                    #la posicion del global
                    for a in range(pondCandidatasGlobalTmp.shape[1]-1, j, -1):
                        colsCandidatasGlobalTmp[tx, a] = colsCandidatasGlobalTmp[tx, a-1] 
                        pondCandidatasGlobalTmp[tx, a] = pondCandidatasGlobalTmp[tx, a-1] 
                    colsCandidatasGlobalTmp[tx, j] = colsCandidatasBloque[tx, i] 
                    pondCandidatasGlobalTmp[tx, j] = pondCandidatasBloque[tx, i] 
                    
            
        for i in range(pondCandidatasGlobalTmp.shape[1]):
            colsCandidatasGlobal[tx, i] = colsCandidatasGlobalTmp[tx, i]

        cuda.syncthreads()
