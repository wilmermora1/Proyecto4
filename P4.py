# ---
# 
# ## Universidad de Costa Rica
# ### Escuela de Ingeniería Eléctrica
# #### IE0405 - Modelos Probabilísticos de Señales y Sistemas
# 
# ---
# 
# * Estudiante: **Wilmer Daniel Mora Pereira**
# * Carné: **B95188**
# * Grupo: **2**
# 
# ---
# # `P4` - *Modulación digital IQ*
# 
# ---
# * Elaboración de nota teórica y demostración: **Jeaustin Sirias Chacón**, como parte de IE0499 - Proyecto Eléctrico: *Estudio y simulación de aplicaciones de la teoría de probabilidad en la ingeniería eléctrica*.
# * Revisión y edición: **Fabián Abarca Calderón**


from PIL import Image
import numpy as np


#####################################################
# Obtención de la imagen

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)


#####################################################
# Conversión de la imagen a bits

def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)


#####################################################
# Función para modular la señal

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora_I = np.cos(2*np.pi*fc*t_periodo)
    portadora_Q = np.sin(2*np.pi*fc*t_periodo)
    

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    senal_Tx_I = np.zeros(t_simulacion.shape)
    senal_Tx_Q = np.zeros(t_simulacion.shape)
    #moduladora = np.zeros(t_simulacion.shape)  # (opcional) señal de bits

    # Índices de los bits b1, b2, b3 y b4.
    bi_1 = range(0, len(bits), 4)
    bi_2 = range(1, len(bits), 4)
    bi_3 = range(2, len(bits), 4)
    bi_4 = range(3, len(bits), 4)

    # 4. Asignar las formas de onda según la modulación QAM
    for i in range(N//4): # i corresponde a cada cuatro bits debido a QAM
        
        # Selección mediante if de la amplitud y fase de la portadora I
        if bits[bi_1[i]] == 0:
            if bits[bi_2[i]] == 0:
                senal_Tx_I[4*i*mpp : (4*i+1)*mpp] = portadora_I * -3
            else:
                senal_Tx_I[4*i*mpp : (4*i+1)*mpp] = portadora_I * -1
        else:
            if bits[bi_2[i]] == 0:
                senal_Tx_I[4*i*mpp : (4*i+1)*mpp] = portadora_I * 3
            else:
                senal_Tx_I[4*i*mpp : (4*i+1)*mpp] = portadora_I * 1
        
        # Selección mediante if de la amplitud y fase de la portadora Q
        if bits[bi_3[i]] == 0:
            if bits[bi_4[i]] == 0:
                senal_Tx_Q[(4*i)*mpp : (4*i+1)*mpp] = portadora_Q * 3
            else:
                senal_Tx_Q[(4*i)*mpp : (4*i+1)*mpp] = portadora_Q * 1
        else:
            if bits[bi_4[i]] == 0:
                senal_Tx_Q[(4*i)*mpp : (4*i+1)*mpp] = portadora_Q * -3
            else:
                senal_Tx_Q[(4*i)*mpp : (4*i+1)*mpp] = portadora_Q * -1
                
    senal_Tx = senal_Tx_I + senal_Tx_Q # Se suman las señales moduladas I y Q
        
    
    # 5. Calcular la potencia promedio de la señal modulada
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    
    
    return senal_Tx, P_senal_Tx, portadora_I, portadora_Q  



#####################################################
# Construcción de un canal con ruido AWGN

def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx



#####################################################
# Esquema de demodulación

def demodulador(senal_Rx, portadora_I, portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)
   

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora
    Es_I = np.sum(portadora_I * portadora_I)
    Es_Q = np.sum(portadora_Q * portadora_Q)

    # Demodulación
    for i in range(N//4):
        # Producto interno de dos funciones para señales portadoras
        producto_I = senal_Rx[4*i*mpp : (4*i+1)*mpp] * portadora_I
        producto_Q = senal_Rx[4*i*mpp : (4*i+1)*mpp] * portadora_Q
        Ep_I = np.sum(producto_I)
        Ep_Q = np.sum(producto_Q)
        senal_demodulada[i*mpp : (i+1)*mpp] = producto_I + producto_Q
        
        # Criterio de decisión por detección de energía
        # Si Ep_I es negativo, el bit b1 es 0
        if Ep_I < 0:
            # Por pruebas, si el valor absoluto del error es mayor
            # de 15, el bit b2 es cero
            if abs(Ep_I) > 15:
                bits_Rx[4*i] = 0
                bits_Rx[4*i + 1] = 0
            else:
                bits_Rx[4*i] = 0
                bits_Rx[4*i + 1] = 1
        else:
            if abs(Ep_I) > 15:
                bits_Rx[4*i] = 1
                bits_Rx[4*i + 1] = 0
            else:
                bits_Rx[4*i] = 1
                bits_Rx[4*i + 1] = 1
                
                
        
        # Si Ep_Q es positivo, el bit b3 es 0
        if Ep_Q > 0:
            # Por pruebas, si el valor absoluto del error es mayor
            # de 15, el bit b4 es cero
            if abs(Ep_Q) > 15:
                bits_Rx[4*i + 2] = 0
                bits_Rx[4*i + 3] = 0
            else:
                bits_Rx[4*i + 2] = 0
                bits_Rx[4*i + 3] = 1
        else:
            if abs(Ep_Q) > 15:
                bits_Rx[4*i + 2] = 1
                bits_Rx[4*i + 3] = 0
            else:
                bits_Rx[4*i + 2] = 1
                bits_Rx[4*i + 3] = 1

             
    return bits_Rx.astype(int), senal_demodulada



#####################################################
# Reconstrucción de la imagen

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)


#####################################################
# Simulación del sistema de comunicaciones con modulación QAM



import matplotlib.pyplot as plt
import time

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = -5   # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape

# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora_I, portadora_Q = modulador(bits_Tx, fc, mpp)

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora_I, portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')
Fig.tight_layout()

plt.imshow(imagen_Rx)





#####################################################
# Visualizar el cambio entre las señales

fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

# La onda cuadrada moduladora (bits de entrada)
# ax1.plot(moduladora[0:600], color='r', lw=2) 
# ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()



#####################################################
# Empleando el siguiente código dado por el profesor, se obtiene
# la dráfica de densidad espectral

from scipy import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()


# ---
# 
# ### Universidad de Costa Rica
# #### Facultad de Ingeniería
# ##### Escuela de Ingeniería Eléctrica
# 
# ---
