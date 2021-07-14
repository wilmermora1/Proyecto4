## Universidad de Costa Rica
### Escuela de Ingeniería Eléctrica
#### IE0405 - Modelos Probabilísticos de Señales y Sistemas


* Estudiante: **Wilmer Daniel Mora Pereira**
* Carné: **B95188**
* Grupo: **2**

---
# P4 - *Modulación digital IQ*

En este archivo se documentan los resultados obtenidos en la elaboración del problema.
El proyecto se encuentra en un trabajo elaborado por Jeaustin Sirias Chacón sobre
modulación digital mediante BPSK, pero el problema consiste en utilizar modulación
QAM en lugar de la BPSK. Se hará una simulación en la que se envía una imagen
montada en una señal modulada, se le agrega ruido y se realiza una demodulación de la
señal y a partir de esta, se hace la reconstrucción de la imagen.

### Resultados obtenidos

En el documento P4.py corresponde a una extracción de las funciones del documento
original del proyecto, junto con las modificaciones realizadas a las funciones para
poder realizar la modulación 16-QAM.

La imagen obtenida de la simulación se muestra a continuación:

![Imágen no se encuentra](https://github.com/wilmermora1/Proyecto4/blob/main/comparacion_imagenes.png)

Se observa que hay alguna información perdida debido al ruido que se agregó a la señal
modulada en la simulación. Además, al correr el programa se obtiene que hubo 29 833 errores,
generando un BER de 0.0705. Esta diferencia en el ruido respecto a la modulación BPSK se debe a
que es una modulación que envía información más rápido pues cada símbolo contiene 4 bits.

Así mismo, se muestra a continuación una imágen donde se observan la señal modulada en verde, 
la señal modulada más el ruido en azul y la señal demodulada. Como se observa, el ruido genera una 
distorción considerable, pero por el resultado, aún se puede extraer información suficiente
para tener una imagen observable.

![Imágen no se encuentra](https://github.com/wilmermora1/Proyecto4/blob/main/graficas.png)
