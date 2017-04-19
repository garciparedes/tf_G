# Algoritmos para Big Data

- Métodos de resumen (synopses):
  Consiste en realizar una compresión de los datos originales que ocupen un menor espacio y a partir de los cuales se pueda realizar cálculos aproximados con un menor coste computacional que con el conjunto completo.

  Existen distintas técnicas para ello, entre las que se encuentran:
  - **Sampling**: Seleccionar un subconjunto aleatorio a partir del global que será utilizado como muestra
  - **Histogram**:
  - **Waveless**:
  - **Sketches**:


- Algoritmos en Streaming:
  Son utilizados para mantener las estructuras de sketching conforme los datos van siendo obtenidos.

- Problemas:
  - Número de veces que ocurre un evento: Morris
  - Número de eventos distintos que ocurren en una secuencia: Flajolet–Martin
  - Evento más frecuente (heavy hitters)
  - Frecuencia de elementos más frecuentes
  - Elementos dentro de un rango
  - Pertenencia al conjunto

- Estructuras de Datos
  - LogLog Counter
  - Bloom Filter
  - Count-Min Sketch
  - Count-Mean-Min Sketch
  - Stream-Summary
