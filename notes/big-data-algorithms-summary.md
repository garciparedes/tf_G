El big data puede resumirse como el procesamiento de datos cuyo tamaño es lo suficientemente grande como para que no quepa en la memoria RAM de uno ordenador. Para hacer frente a dicha problemática, se utilizan estimadores estadísticos que resumen la información que suministra el gran conjunto de datos.

Para generar y mantener estos indicadores se utilizan estructuras de datos que se apoyan en gran medida en propiedades estadísticas. Existen diversas alternativas como Sampling, Histograms, Wavelets o Sketches. A partir de ellas se pretende obtener medidas como la existencia de un suceso, el suceso más común etc. Estas estructuras de datos están restringidas en espacio exigiendo que, este sea de orden sublinear al del conjunto global.

En los últimos años ha tenido un gran avance la investigación sobre algoritmos en streaming. Estos son utilizados para mantener dichas estructuras de datos, en especial, destaca su uso con Sketches, por su simplicidad y eficiencia cuando son utilizados en conjunto.

Se ha trabajado en técnicas de:
  - conteo de sucesos[Morris],
  - número de sucesos distintos[Flajolet, Martin],
  - obtención de momentos de frecuencias[Alon, Matias, Szegedy]

La combinación de estas técnicas junto con otras propiedades estadísticas permiten mantener Sketches (estructuras de datos que almacenan resúmenes del conjunto de datos global). Los Sketches se caracterizan por ser actualizados a través de un mismo tipo de operación que se aplica a todos los datos que llegan al Streaming.

Existe un gran número de alternativas según la información que se desee mantener. Entre ellas destacan:
  - Count-Min Sketch
  - Count Sketch
  - AMS Sketch
  - HyperLogLog
  - MinHash
  - etc.

En cuanto a búsquedas dentro del conjunto de datos, se utilizan métodos relacionados con encontrar el sub-conjunto de vecinos más cercanos a un determinado dato. Para ello, se utilizan funciones de reducción de dimensionalidad, donde tiene especial importancia la transformada de Johnson-Lindenstrauss.

Otro de los temas importantes son los métodos aproximados para el cálculo de operaciones algebráicas de matrices de gran tamaño, utilizados en campos donde se puede admitir un determinado margen de error.

También se ha investigado sobre resolución de sistemas de optimización convexa (generalización de programación lineal) mediante estrategias basadas en el gradiente descendente.

Los algoritmos de big data aplicados a grafos son una rama que todavía no está muy desarrollada debido a su mayor grado de complejidad. A pesar de ello, existen distintos trabajos sobre algoritmos para encontrar el árbol recubridor mínimo.
