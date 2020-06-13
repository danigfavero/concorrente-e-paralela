all: mandelbrot_seq mandelbrot_pth mandelbrot_omp mandelbrot_cuda mandelbrot_openmpi

mandelbrot_seq: mandelbrot_seq.c
	gcc -Wall -o mandelbrot_seq mandelbrot_seq.c

mandelbrot_pth: mandelbrot_pth.c
	gcc -pthread -Wall -o mandelbrot_pth mandelbrot_pth.c

mandelbrot_omp: mandelbrot_omp.c
	gcc -fopenmp -Wall -o mandelbrot_omp mandelbrot_omp.c

mandelbrot_cuda: mandelbrot_cuda.cu
	nvcc -Wall -o mandelbrot_cuda mandelbrot_cuda.cu

mandelbrot_openmpi: mandelbrot_openmpi.c
	mpicc -Wall -o mandelbrot_openmpi mandelbrot_openmpi.c

clean:
	rm -rf *.o *~ all
