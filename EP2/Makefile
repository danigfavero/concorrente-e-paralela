all: mandelbrot_seq mandelbrot_pth mandelbrot_omp mandelbrot_cuda mandelbrot_openmpi mandelbrot_openmpi+omp mandelbrot_openmpi+cuda
VPATH=Cuda:MPI:MPI+Cuda:MPI+OMP:OMP:Pthreads:Sequential

mandelbrot_seq: mandelbrot_seq.c
	gcc -o Sequential/mandelbrot_seq Sequential/mandelbrot_seq.c

mandelbrot_pth: mandelbrot_pth.c
	gcc -pthread -Wall -o Pthreads/mandelbrot_pth Pthreads/mandelbrot_pth.c

mandelbrot_omp: mandelbrot_omp.c
	gcc -fopenmp -Wall -o OMP/mandelbrot_omp OMP/mandelbrot_omp.c

mandelbrot_cuda: mandelbrot_cuda.cu
	nvcc -Xcompiler -Wall -o Cuda/mandelbrot_cuda Cuda/mandelbrot_cuda.cu

mandelbrot_openmpi: mandelbrot_openmpi.c
	mpicc -o MPI/mandelbrot_openmpi MPI/mandelbrot_openmpi.c

mandelbrot_openmpi+omp: mandelbrot_openmpi+omp.c
	mpicc -fopenmp -o MPI+OMP/mandelbrot_openmpi+omp MPI+OMP/mandelbrot_openmpi+omp.c

mandelbrot_openmpi+cuda: mandelbrot_openmpi+cuda.cu
	nvcc -o MPI+Cuda/mandelbrot_openmpi+cuda -Xcompiler -lmpi -lstdc++ -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi MPI+Cuda/mandelbrot_openmpi+cuda.cu
clean:
	rm -rf *.o *~ all
