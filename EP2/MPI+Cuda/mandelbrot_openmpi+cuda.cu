#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define  MASTER		0

/*Dados HOST*/
struct data{
    double c_x_min;
    double c_x_max;
    double c_y_min;
    double c_y_max;

    double pixel_width;
    double pixel_height;

    int i_x_max;
    int i_y_max;

    int n_blocks;
    int n_threads;

    int begin;
    int chunk;
};

struct data * global_data;

int colors[17][3] = {
    {66, 30, 15},
    {25, 7, 26},
    {9, 1, 47},
    {4, 4, 73},
    {0, 7, 100},
    {12, 44, 138},
    {24, 82, 177},
    {57, 125, 209},
    {134, 181, 229},
    {211, 236, 248},
    {241, 233, 191},
    {248, 201, 95},
    {255, 170, 0},
    {204, 128, 0},
    {153, 87, 0},
    {106, 52, 3},
    {16, 16, 16},
};

int gradient_size = 16;
unsigned char **image_buffer;
int image_size;
int image_buffer_size;

int * buffer;
//////////////////////////////////////////
struct timer_info {
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info timer;


/////////////////////////////////////////
void allocate_image_buffer(){
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);
    for(int i = 0; i < image_buffer_size; i++){
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[]){

    global_data = (struct data * )malloc(sizeof(struct data));

    if(argc < 7){
        printf("usage: mpirun --host localhost:num_rocesses c_x_min c_x_max c_y_min c_y_max image_size blocks threads\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         mpirun --host localhost:2 -2.5 1.5 -2.0 2.0 11500 4 256\n");
        printf("    Seahorse Valley:      mpirun --host localhost:2 -0.8 -0.7 0.05 0.15 11500 4 256\n");
        printf("    Elephant Valley:      mpirun --host localhost:2 0.175 0.375 -0.1 0.1 11500 4 256\n");
        printf("    Triple Spiral Valley: mpirun --host localhost:2 -0.188 -0.012 0.554 0.754 11500 4 256\n");
        exit(0);
    }
    else{
        sscanf(argv[1], "%lf", &global_data->c_x_min);
        sscanf(argv[2], "%lf", &global_data->c_x_max);
        sscanf(argv[3], "%lf", &global_data->c_y_min);
        sscanf(argv[4], "%lf", &global_data->c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &global_data->n_blocks);
        sscanf(argv[7], "%d", &global_data->n_threads);

        global_data->i_x_max           = image_size;
        global_data->i_y_max           = image_size;
        image_buffer_size = image_size * image_size;

        global_data->pixel_width       = (global_data->c_x_max - global_data->c_x_min) / global_data->i_x_max;
        global_data->pixel_height      = (global_data->c_y_max - global_data->c_y_min) / global_data->i_y_max;
    };

    buffer = (int *) malloc(global_data->i_x_max * global_data->i_y_max * sizeof(int));
};

void update_rgb_buffer(int iteration, int x, int y){
    int color;
    int iteration_max = 200;
    if(iteration == iteration_max){
        image_buffer[(global_data->i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(global_data->i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(global_data->i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else{
        color = iteration % gradient_size;

        image_buffer[(global_data->i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(global_data->i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(global_data->i_y_max * y) + x][2] = colors[color][2];
    };
};
__global__
void compute_mandelbrot_thread(struct data * gpu_data, int * out){
    
    int n_threads = gpu_data->n_threads * gpu_data->n_blocks; //Numero de threads total
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int chunkSize = gpu_data->chunk / n_threads;
    int leftover = gpu_data->chunk % n_threads;
    
    int begin = gpu_data->begin + ind * chunkSize + ((ind != 0) ? leftover : 0);
    int end = begin + chunkSize + (ind == 0 ? leftover : 0);
    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;
    int iteration;
    int i_x;
    int i_y;

    double c_x;
    double c_y;

    double c_y_min = gpu_data->c_y_min;
    double c_x_min = gpu_data->c_x_min;
    int i_x_max = gpu_data->i_x_max;
    int iteration_max = 200;
    double pixel_height = gpu_data->pixel_height;
    double pixel_width = gpu_data->pixel_width;

    for(i_y = begin; i_y < end; i_y++){
        c_y = c_y_min + i_y * pixel_height;

        if(fabs(c_y) < pixel_height / 2){
            c_y = 0.0;
        };

        for(i_x = 0; i_x < i_x_max; i_x++){
            c_x         = c_x_min + i_x * pixel_width;

            z_x         = 0.0;
            z_y         = 0.0;

            z_x_squared = 0.0;
            z_y_squared = 0.0;

            for(iteration = 0;
                iteration < iteration_max && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++){
                z_y         = 2 * z_x * z_y + c_y;
                z_x         = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };
            out[ (i_y)*i_x_max + i_x       ] = iteration;
        };
    };

};



void run_gpu(struct data * global_data, struct data * gpu_data, int * out, int * d_out)
{
    
    /*Lançamento da gpu*/
    int n_threads = global_data->n_threads;
    int n_blocks = global_data->n_blocks;
    compute_mandelbrot_thread<<<n_blocks, n_threads>>>(gpu_data, d_out);

    cudaDeviceSynchronize();
    cudaMemcpy(out, d_out, global_data->i_x_max * global_data->i_y_max * sizeof(int), cudaMemcpyDeviceToHost);


};

void write_to_file(){
    FILE * file;
    char * filename               = "Output/openmpi+cuda.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            global_data->i_x_max, global_data->i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size; i++){
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};

int main(int argc, char *argv[]){
    int numtasks, taskid;
    MPI_Init(&argc, &argv);
    init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    //printf ("MPI task %d has started...  \n", taskid);
    int chunksize = (global_data->i_y_max / numtasks);
    int leftover = (global_data->i_y_max % numtasks);
    int offset;
    MPI_Status status;
    
    allocate_image_buffer();

    if(taskid == MASTER){
        
        clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
        offset = chunksize + leftover;
        for(int dest = 1; dest<numtasks; dest++){
            MPI_Send(&offset, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            offset += chunksize;
        }

        /* Master does its part of the work */
        offset = 0;
        
        struct data * gpu_data = NULL;
        int * d_out = NULL;
        cudaMalloc((void **)&d_out, global_data->i_x_max * global_data->i_y_max * sizeof(int));
        cudaMalloc((void **)&gpu_data, sizeof(struct data));
        global_data->begin = offset;
        global_data->chunk = chunksize+leftover;
        cudaMemcpy(gpu_data, global_data, sizeof(struct data), cudaMemcpyHostToDevice);


        run_gpu(global_data, gpu_data, buffer, d_out);

        /* Wait to receive results from each task */
        
        offset = (chunksize + leftover)*global_data->i_y_max;
        for (int i=1; i<numtasks; i++) {
            int source = i;
            MPI_Recv(&buffer[offset], chunksize*global_data->i_x_max, MPI_INT, source, 1,
                     MPI_COMM_WORLD, &status);

            offset += chunksize*global_data->i_x_max;

        }

        for(int i_x = 0; i_x < global_data->i_x_max; i_x++){
            for(int i_y = 0; i_y < global_data->i_y_max; i_y++){
                update_rgb_buffer(buffer[i_y*global_data->i_x_max + i_x], i_x,i_y );
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &timer.t_end);
        printf("%f\n",
               (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0);
        
               write_to_file();
    }
    else{
        int source = MASTER;
        MPI_Recv(&offset, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);

        struct data * gpu_data = NULL;
        int * d_out = NULL;
        cudaMalloc((void **)&d_out, global_data->i_x_max * global_data->i_y_max * sizeof(int));
        cudaMalloc((void **)&gpu_data, sizeof(struct data));
        global_data->begin = offset;
        global_data->chunk = chunksize;
        cudaMemcpy(gpu_data, global_data, sizeof(struct data), cudaMemcpyHostToDevice);

        run_gpu(global_data, gpu_data, buffer, d_out);
        
        MPI_Send(&buffer[offset*global_data->i_x_max], global_data->i_x_max*chunksize, MPI_INT, MASTER, 1, MPI_COMM_WORLD);

    }
    MPI_Finalize();
    
    /////////////////////////////////////////

    return 0;
};
