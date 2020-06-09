#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

struct global_data{
    double c_x_min;
    double c_x_max;
    double c_y_min;
    double c_y_max;

    double pixel_width;
    double pixel_height;

    int iteration_max = 200;
    int n_threads;
    int image_size;
    int image_buffer_size;
    int i_x_max;
    int i_y_max;

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
    int entra = 0;
};



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


struct thread_data {
    int begin;
    int end;
};
struct thread_data * thread_data_array;
struct global_data * global_data;

/////////////////////////////////////////
void allocate_image_buffer(){
    int rgb_size = 3;
    global_data->image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * global_data->image_buffer_size);

    for(int i = 0; i < global_data->image_buffer_size; i++){
        global_data->image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[]){
    if(argc < 7){
        printf("usage: ./mandelbrot_pth c_x_min c_x_max c_y_min c_y_max image_size n_threads\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_pth -2.5 1.5 -2.0 2.0 11500 4\n");
        printf("    Seahorse Valley:      ./mandelbrot_pth -0.8 -0.7 0.05 0.15 11500 4\n");
        printf("    Elephant Valley:      ./mandelbrot_pth 0.175 0.375 -0.1 0.1 11500 4\n");
        printf("    Triple Spiral Valley: ./mandelbrot_pth -0.188 -0.012 0.554 0.754 11500 4\n");
        exit(0);
    }
    else{
        sscanf(argv[1], "%lf", &global_data->c_x_min);
        sscanf(argv[2], "%lf", &global_data->c_x_max);
        sscanf(argv[3], "%lf", &global_data->c_y_min);
        sscanf(argv[4], "%lf", &global_data->c_y_max);
        sscanf(argv[5], "%d", &global_data->image_size);
        sscanf(argv[6], "%d", &global_data->n_threads);

        global_data->i_x_max           = global_data->image_size;
        global_data->i_y_max           = global_data->image_size;
        global_data->image_buffer_size = global_data->image_size * global_data->image_size;

        global_data->pixel_width       = (global_data->c_x_max - global_data->c_x_min) / global_data->i_x_max;
        global_data->pixel_height      = (global_data->c_y_max - global_data->c_y_min) / global_data->i_y_max;
    };
};
__managed__ int entra = 0;
__device__
void update_rgb_buffer(int iteration, int x, int y, struct global_data * global_data, struct thread_data * thread_data){
    int color;
    if(iteration == global_data->iteration_max){
        global_data->image_buffer[(global_data->i_y_max * y) + x][0] = global_data->colors[global_data->gradient_size][0];
        global_data->image_buffer[(global_data->i_y_max * y) + x][1] = global_data->colors[global_data->gradient_size][1];
        global_data->image_buffer[(global_data->i_y_max * y) + x][2] = global_data->colors[global_data->gradient_size][2];
    }
    else{
        color = iteration % global_data->gradient_size;

        global_data->image_buffer[(global_data->i_y_max * y) + x][0] = global_data->colors[color][0];
        global_data->image_buffer[(global_data->i_y_max * y) + x][1] = global_data->colors[color][1];
        global_data->image_buffer[(global_data->i_y_max * y) + x][2] = global_data->colors[color][2];
    };
};

void write_to_file(){
    FILE * file;
    char * filename               = "output_cuda.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
    global_data->i_x_max, global_data->i_y_max, max_color_component_value);

    for(int i = 0; i < global_data->image_buffer_size; i++){
        fwrite(global_data->image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};



__global__
void compute_mandelbrot_thread(struct global_data * global_data, struct thread_data * thread_data_array){

    global_data->entra++;
    int ind = (blockIdx.x) * blockDim.x + (threadIdx.x);
    int begin = thread_data_array[ind].begin;
    int end = thread_data_array[ind].end;
    //printf("END = %d\n", end);
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

    for(i_y = begin; i_y < end; i_y++){
        c_y = global_data->c_y_min + i_y * global_data->pixel_height;

        if(fabs(c_y) < global_data->pixel_height / 2){
            c_y = 0.0;
        };
        printf("i_x = %d\n", global_data->i_x_max);
        for(i_x = 0; i_x < global_data->i_x_max; i_x++){
            c_x         = global_data->c_x_min + i_x * global_data->pixel_width;

            z_x         = 0.0;
            z_y         = 0.0;

            z_x_squared = 0.0;
            z_y_squared = 0.0;

            for(iteration = 0;
                iteration < global_data->iteration_max && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++){
                z_y         = 2 * z_x * z_y + c_y;
                z_x         = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };
            update_rgb_buffer(iteration, i_x, i_y, global_data, thread_data_array);
            
        };
    };
    
};

int main(int argc, char *argv[]){
    

    ///////////////////////////////////////////

    
    cudaError_t batata1 = cudaMallocManaged(&global_data, sizeof(global_data));
    init(argc, argv);
    cudaError_t batata = cudaMallocManaged(&thread_data_array, global_data->n_threads*sizeof(struct thread_data));
    printf("batata = %d\n", batata1);
    
    allocate_image_buffer();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_start);

    int n_threads_block = 1024;
    int N_BLOCKS = (global_data->n_threads+n_threads_block-1) / n_threads_block;
    printf("N_THreads = %d\n", global_data->n_threads);
    printf("y_max = %d\n" ,global_data->i_y_max );
    printf("N_BLOCKS = %d\n", N_BLOCKS);

    for (int i = 0; i < global_data->n_threads; i++) {
        
        if (i == 0) {
            thread_data_array[i].begin = 0;
        } else {
            thread_data_array[i].begin = thread_data_array[i-1].end;
        }
        int gap = global_data->i_y_max / global_data->n_threads; 
        if (i == global_data->n_threads - 1) {
            thread_data_array[i].end = global_data->i_y_max;
        } else {
            thread_data_array[i].end = thread_data_array[i].begin + gap;
        }
    }
    compute_mandelbrot_thread<<<1, 1024>>>(global_data, thread_data_array); 

    cudaDeviceSynchronize();
    printf("entra = %d\n", global_data->entra);
    clock_gettime(CLOCK_MONOTONIC, &timer.t_end);

    printf("%f\n",
               (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0);
    /////////////////////////////////////////

    write_to_file();
    cudaFree(global_data);
    cudaFree(thread_data_array);

    return 0;
};