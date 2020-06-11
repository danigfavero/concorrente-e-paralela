#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

/*Variaveis globais HOST*/
double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;
double pixel_width;
double pixel_height;
int iteration_max = 200;
int n_threads;
int image_size;
unsigned char ** image_buffer;
unsigned char * image_buffer_linear;
int i_x_max;
int i_y_max;
int image_buffer_size;
int gradient_size = 16;
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
//////////////////////////////////////////

/*Variaveis globais device*/
__device__ double d_c_x_min;
__device__ double d_c_x_max;
__device__ double d_c_y_min;
__device__ double d_c_y_max;
__device__ double d_pixel_width;
__device__ double d_pixel_height;
__device__ int d_iteration_max = 200;
__device__ int d_n_threads;
__device__ int d_image_size;
__device__ int d_i_x_max;
__device__ int d_i_y_max;
__device__ int d_image_buffer_size;
__device__ int d_gradient_size = 16;
__device__ int d_colors[17][3] = {
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
//////////////////////////////////////////

/*Timer info*/
struct timer_info {
    clock_t c_start;
    clock_t c_end;
    struct timespec t_start;
    struct timespec t_end;
    struct timeval v_start;
    struct timeval v_end;
};

struct timer_info timer;
//////////////////////////////////////////

/*Thread info*/

struct thread_data {
    int begin;
    int end;
};

struct thread_data **thread_data_array;
/////////////////////////////////////////

void allocate_image_buffer(){
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);
    image_buffer_linear = (unsigned char *) malloc(image_buffer_size * 3 * sizeof(unsigned char));
    for(int i = 0; i < image_buffer_size; i++){
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
    };
};

void init(int argc, char *argv[]){
    if(argc < 7){
        printf("usage: ./mandelbrot_cuda c_x_min c_x_max c_y_min c_y_max image_size n_threads(total)\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_cuda -2.5 1.5 -2.0 2.0 11500 1024\n");
        printf("    Seahorse Valley:      ./mandelbrot_cuda -0.8 -0.7 0.05 0.15 11500 1024\n");
        printf("    Elephant Valley:      ./mandelbrot_cuda 0.175 0.375 -0.1 0.1 11500 1024\n");
        printf("    Triple Spiral Valley: ./mandelbrot_cuda -0.188 -0.012 0.554 0.754 11500 1024\n");
        exit(0);
    }
    else{
        sscanf(argv[1], "%lf", &c_x_min);
        sscanf(argv[2], "%lf", &c_x_max);
        sscanf(argv[3], "%lf", &c_y_min);
        sscanf(argv[4], "%lf", &c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &n_threads);

        i_x_max           = image_size;
        i_y_max           = image_size;
        image_buffer_size = image_size * image_size;

        pixel_width       = (c_x_max - c_x_min) / i_x_max;
        pixel_height      = (c_y_max - c_y_min) / i_y_max;

    };

    cudaMemcpy(&d_c_x_min, &c_x_min, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_c_x_max, &c_x_max, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_c_y_min, &c_y_min, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_pixel_width, &pixel_width, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_pixel_height, &pixel_height, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_n_threads, &n_threads, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_image_size, &image_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_i_x_max, &i_x_max, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_image_size, &image_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_image_buffer_size, &image_buffer_size, sizeof(int), cudaMemcpyHostToDevice);
};

__device__
void update_rgb_buffer(unsigned char * d_image_buffer, int iteration, int x, int y){
    int color;

    if(iteration == 1){
        atomicExch((int*)&d_image_buffer[0], 1);
        //d_image_buffer[((d_i_y_max * y) + x)*3] = d_colors[d_gradient_size ][0];
        //atomicExch(&d_image_buffer[((d_i_y_max * y) + x)*3 + 1], d_colors[gradient_size][1]):
        //atomicExch(d_image_buffer[((d_i_y_max * y) + x)*3 + 2], d_colors[gradient_size][2]):
    }
    else{
        color = iteration % 1;

        //atomicExch(d_image_buffer[((d_i_y_max * y) + x)*3 ], d_colors[color][0]);
        //atomicExch(d_image_buffer[((d_i_y_max * y) + x)*3 + 1], d_colors[color][1]);
        //atomicExch(d_image_buffer[((d_i_y_max * y) + x)*3 + 2], d_colors[color][2]);
    };
};

void write_to_file(){
    FILE * file;
    char * filename               = "output_pth.ppm";
    char * comment                = "# ";

    int max_color_component_value = 255;

    file = fopen(filename,"wb");

    fprintf(file, "P6\n %s\n %d\n %d\n %d\n", comment,
            i_x_max, i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size; i++){
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};
__managed__ int entra = 0;

__global__
void compute_mandelbrot_thread(struct thread_data ** d_thread_data_array, unsigned char * d_image_buffer){
    
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    int begin = d_thread_data_array[ind]->begin;
    int end = d_thread_data_array[ind]->end;
    printf("ind = %d\n",begin);
    

    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;
    //d_thread_data_array[ind] = NULL;
    int iteration;
    int i_x;
    int i_y;

    double c_x;
    double c_y;

    for(i_y = begin; i_y < end; i_y++){
        c_y = 50 + i_y * 50;

        if(fabs(c_y) < 50 / 2){
            c_y = 0.0;
        };

        for(i_x = 0; i_x < 50; i_x++){
            c_x         = 50 + i_x * 50;

            z_x         = 0.0;
            z_y         = 0.0;

            z_x_squared = 0.0;
            z_y_squared = 0.0;

            for(iteration = 0;
                iteration < 50 && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++){
                z_y         = 2 * z_x * z_y + c_y;
                z_x         = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };
            //update_rgb_buffer(d_image_buffer, iteration,i_x,i_y);
        };
    };
    atomicAdd(&entra, 1);
};

void linear_to_2d(){
    for(int i = 0; i<3*image_buffer_size; i++){
        int linha = i/3;
        int coluna = i%3;
        image_buffer[linha][coluna] = image_buffer_linear[i];
    }

}
int main(int argc, char *argv[]){
    init(argc, argv);
    
    unsigned char * d_image_buffer;
    
    allocate_image_buffer();
    cudaMalloc( (void**) &d_image_buffer, image_buffer_size * 3 * sizeof(unsigned char));

    /*Informacoes das threads*/
    struct thread_data ** d_thread_data_array;
    thread_data_array = (struct thread_data **) malloc(n_threads * sizeof(struct thread_data *));
    int n_threads_block = 1024;
    int N_BLOCKS = (n_threads+n_threads_block-1) / n_threads_block;

    printf("n = %d\n", n_threads);
    for (int i = 0; i < n_threads; i++) {
        thread_data_array[i] = (thread_data *) malloc(sizeof(struct thread_data));
        if (i == 0) {
            thread_data_array[i]->begin = 0;
        } else {
            thread_data_array[i]->begin = thread_data_array[i-1]->end;
        }

        int gap = i_y_max / n_threads; 
        if (i == n_threads - 1) {
            thread_data_array[i]->end = i_y_max;
        } else {
            thread_data_array[i]->end = thread_data_array[i]->begin + gap;
        }

    }
    
    printf("end = %d\n", thread_data_array[0]->end);

    /*Passa essas informacoes para o device */
    cudaMalloc(&d_thread_data_array, n_threads * sizeof(struct thread_data *));
    int err = cudaMemcpy(&d_thread_data_array, &thread_data_array, n_threads * sizeof(struct thread_data *), cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_MONOTONIC, &timer.t_start);
    compute_mandelbrot_thread<<<N_BLOCKS, min(n_threads,1024)>>>(d_thread_data_array, d_image_buffer);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &timer.t_end);

    printf("%f\n",
               (double) (timer.t_end.tv_sec - timer.t_start.tv_sec) +
               (double) (timer.t_end.tv_nsec - timer.t_start.tv_nsec) / 1000000000.0);
    /////////////////////////////////////////

    /*Recupera buffer*/
    cudaMemcpy(&image_buffer_linear, &d_image_buffer, 3*image_buffer_size*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    printf("err = %d\n", err);
    printf("entraaaaaaaaaa =%d\n", entra);
    /*Deslineariza a matriz*/
    linear_to_2d();

    write_to_file();

    return 0;
};