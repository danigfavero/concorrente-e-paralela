#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

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

    int n_threads;
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
        printf("usage: ./mandelbrot_cuda c_x_min c_x_max c_y_min c_y_max image_size n_threads(total)\n");
        printf("examples with image_size = 11500:\n");
        printf("    Full Picture:         ./mandelbrot_cuda -2.5 1.5 -2.0 2.0 11500 1024\n");
        printf("    Seahorse Valley:      ./mandelbrot_cuda -0.8 -0.7 0.05 0.15 11500 1024\n");
        printf("    Elephant Valley:      ./mandelbrot_cuda 0.175 0.375 -0.1 0.1 11500 1024\n");
        printf("    Triple Spiral Valley: ./mandelbrot_cuda -0.188 -0.012 0.554 0.754 11500 1024\n");
        exit(0);
    }
    else{
        sscanf(argv[1], "%lf", &global_data->c_x_min);
        sscanf(argv[2], "%lf", &global_data->c_x_max);
        sscanf(argv[3], "%lf", &global_data->c_y_min);
        sscanf(argv[4], "%lf", &global_data->c_y_max);
        sscanf(argv[5], "%d", &image_size);
        sscanf(argv[6], "%d", &global_data->n_threads);

        global_data->i_x_max           = image_size;
        global_data->i_y_max           = image_size;
        image_buffer_size = image_size * image_size;

        global_data->pixel_width       = (global_data->c_x_max - global_data->c_x_min) / global_data->i_x_max;
        global_data->pixel_height      = (global_data->c_y_max - global_data->c_y_min) / global_data->i_y_max;
    };

};

void update_rgb_buffer(int iteration, int x, int y){
    int color;

    int i_y_max = global_data->i_y_max;
    int iteration_max = 200;

    if(iteration == iteration_max){
        image_buffer[(i_y_max * y) + x][0] = colors[gradient_size][0];
        image_buffer[(i_y_max * y) + x][1] = colors[gradient_size][1];
        image_buffer[(i_y_max * y) + x][2] = colors[gradient_size][2];
    }
    else{
        color = iteration % gradient_size;

        image_buffer[(i_y_max * y) + x][0] = colors[color][0];
        image_buffer[(i_y_max * y) + x][1] = colors[color][1];
        image_buffer[(i_y_max * y) + x][2] = colors[color][2];
    };
};

void write_to_file(){
    FILE * file;
    char * filename               = "output_pth.ppm";
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

__global__
void compute_mandelbrot_thread(struct data * gpu_data, int * out){
    
    int i_y_max = gpu_data->i_y_max;
    int n_threads = gpu_data->n_threads;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    int chunkSize = i_y_max / n_threads;
    int leftover = i_y_max % n_threads;
    int begin = ind * chunkSize + ((ind != 0) ? leftover : 0);
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
            out[ (i_x)*i_y_max + i_y       ] = iteration;
        };
    };

    
};


int main(int argc, char *argv[]){
    int * d_out;
    int * out;
    struct data * gpu_data;

    init(argc, argv);
    allocate_image_buffer();

    cudaSetDevice(0);
    cudaMalloc((void **)&d_out, global_data->i_x_max * global_data->i_y_max * sizeof(int));
    cudaMalloc((void **)&gpu_data, sizeof(struct data));
    cudaMemcpy(gpu_data, global_data, sizeof(struct data), cudaMemcpyHostToDevice);
    
    /*Lançamento da gpu*/
    int n_threads = global_data->n_threads;
    int n_threads_block = min(n_threads,1024);
    int N_BLOCKS = (n_threads+n_threads_block-1) / n_threads_block;
    compute_mandelbrot_thread<<<N_BLOCKS, n_threads_block>>>(gpu_data, d_out);
    cudaDeviceSynchronize();
    
    /*Copia dados para cpu*/
    out = (int *) malloc(global_data->i_x_max * global_data->i_y_max * sizeof(int));

    cudaMemcpy(out, d_out, global_data->i_x_max * global_data->i_y_max * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i<global_data->i_x_max * global_data->i_y_max; i++){
        int linha = i/global_data->i_y_max;
        int coluna = i%global_data->i_y_max;
        int iteration = out[i];
        update_rgb_buffer(iteration, linha, coluna);
    }

    write_to_file();

    return 0;
};