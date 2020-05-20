#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

double c_x_min;
double c_x_max;
double c_y_min;
double c_y_max;

double pixel_width;
double pixel_height;

int iteration_max = 200;
int n_threads;

int image_size;
unsigned char **image_buffer;

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

struct thread_data {
    int begin;
    int end;
    void (*f)(int, int, int);

};

struct package {
    int i_x;
    int i_y;
    double z_x;
    double z_y;
    double c_x;
    double c_y;
    double z_x_squared;
    double z_y_squared;
};

struct thread_data *thread_data_array;
struct package *linearized_matrix;

/////////////////////////////////////////
void allocate_image_buffer(){
    int rgb_size = 3;
    image_buffer = (unsigned char **) malloc(sizeof(unsigned char *) * image_buffer_size);

    for(int i = 0; i < image_buffer_size; i++){
        image_buffer[i] = (unsigned char *) malloc(sizeof(unsigned char) * rgb_size);
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
};

void update_rgb_buffer(int iteration, int x, int y){
    int color;

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
            i_x_max, i_y_max, max_color_component_value);

    for(int i = 0; i < image_buffer_size; i++){
        fwrite(image_buffer[i], 1 , 3, file);
    };

    fclose(file);
};

void *compute_mandelbrot_thread(void *args){

    struct thread_data *arg_struct = (struct thread_data*) args; 

    double z_x;
    double z_y;
    double z_x_squared;
    double z_y_squared;
    double escape_radius_squared = 4;

    int iteration;

    double c_x;
    double c_y;
    for(int i = arg_struct->begin; i < arg_struct->end; i++){
        
        c_x = linearized_matrix[i].c_x;
        c_y = linearized_matrix[i].c_y;
        z_x = linearized_matrix[i].z_x;
        z_y = linearized_matrix[i].z_y;
        z_x_squared = linearized_matrix[i].z_x_squared;
        z_y_squared = linearized_matrix[i].z_y_squared;
        
        for(iteration = 0;
                iteration < iteration_max && \
                ((z_x_squared + z_y_squared) < escape_radius_squared);
                iteration++){
                z_y         = 2 * z_x * z_y + c_y;
                z_x         = z_x_squared - z_y_squared + c_x;

                z_x_squared = z_x * z_x;
                z_y_squared = z_y * z_y;
            };

            arg_struct->f(iteration, linearized_matrix[i].i_x, linearized_matrix[i].i_y);
    }

    pthread_exit(NULL);
     
};

int main(int argc, char *argv[]){
    init(argc, argv);

    allocate_image_buffer();

    linearized_matrix = malloc(i_x_max * i_y_max * sizeof(struct package));
    int counter = 0;
    for(int i_y = 0; i_y < i_y_max; i_y++){
        for(int i_x = 0; i_x < i_x_max; i_x++){

            linearized_matrix[counter].i_x =                          i_x;
            linearized_matrix[counter].i_y =                          i_y;
            linearized_matrix[counter].c_x =  c_x_min + i_x * pixel_width;
            linearized_matrix[counter].c_y = c_y_min + i_y * pixel_height;
            if(fabs(linearized_matrix[counter].c_y < pixel_height / 2)){
                linearized_matrix[counter].c_y = 0.0;
            }    
            linearized_matrix[counter].z_x =                          0.0;
            linearized_matrix[counter].z_y =                          0.0;
            linearized_matrix[counter].z_x_squared =                  0.0;
            linearized_matrix[counter].z_y_squared =                  0.0;
            counter++;
        };
    };

    pthread_t tids[n_threads];
    thread_data_array = malloc(n_threads * sizeof(struct thread_data));

    clock_t start = clock();
    for (int i = 0; i < n_threads; i++) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);

        if (i == 0) {
            thread_data_array[i].begin = 0;
        } else {
            thread_data_array[i].begin = thread_data_array[i-1].end;
        }

        int gap = (i_x_max*i_y_max) / n_threads; 
        if (i == n_threads - 1) {
            thread_data_array[i].end = i_x_max*i_y_max - 1;
        } else {
            thread_data_array[i].end = thread_data_array[i].begin + gap;
        }

        thread_data_array[i].f = update_rgb_buffer;

        pthread_create(&tids[i], &attr, 
                           compute_mandelbrot_thread, &thread_data_array[i]);
    };

    
    for (int i = 0; i < n_threads; i++) {
        pthread_join(tids[i], NULL);
    }

    clock_t end = clock();

    long double time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%Lf\n", time);
    /////////////////////////////////////////

    write_to_file();

    return 0;
};
