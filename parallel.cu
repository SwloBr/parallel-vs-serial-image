#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <jpeglib.h>
#include <chrono>
#include <cuda_runtime.h>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <random>

namespace fs = std::filesystem;

// Definir o diretório do dataset aqui
const std::string dataset_directory = "/home/swlo/Pictures/photos";
const int MAX_IMAGES_IN_MEMORY = 10;
const int TOTAL_IMAGES_TO_PROCESS = 10; // Defina quantas imagens deseja processar

std::mutex mtx;
std::condition_variable cv;
std::queue<std::string> image_queue;
bool done_reading = false;
int images_processed = 0;

// Kernel CUDA para converter uma imagem para tons de cinza
__global__ void convertToGrayscaleKernel(unsigned char* d_image, int rows, int cols, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) {
        int pixel_index = (idy * cols + idx) * channels;
        unsigned char blue = d_image[pixel_index];
        unsigned char green = d_image[pixel_index + 1];
        unsigned char red = d_image[pixel_index + 2];

        unsigned char gray = static_cast<unsigned char>(red * 0.298 + green * 0.587 + blue * 0.114);

        d_image[pixel_index] = gray;
        d_image[pixel_index + 1] = gray;
        d_image[pixel_index + 2] = gray;
    }
}

// Função para converter uma imagem para tons de cinza usando CUDA
void convertToGrayscaleCUDA(std::vector<std::vector<std::vector<unsigned char>>>& image) {
    int rows = image.size();
    int cols = image[0].size();
    int channels = 3;
    size_t image_size = rows * cols * channels * sizeof(unsigned char);

    // Alocar memória na GPU
    unsigned char* d_image;
    cudaMalloc(&d_image, image_size);

    // Copiar a imagem do host para a memória da GPU
    unsigned char* h_image = new unsigned char[rows * cols * channels];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = (i * cols + j) * channels;
            h_image[index] = image[i][j][0];
            h_image[index + 1] = image[i][j][1];
            h_image[index + 2] = image[i][j][2];
        }
    }
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);

    // Definir dimensões do grid e dos blocos
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Executar o kernel
    convertToGrayscaleKernel<<<gridDim, blockDim>>>(d_image, rows, cols, channels);
    cudaDeviceSynchronize();

    // Copiar a imagem processada de volta para o host
    cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = (i * cols + j) * channels;
            image[i][j][0] = h_image[index];
            image[i][j][1] = h_image[index + 1];
            image[i][j][2] = h_image[index + 2];
        }
    }

    // Liberar memória
    delete[] h_image;
    cudaFree(d_image);
}

// Função para ler uma imagem JPEG
bool readJPEG(const std::string& filename, std::vector<std::vector<std::vector<unsigned char>>>& image, int& rows, int& cols) {
    std::cout << "Reading JPEG file: " << filename << std::endl;
    FILE* infile = fopen(filename.c_str(), "rb");
    if (!infile) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, infile);
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);

    rows = cinfo.output_height;
    cols = cinfo.output_width;
    int channels = cinfo.output_components;

    if (channels != 3) {
        std::cerr << "Unsupported number of channels: " << channels << " in file: " << filename << std::endl;
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return false;
    }

    image.resize(rows, std::vector<std::vector<unsigned char>>(cols, std::vector<unsigned char>(3)));
    unsigned char* row_pointer = new unsigned char[cols * channels];
    while (cinfo.output_scanline < cinfo.output_height) {
        int row = cinfo.output_scanline;
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
        for (int col = 0; col < cols; ++col) {
            image[row][col][0] = row_pointer[col * channels];     // Blue
            image[row][col][1] = row_pointer[col * channels + 1]; // Green
            image[row][col][2] = row_pointer[col * channels + 2]; // Red
        }
    }

    delete[] row_pointer;
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(infile);
    return true;
}

// Função para salvar uma imagem JPEG
void saveJPEG(const std::string& filename, const std::vector<std::vector<std::vector<unsigned char>>>& image, int rows, int cols) {
    std::cout << "Saving JPEG file: " << filename << std::endl;
    FILE* outfile = fopen(filename.c_str(), "wb");
    if (!outfile) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = cols;
    cinfo.image_height = rows;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 85, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    unsigned char* row_pointer = new unsigned char[cols * 3];
    while (cinfo.next_scanline < cinfo.image_height) {
        int row = cinfo.next_scanline;
        for (int col = 0; col < cols; ++col) {
            row_pointer[col * 3] = image[row][col][0];     // Blue
            row_pointer[col * 3 + 1] = image[row][col][1]; // Green
            row_pointer[col * 3 + 2] = image[row][col][2]; // Red
        }
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    delete[] row_pointer;
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

void readerThread() {
    std::vector<std::string> all_images;
    for (const auto& entry : fs::directory_iterator(dataset_directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jpg") {
            all_images.push_back(entry.path().string());
        }
    }

    int images_to_process = 0;
    while (images_processed < TOTAL_IMAGES_TO_PROCESS) {
        const std::string& image_path = all_images[images_to_process % all_images.size()];
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [] { return image_queue.size() < MAX_IMAGES_IN_MEMORY; });
            image_queue.push(image_path);
            images_processed++;
            std::cout << "Added image " << images_processed << " to queue: " << image_path << std::endl;
            cv.notify_all();
        }
        images_to_process++;
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        done_reading = true;
    }
    cv.notify_all();
}

void workerThread(int thread_id, const std::string& output_directory) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 1000);

    while (true) {
        std::string image_path;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [] { return !image_queue.empty() || done_reading; });
            if (image_queue.empty() && done_reading) {
                break;
            }
            image_path = image_queue.front();
            image_queue.pop();
            std::cout << "Worker " << thread_id << " processing image: " << image_path << std::endl;
            cv.notify_all();
        }

        // Processar a imagem
        int rows, cols;
        std::vector<std::vector<std::vector<unsigned char>>> image;
        if (!readJPEG(image_path, image, rows, cols)) {
            continue;
        }

        if (rows < 500 || cols < 500) {
            std::cerr << "Image too small (must be at least 500x500 pixels): " << image_path << std::endl;
            continue;
        }

        // Converter para tons de cinza usando CUDA
        convertToGrayscaleCUDA(image);

        // Salvar a imagem convertida
        int random_id = dist(gen);
        std::string output_path = output_directory + "/" + std::to_string(random_id) + "_" + fs::path(image_path).filename().string();
        saveJPEG(output_path, image, rows, cols);
    }
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Collecting image paths..." << std::endl;

    // Criar diretório de saída
    const std::string output_directory = dataset_directory + "/out";
    fs::create_directory(output_directory);

    // Iniciar threads
    std::thread reader(readerThread);
    std::vector<std::thread> workers;
    for (int i = 0; i < MAX_IMAGES_IN_MEMORY; ++i) {
        workers.emplace_back(workerThread, i, output_directory);
    }

    // Aguardar conclusão
    reader.join();
    for (auto& worker : workers) {
        worker.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Total processing time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}
