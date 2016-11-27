#include "header.h"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

extern "C"
void multi_streams(std::string cur_dir);

__global__ void multi_streams_kernel(unsigned char* input, unsigned char* output, int stream_idx, unsigned int pixels_per_stream ,unsigned int pixel_cnt) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned idx = stream_idx * pixels_per_stream + tid;

	if( tid < pixel_cnt ){
		unsigned int color_idx = idx * BYTES_PER_PIXEL;
		unsigned int gray_idx = idx;

		unsigned char blue	= input[color_idx];
		unsigned char green	= input[color_idx + 1];
		unsigned char red	= input[color_idx + 2];
		float gray = red * 0.3f + green * 0.59f + blue * 0.11f;
		output[gray_idx] = static_cast<unsigned char>(gray);
	}
}

// get free memory space
unsigned long int get_free_mem(){
	std::string token;
	std::ifstream file("/proc/meminfo");
	while(file >> token){
		if(token == "MemFree:"){
			unsigned long int free_mem;
			if(file >> free_mem)
				return free_mem;
		}
	}
	return 0;
}

void gray_processing_CPU(unsigned char* color_pixels, unsigned char* gray_pixels, unsigned long int pixel_cnt){
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	unsigned char blue;
	unsigned char green;
	unsigned char red;
	float gray;
	std::cout<<"CPU processing......"<<std::endl;
	for( unsigned long int i=0; i<pixel_cnt; i++ ){
		blue	= color_pixels[i*3];
		green	= color_pixels[i*3+1];
		red	= color_pixels[i*3+2];
		gray	= red * 0.3f + green * 0.59f + blue * 0.11f;
		gray_pixels[i] = static_cast<unsigned char>(gray);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"CPU time: "<<milliseconds<< " ms"<<std::endl;
}

void gray_processing_multi_streams(unsigned char* color_pixels, unsigned char* gray_pixels, unsigned long int pixel_cnt){
	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	unsigned char *d_pixels_in, *d_pixels_out;
	SAFE_CALL(cudaMalloc(&d_pixels_in, pixel_cnt * BYTES_PER_PIXEL * sizeof( unsigned char )), "Malloc colored device memory failed!");
	SAFE_CALL(cudaMalloc(&d_pixels_out, pixel_cnt * sizeof(unsigned char)), "Malloc grayed device memory failed!");

	// get the number of pixesl each stream
	// unsigned pixels_per_stream = (pixel_cnt % STREAMS_CNT == 0 )? ( pixel_cnt/STREAMS_CNT ) : ( pixel_cnt/STREAMS_CNT + 1 );
	unsigned pixels_per_stream = (pixel_cnt + STREAMS_CNT - 1)/STREAMS_CNT;
	dim3 block(BLOCK_SIZE);
	dim3 grid((pixels_per_stream + BLOCK_SIZE -1 )/BLOCK_SIZE);

	std::cout<<"GPU processing with multi streams......"<<std::endl;

	// crate streams for current device
	cudaStream_t* streams = (cudaStream_t*) malloc( STREAMS_CNT * sizeof( cudaStream_t ) );
	for(int i=0; i<STREAMS_CNT; i++)
		SAFE_CALL(cudaStreamCreate(&streams[i]), "Create stream failed!");

	// pixel count in current stream
	// the number of pixels in last stream normally should be different with previous streams
	unsigned int pixel_in_cur_stream = 0;

	// start the stream execution for current device
	for( int i=0; i<STREAMS_CNT; i++ ){
		// this is the boundary check for the pixel number in last stream
		// normally, it coule not be the same number of pixels in each stream
		if ( i == STREAMS_CNT -1  )
			pixel_in_cur_stream = pixel_cnt - pixels_per_stream * (STREAMS_CNT - 1);
		else
			pixel_in_cur_stream = pixels_per_stream;

		// copy data from host to device
		SAFE_CALL(cudaMemcpyAsync(&d_pixels_in[i * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof( unsigned char )],
								&color_pixels[i * pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof( unsigned char )],
								pixel_in_cur_stream * BYTES_PER_PIXEL * sizeof(unsigned char),
								cudaMemcpyHostToDevice,
								streams[i]),
								"Device memory asynchronized copy failed!");

		// kernel launch
		multi_streams_kernel<<< grid, block, 0, streams[i] >>>(d_pixels_in, d_pixels_out, i, pixel_in_cur_stream, pixel_cnt);

		// copy data back from device to host
		SAFE_CALL(cudaMemcpyAsync(&gray_pixels[i * pixel_in_cur_stream ],
								&d_pixels_out[i * pixel_in_cur_stream ],
								pixel_in_cur_stream * sizeof(unsigned char),
								cudaMemcpyDeviceToHost,
								streams[i]),
								"Host memory asynchronized copy failed!");
	}

	// synchronize
	//cudaDeviceSynchronize();

	cudaEventRecord(stop);
	// cudaEventSynchronize(): wait until the completion of all device work preceding the most recent call to cudaEventRecored()
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout<<"GPU time: "<<milliseconds<< " ms"<<std::endl;

	// destroy streams
	for( int i=0; i<STREAMS_CNT; i++ ){
		cudaStreamDestroy(streams[i]);
	}
	// release the memory in device
	SAFE_CALL(cudaFree(d_pixels_in),"Free device color memory failed!");
	SAFE_CALL(cudaFree(d_pixels_out), "Free device gray memory failed!");
}

// write the images
void write_images(unsigned char* gray_pixels,
				std::string tar_dir,
				std::vector<std::string> name,
				std::vector<unsigned int> size,
				std::vector<int> row,
				std::vector<int> col,
				unsigned int img_cnt){
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	unsigned int img_ent_start_index = 0;
	std::string ent_str;

	std::cout<<"Writing "<< img_cnt <<" images to hard drives......"<<std::endl<<std::endl;
	// write gray pixels to images
	for(unsigned int i=0; i<img_cnt; i++){
		ent_str = tar_dir + "/" + name[i];
		cv::Mat img_mat(row[i], col[i], CV_8UC1, &gray_pixels[img_ent_start_index]);
		cv::imwrite(ent_str, img_mat, compression_params);
		img_ent_start_index += size[i];
		ent_str.clear();
	}
}

// multiple gpus implementattion
void multi_streams(std::string cur_dir){

	// CPU information
	unsigned long int cpu_free_mem = get_free_mem()*1024;
	std::cout<<std::endl<<"/***** CPU basic information *****/"<<std::endl;
	std::cout<<"Free memory size in CPU(Bs): "<<cpu_free_mem<<std::endl;
	std::cout<<"Free memory size in CPU(Gs): "<<(double)get_free_mem()/(double)(1024*1024)<<std::endl<<std::endl;


	// get the number of GPU devices
	int device_cnt;
	SAFE_CALL(cudaGetDeviceCount(&device_cnt), "Get device count failed!");

	// no cuda supported device
	if( device_cnt<=0 ){
		std::cout<<"There is no CUDA support device!"<<std::endl;
		return;
	}

	// output the GPU information

	/* std::vector<unsigned long int> gpu_free_mem_size_vec; // available memory size for each GPU device */
	/* for( int i=0; i<device_cnt; i++ ){ */
	/*     cudaSetDevice(i); */
	/*     cudaDeviceProp prop; */
	/*     SAFE_CALL(cudaGetDeviceProperties(&prop, i), "Get device properity failed!"); */
	/*     std::cout<<"***** GPU device basic information *****"<<std::endl; */
	/*     std::cout<<"Devicd index: "<<i<<std::endl; */
	/*     std::cout<<"Device name: "<<prop.name<<std::endl; */
	/*     std::cout<<"Global memory(Bytes): "<<prop.totalGlobalMem<<std::endl; */
	/*     std::cout<<"Global memory(GBs): "<<(double)(prop.totalGlobalMem)/(double)(1024*1024*1024)<<std::endl; */
	/*     SAFE_CALL(cudaMemGetInfo(&gpu_free_mem, &gpu_tot_mem), "Get GPU memory info failed!"); */
	/*     std::cout<<"Free memory(Bytes): "<<gpu_free_mem<<std::endl; */
	/*     std::cout<<"Free memory(Gs): "<<(double)gpu_free_mem/(double)(1024*1024*1024)<<std::endl<<std::endl; */
	/*     gpu_free_mem_size_vec.push_back(gpu_free_mem); */
	/*     gpu_free_mem += gpu_free_mem; */
	/* } */

	size_t gpu_free_mem;
	size_t gpu_tot_mem;

	cudaDeviceProp prop;
	SAFE_CALL(cudaGetDeviceProperties(&prop, 0), "Get device properity failed!");
	std::cout<<"/***** GPU device basic information *****/"<<std::endl;
	std::cout<<"Device name: "<<prop.name<<std::endl;
	std::cout<<"Global memory(Bytes): "<<prop.totalGlobalMem<<std::endl;
	std::cout<<"Global memory(GBs): "<<(double)(prop.totalGlobalMem)/(double)(1024*1024*1024)<<std::endl;
	SAFE_CALL(cudaMemGetInfo(&gpu_free_mem, &gpu_tot_mem), "Get GPU memory info failed!");
	std::cout<<"Free memory(Bytes): "<<gpu_free_mem<<std::endl;
	std::cout<<"Free memory(Gs): "<<(double)gpu_free_mem/(double)(1024*1024*1024)<<std::endl<<std::endl;

	// output the total memory information of all GPU devices
	std::cout<<"/***** GPU memory information *****/"<<std::endl;
	std::cout<<"Total free memory in GPU(Bytes): "<<gpu_free_mem<<std::endl;
	std::cout<<"Total free memory in GPU(GBs): "<<(double)gpu_free_mem/(double)(1024*1024*1024)<<std::endl<<std::endl;

	unsigned long int max_mem = (cpu_free_mem<=gpu_free_mem)? cpu_free_mem : gpu_free_mem;
	std::cout<<"Memory limitation(Bytes): "<<max_mem<<std::endl;
	std::cout<<"Memory limitation(GBs): "<<(double)max_mem/(double)(1024*1024*1024)<<std::endl;

	// since each pixel take three bytes, we have to divide the max_mem by some number to get the max pixels we can allocate for one process
	unsigned long int max_pixels_cnt = max_mem/5;

	std::cout<<"Pixels limitation number: "<< max_pixels_cnt<<std::endl;
	std::cout<<"Pixels limitation memory: "<< (double)(max_pixels_cnt*(BYTES_PER_PIXEL+1))/(double)(1024*1024*1024)<<std::endl<<std::endl;

	cudaSetDevice(0);
	std::string src_dir_str(cur_dir+"/src");
	std::string tar_dir_gpu_str(cur_dir+"/gpu_tar");
	std::string tar_dir_cpu_str(cur_dir+"/cpu_tar");

	// string to store and target the directory of source image entity
	std::string img_src_ent_str, img_tar_cpu_ent_str;
	// open the directory of source and enumarate the image files in the directory
	DIR *img_src = opendir(src_dir_str.c_str());
	// open the directory of target of gpu and store the processed image in the directory
	DIR *img_tar_gpu = opendir(tar_dir_gpu_str.c_str());

	// if source and target folder open failed
	if(img_src == NULL || img_tar_gpu == NULL){
		std::cout<<"Can not to open the directory!!!"<<std::endl;
		return;
	}

	// each image entity in source directory
	dirent *img_ent;
	std::vector<std::string> img_name_vec;
	std::vector<unsigned int> img_size_vec; // store the number of pixels each image
	std::vector<int> img_rows_vec;
	std::vector<int> img_cols_vec;

	std::cout<<"Allocating "<< max_pixels_cnt*(BYTES_PER_PIXEL+1)<< "(Bytes)/"<<(double)(max_pixels_cnt*(BYTES_PER_PIXEL+1))/(double)(1024*1024*1024)<<"(GBs) host memory......"<<std::endl;
	unsigned char *h_color_img_pixels, *h_gray_img_pixels, *gray_img_pixels;
	SAFE_CALL(cudaMallocHost( &h_color_img_pixels, max_pixels_cnt * BYTES_PER_PIXEL * sizeof( unsigned char ) ), "Allocate color host image memory failed!");
	SAFE_CALL(cudaMallocHost( &h_gray_img_pixels, max_pixels_cnt * sizeof( unsigned char )), "Allocate gray host image memory failed!");
	gray_img_pixels = (unsigned char*)malloc( max_pixels_cnt * sizeof(unsigned char) );

	cv::Mat img;
	unsigned int tot_img_cnt = 0;
	unsigned int collect_img_cnt = 0;
	unsigned long int process_pixels_cnt = 0;

	// enumerate every image file in current directory and process
	std::cout<<"Collecting images pixels......"<<std::endl;
	while((img_ent = readdir(img_src))){
		// if detect "." and ".." , we ignore them
		if(strcmp(img_ent->d_name, ".") == 0 || strcmp(img_ent->d_name, "..")==0)
			continue;
		else{
			img_src_ent_str = src_dir_str + "/" + img_ent->d_name;
			img = cv::imread(img_src_ent_str, CV_LOAD_IMAGE_COLOR);

			if(img.empty()){
				std::cout<<"Image Not Found!"<<std::endl;
				return;
			}

			tot_img_cnt++;

			//std::cout<<"Processing "<<tot_img_cnt<<" images"<<std::endl;
			if( process_pixels_cnt + img.total() <= max_pixels_cnt ){
				collect_img_cnt++;
				//std::cout<<"Image name: "<<img_ent->d_name<<std::endl<<std::endl;
				img_name_vec.push_back(img_ent->d_name);
				img_size_vec.push_back(img.total());
				img_rows_vec.push_back(img.rows);
				img_cols_vec.push_back(img.cols);
				memcpy(&h_color_img_pixels[process_pixels_cnt*BYTES_PER_PIXEL*sizeof(unsigned char)],img.ptr(),img.total()*BYTES_PER_PIXEL*sizeof(unsigned char));
				process_pixels_cnt += img.total();
			} else {
				std::cout<<"Process pixels(GBs): "<<(double)(process_pixels_cnt*BYTES_PER_PIXEL)/(double)(1024*1024*1024)<<std::endl<<std::endl;
				// CPU processing
				gray_processing_CPU(h_color_img_pixels, gray_img_pixels, process_pixels_cnt);
				write_images(gray_img_pixels,tar_dir_cpu_str , img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

				// GPU processing
				gray_processing_multi_streams(h_color_img_pixels, h_gray_img_pixels, process_pixels_cnt);
				write_images(h_gray_img_pixels, tar_dir_gpu_str, img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

				// clear the variables after one CPU and GPU processing
				memset(h_color_img_pixels, 0, max_pixels_cnt*BYTES_PER_PIXEL*sizeof(unsigned char));
				memset(h_gray_img_pixels, 0, max_pixels_cnt*sizeof(unsigned char));
				memset(gray_img_pixels, 0, max_pixels_cnt*sizeof(unsigned char));
				img_name_vec.clear();
				img_size_vec.clear();
				img_rows_vec.clear();
				img_cols_vec.clear();

				// add current image to variables
				// for next big chunk processing
				collect_img_cnt = 1;
				img_name_vec.push_back(img_ent->d_name);
				img_size_vec.push_back(img.total());
				img_rows_vec.push_back(img.rows);
				img_cols_vec.push_back(img.cols);
				memcpy(&h_color_img_pixels[0], img.ptr(), img.total()*BYTES_PER_PIXEL*sizeof(unsigned char));
				process_pixels_cnt = img.total();
				std::cout<<"Collecting images pixels......"<<std::endl;
			}
		}
	}

	std::cout<<"Process pixels(GBs): "<<(double)(process_pixels_cnt*BYTES_PER_PIXEL)/(double)(1024*1024*1024)<<std::endl<<std::endl;
	// CPU processing for last collection
	gray_processing_CPU(h_color_img_pixels, gray_img_pixels, process_pixels_cnt);
	write_images(gray_img_pixels,tar_dir_cpu_str , img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

	// GPU processing for last collection
	gray_processing_multi_streams(h_color_img_pixels, h_gray_img_pixels, process_pixels_cnt);
	write_images(h_gray_img_pixels, tar_dir_gpu_str, img_name_vec, img_size_vec, img_rows_vec, img_cols_vec, collect_img_cnt);

	// release allocated memory
	free(gray_img_pixels);
	SAFE_CALL(cudaFreeHost(h_color_img_pixels), "Free host color memory failed!");
	SAFE_CALL(cudaFreeHost(h_gray_img_pixels), "Free host gray memory failed!");

	std::cout<<"Images processing completed"<<std::endl;
}
