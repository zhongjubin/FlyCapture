#include <unistd.h>
#include "stdafx.h" 
#include <iostream> 
#include <sstream>
#include "FlyCapture2.h"
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>  
#include <opencv2/legacy/compat.hpp>
#include <zbar.h>  
#include <stdlib.h> 
#include <time.h>
#include "FlyCapture2Defs.h"
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <python2.7/Python.h>
#include <math.h>
#include <queue>

#define show_image
#define image_interval 300
#define core 12
#define divider 12 
//#define serial_t
#define local_test
#define shrink 1
#define spand 1


using namespace cv;  
using namespace std;  
using namespace zbar;  
using namespace FlyCapture2;

static pthread_mutex_t img_reading_mtx;
static pthread_mutex_t img_cnt_mtx;
static pthread_mutex_t global_exit_mtx; 
static pthread_mutex_t printf_mtx;
static int  global_exit =0;

int cp_blk_size=9;
int cp_blk_par=25; 
int blk_size=3;
int blk_par=20; 

const int AREA_SIZE=18000;
const double RATIO=0.7;

typedef struct rect
{
	Rect r;
	double ratio;
}str_rect;
int comp(const void*a, const void*b)
{
	return *(double*)a-*(double*)b;
}


void cvWiener2( const void* srcArr, void* dstArr, int szWindowX, int szWindowY )
{
	CV_FUNCNAME( "cvWiener2" );

	int nRows;
	int nCols;
    CvMat *p_kernel = NULL;
    CvMat srcStub, *srcMat = NULL;    CvMat *p_tmpMat1, *p_tmpMat2, *p_tmpMat3, *p_tmpMat4;
	double noise_power;

	//// DO CHECKING ////

	if ( srcArr == NULL) {
		printf("Source array null" );
	}
	if ( dstArr == NULL) {
		printf( "Dest. array null" );
	}

	nRows = szWindowY;
	nCols = szWindowX;


	p_kernel = cvCreateMat( nRows, nCols, CV_32F );
	cvSet( p_kernel, cvScalar( 1.0 / (double) (nRows * nCols)) );

	//Convert to matrices
	srcMat = (CvMat*) srcArr;

	if ( !CV_IS_MAT(srcArr) ) {
		srcMat = cvGetMat(srcMat, &srcStub, 0, 1) ;
	}

	//Now create a temporary holding matrix
	p_tmpMat1 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
	p_tmpMat2 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
	p_tmpMat3 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
	p_tmpMat4 = cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));

	//Local mean of input
	cvFilter2D( srcMat, p_tmpMat1, p_kernel, cvPoint(nCols/2, nRows/2)); //localMean

	//Local variance of input
	cvMul( srcMat, srcMat, p_tmpMat2);	//in^2
	cvFilter2D( p_tmpMat2, p_tmpMat3, p_kernel, cvPoint(nCols/2, nRows/2));
	
	//Subtract off local_mean^2 from local variance
	cvMul( p_tmpMat1, p_tmpMat1, p_tmpMat4 ); //localMean^2
	cvSub( p_tmpMat3, p_tmpMat4, p_tmpMat3 ); //filter(in^2) - localMean^2 ==> localVariance

	//Estimate noise power
	CvScalar noise_tmp = cvAvg(p_tmpMat3, 0);
	noise_power = 0.299*noise_tmp.val[0]+0.587*noise_tmp.val[1]+0.114*noise_tmp.val[2];
	// result = local_mean  + ( max(0, localVar - noise) ./ max(localVar, noise)) .* (in - local_mean)

	cvSub ( srcMat, p_tmpMat1, dstArr);		     //in - local_mean
	cvMaxS( p_tmpMat3, noise_power, p_tmpMat2 ); //max(localVar, noise)

	cvAddS( p_tmpMat3, cvScalar(-noise_power), p_tmpMat3 ); //localVar - noise
	cvMaxS( p_tmpMat3, 0, p_tmpMat3 ); // max(0, localVar - noise)

	cvDiv ( p_tmpMat3, p_tmpMat2, p_tmpMat3 );  //max(0, localVar-noise) / max(localVar, noise)

	cvMul ( p_tmpMat3, dstArr, dstArr );
	cvAdd ( dstArr, p_tmpMat1, dstArr );
	
//	cvDFT(srcMat,dstArr,CV_DXT_INV_SCALE);
	cvReleaseMat( &p_kernel  );
	cvReleaseMat( &p_tmpMat1 );
	cvReleaseMat( &p_tmpMat2 );
	cvReleaseMat( &p_tmpMat3 );
	cvReleaseMat( &p_tmpMat4 );
}

void PrintBuildInfo()
{
    FC2Version fc2Version;
    Utilities::GetLibraryVersion( &fc2Version );
    
	ostringstream version;
	version << "FlyCapture2 library version: " << fc2Version.major << "." << fc2Version.minor << "." << fc2Version.type << "." << fc2Version.build;
	cout << version.str() << endl;  
    
	ostringstream timeStamp;
    timeStamp <<"Application build date: " << __DATE__ << " " << __TIME__;
	cout << timeStamp.str() << endl << endl;  
}

void PrintCameraInfo( CameraInfo* pCamInfo )
{
    cout << endl;
	cout << "*** CAMERA INFORMATION ***" << endl;
	cout << "Serial number -" << pCamInfo->serialNumber << endl;
    cout << "Camera model - " << pCamInfo->modelName << endl;
    cout << "Camera vendor - " << pCamInfo->vendorName << endl;
    cout << "Sensor - " << pCamInfo->sensorInfo << endl;
    cout << "Resolution - " << pCamInfo->sensorResolution << endl;
    cout << "Firmware version - " << pCamInfo->firmwareVersion << endl;
    cout << "Firmware build time - " << pCamInfo->firmwareBuildTime << endl << endl;	
}

void PrintError( Error error )
{
    error.PrintErrorTrace();
}

CvPoint transformPoint(const CvPoint pointToTransform, const CvMat* matrix) 
{
    double coordinates[3] = {pointToTransform.x, pointToTransform.y, 1};
    CvMat originVector = cvMat(3, 1, CV_64F, coordinates);
    CvMat transformedVector = cvMat(3, 1, CV_64F, coordinates);
    cvMatMul(matrix, &originVector, &transformedVector);
    CvPoint outputPoint = cvPoint((int)(cvmGet(&transformedVector, 0, 0) / cvmGet(&transformedVector, 2, 0)), (int)(cvmGet(&transformedVector, 1, 0) / cvmGet(&transformedVector, 2, 0)));
    return outputPoint;
}

Camera cam;    // Connect to a camera
int decodeCnt=0;
int imageCnt=0;
int decoded = 0;
int pre_decoded=0;
FlyCapture2::Image rawImage[core];
cv::Mat frame[core], grey[core];
IplImage *grey2[core],*dst[core],*shw_tmp[core];
IplImage ipltmp[core];
FlyCapture2::Image convertedImage[core];

static char svr_in_buf[100];
queue<char*> cam_out_buf;
char cam_out_once[100];

void* image_processing(void* arg)
{
	int count=*((int*)arg);
	Error error;
	int local_exit = 0;
	zbar::ImageScanner scanner;  
	scanner.set_config(ZBAR_NONE, ZBAR_CFG_ENABLE, 1);   
	//FlyCapture2::Image convertedImage;
	
	while(!local_exit)
	{
		pthread_mutex_lock(&global_exit_mtx);
		if(global_exit==1)
			local_exit=1;
		pthread_mutex_unlock(&global_exit_mtx);
		
		//The camera can support up to 8 channels for simutanously data transfer.
		//Two things for noticing:
		//1.Do not set up more than 8 connections between odroid memory and camera buffer(that's why we use count%8)
		//2.If we send two request in too short time, the camera may crash("No new buffer error"...that's why we have to use mutex).	
		pthread_mutex_lock(&img_reading_mtx);	
		error = cam.RetrieveBuffer(&rawImage[count%8]);
		if (error != PGRERROR_OK)
		{
			PrintError( error );
			continue;
		}    	
		error = rawImage[count%8].Convert(FlyCapture2::PIXEL_FORMAT_BGR,&convertedImage[count]);
		pthread_mutex_unlock(&img_reading_mtx);
				
		// error = rawImage.Convert( PIXEL_FORMAT_MONO8, &convertedImage ); 

		unsigned int rowBytes = (double)convertedImage[count].GetReceivedDataSize()/(double)convertedImage[count].GetRows();
		frame[count] = cv::Mat(convertedImage[count].GetRows(),convertedImage[count].GetCols(),CV_8UC3,convertedImage[count].GetData(),rowBytes); 	
		cvtColor(frame[count],grey[count],CV_BGR2GRAY);
		
		// Convert the raw image
		pthread_mutex_lock(&img_cnt_mtx);
		imageCnt = (imageCnt+1)%1510;
		int local_imageCnt=imageCnt;
		pthread_mutex_unlock(&img_cnt_mtx);
		
		//Read form local picture.	
	#ifdef local_test		
		char read_image[20];
		sprintf(read_image,"./image1/%d.bmp",local_imageCnt);
		//sprintf(read_image,"./image2/2.bmp");
		int fd = open(read_image,O_RDWR);
		if(fd==-1)
			continue;
		else
			close(fd);
		grey[count]=imread(read_image);
		cvtColor(grey[count],grey[count],CV_BGR2GRAY);	
	#endif

		//Sharpening: not good
		/*
		Mat kernel(3,3,CV_32F,Scalar(-1));
		kernel.at<float>(1,1)=8.99;
		filter2D(grey[count],grey[count],grey[count].depth(),kernel);
		*/	
		
		//Median Filter: good
		medianBlur(grey[count],grey[count],3);
			
		//Only create once to avoid memory leakage
		if(grey2[count] == NULL)
			grey2[count] = cvCreateImage(cvSize(grey[count].cols,grey[count].rows),IPL_DEPTH_8U,1);
		ipltmp[count]=grey[count];
		cvCopy(&ipltmp[count],grey2[count]);
		
		//only create once to avoid memory leakage
		if(dst[count] == NULL)
			dst[count] = cvCreateImage(cvGetSize(grey2[count]),IPL_DEPTH_8U,1);
		//cvThreshold(grey2,dst,CV_THRESH_OTSU,255,CV_THRESH_BINARY);
		int block = (dst[count]->width/divider)%2==0?dst[count]->width/divider+1:dst[count]->width/divider;
		
		//----------------------------------cropping the target area--------------------------------------//
		cvAdaptiveThreshold(grey2[count],dst[count],255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY_INV,block,cp_blk_par);	
		Mat crop(dst[count],true);	
		Mat se1(cp_blk_size,cp_blk_size,CV_8U,Scalar(1));
		morphologyEx(crop,crop,MORPH_CLOSE,se1);
		Mat crop1 = crop.clone();
	
		cvAdaptiveThreshold(grey2[count],dst[count],255,CV_ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,block,blk_par);		
		Mat close_tmp(dst[count],true);	
		Mat se2(blk_size,blk_size,CV_8U,Scalar(1));
		morphologyEx(close_tmp,close_tmp,MORPH_CLOSE,se2);
		 
		int largest = 0;
		int largest_index = 0;
		Mat close;		
		vector< Rect > ROI_list;
		vector< vector<Point> > contours;
		vector<Vec4i> hierarchy;
		str_rect contour_list[20];

		findContours(crop,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);		
		if(contours.size()>0)
		{
			int contour_cnt = -1;
			for(int i=0;i<contours.size();i++)
			{
				//counting the area size
				double area_ratio=abs(contourArea(contours[i],false)/AREA_SIZE-1);
				//counting width/height ratio
				RotatedRect r = minAreaRect(contours[i]);   
				Point2f pts[4];   
				r.points(pts);   
				double hw_ratio=abs(sqrt(pow(pts[0].x-pts[1].x,2)+pow(pts[0].y-pts[1].y,2))/sqrt(pow(pts[2].x-pts[1].x,2)+pow(pts[2].y-pts[1].y,2))-1);
				if(area_ratio+hw_ratio<=RATIO)
				{
					str_rect rect_tmp;
					rect_tmp.r=boundingRect(contours[i]);;
					rect_tmp.ratio=area_ratio+hw_ratio;
					contour_cnt++;
					contour_list[contour_cnt]=rect_tmp;
					
				}
			}
			if(contour_cnt>=0)
			{
				qsort(contour_list,contour_cnt+1,sizeof(str_rect),comp);
				for(int i=0;i<3 && i<contour_cnt+1;i++)
				{
					Rect ROI = contour_list[i].r;	
					ROI.x = ROI.x -5>0?ROI.x-5:0;
					ROI.y = ROI.y -5>0?ROI.y-5:0;
					ROI.width = (ROI.x+ROI.width+10)<close_tmp.cols?ROI.width+10:close_tmp.cols-ROI.x;
					ROI.height = (ROI.y+ROI.height+10)<close_tmp.rows?ROI.height+10:close_tmp.rows-ROI.y;
					ROI_list.push_back(ROI);	
				}
			}
		}
		//Simply take tha largest block as the block containing the qrcode.
		/*	
		if(contours.size()>0)
		{
			for(int i=0;i<contours.size();i++)
			{
				double a=contourArea(contours[i],false);
				if(a>largest)
				{
					largest=a;
					largest_index=i;
				}
			}
			Rect ROI=boundingRect(contours[largest_index]);
			ROI.x = ROI.x -5>0?ROI.x-5:0;
			ROI.y = ROI.y -5>0?ROI.y-5:0;
			ROI.width = (ROI.x+ROI.width+10)<close_tmp.cols?ROI.width+10:close_tmp.cols-ROI.x;
			ROI.height = (ROI.y+ROI.height+10)<close_tmp.rows?ROI.height+10:close_tmp.rows-ROI.y;
			close=close_tmp(ROI).clone();
		}
		//Notice: if contours.size()==0, then close=close_tmp(ROI).clone will cause "Segment fault"
		else
			close=close_tmp.clone();
		*/	
		//----------------------------------cropping the target area--------------------------------------//
		

		//Notice: if contours.size()==0, then close=close_tmp(ROI).clone will cause "Segment fault"
		Mat img_merge;	
		Point2f pts[1024][4];
		int local_decoded = 0;	
				
		if(ROI_list.size()<=0)
		{
			close=close_tmp.clone();
			img_merge=close_tmp.clone();
		}
		//only detects the qrcode when something is detected.
		else
		{
			img_merge.create(cvSize(close_tmp.cols+20,close_tmp.rows+20),CV_8UC1);
			img_merge = Scalar::all(0);
			for(int j=0;j<ROI_list.size();j++)
			{
				close=close_tmp(ROI_list[j]).clone();
				close.copyTo(img_merge(ROI_list[j]));
			
				int width = close.cols;  
				int height = close.rows;  
				//cout<<local_imageCnt<<" "<<ROI.x<<" "<<ROI.y<<" "<<width<<" "<<height<<endl;
						
				//cvWiener2(dst[count],dst[count],1,1);
				//Perspective transform: not good
				CvMat* warp_mat = cvCreateMat(3,3,CV_64FC1);
				if(shw_tmp[count]==NULL)
					shw_tmp[count] = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
				
				static	CvPoint2D32f corners[4],corners_trans[4];
				static double scale = (double)height/(spand+height);
				corners[0] = Point2f(0,0);  
				corners[1] = Point2f(width-1,0);  
				corners[2] = Point2f(0,height-1);  
				corners[3] = Point2f(width-1,height-1);  
				corners_trans[0] = Point2f(0,0);  
				corners_trans[1] = Point2f((int)((width-1)*scale),0);  
				corners_trans[2] = Point2f((int)(shrink*scale),(int)((height+spand)*scale));  
				corners_trans[3] = Point2f((int)((width-shrink)*scale),(int)((height+spand)*scale));  
				 
				cvGetPerspectiveTransform(corners,corners_trans,warp_mat);
				cvWarpPerspective(dst[count],shw_tmp[count],warp_mat,CV_WARP_FILL_OUTLIERS+CV_INTER_LINEAR,cvScalarAll(0)); 
			
				uchar *raw = (uchar *)close.data;
				//uchar *raw = (uchar *)shw_tmp[count]->imageData;   		
				//uchar *raw = (uchar *)dst[count]->imageData;
				
				//pthread_mutex_lock(&img_cnt_mtx);	
				//if(local_imageCnt>=1500)
				//	global_exit=1;
				//pthread_mutex_unlock(&img_cnt_mtx);
			
				char file_name[20];
				sprintf(file_name,"./image/%d.bmp",local_imageCnt);		
				//cvSaveImage(file_name,shw_tmp[count]);		
				imwrite(file_name,close);

				
				//PyObject *pModule = NULL;
				//PyObject *pFunc = NULL;
				//PyObject *pArg = NULL;
				//pModule = PyImport_ImportModule("deconvolution");
				//pFunc = PyObject_GetAttrString(pModule,"hello" );	
				//PyEval_CallObject(pFunc,pArg);
				
				// wrap image data   
				zbar::Image image = zbar::Image(width, height, "Y800", raw, width * height);
				// scan the image for barcodes   
				int n = scanner.scan(image);
				
				// extract results	
				vector<Point> vp;	
				for(zbar::Image::SymbolIterator symbol = image.symbol_begin();symbol != image.symbol_end();++symbol)
				{		
					pthread_mutex_lock(&img_cnt_mtx);
					//Send the message, remain true until it is handled(sent through uart)
					decoded=1;
					decodeCnt++;
					int decodeCnt_tmp = decodeCnt;
					pthread_mutex_unlock(&img_cnt_mtx);
				
					char qrdata[100];
					memset(qrdata,0,sizeof(qrdata));
					strcpy(qrdata,symbol->get_data().data());  
					strcat(qrdata,"\n");
					
					pthread_mutex_lock(&img_cnt_mtx);
					pthread_mutex_lock(&printf_mtx); 	
					//if(local_exit==0)
					//	cout <<local_imageCnt<< " decoded "<<decodeCnt<<" " << symbol->get_type_name() << " symbol \"" << symbol->get_data() << "\" " << endl;  		
					if((!cam_out_buf.empty() && strcmp(qrdata,cam_out_buf.back())) || (cam_out_buf.empty() && strcmp(qrdata,cam_out_once)))
					{	
						cam_out_buf.push(qrdata);	 
					}
					pthread_mutex_unlock(&printf_mtx);
					pthread_mutex_unlock(&img_cnt_mtx);
					
					int n = symbol->get_location_size();   
					
					for(int i=0;i<n;i++)
					{   
						vp.push_back(Point(symbol->get_location_x(i),symbol->get_location_y(i)));   
					}   
					RotatedRect r = minAreaRect(vp);     
					r.points(pts[local_decoded]);
					local_decoded++;   
					for(int i=0;i<4;i++)
					{   
						line(grey[count],pts[local_decoded][i],pts[local_decoded][(i+1)%4],Scalar(255,0,0),3);   
					}			
				}		
			}
			if(local_decoded==0)
			{
				pthread_mutex_lock(&img_cnt_mtx);
				pthread_mutex_lock(&printf_mtx);	
				//cout<<local_imageCnt<<" failure\n";
				pthread_mutex_unlock(&printf_mtx);
				pthread_mutex_unlock(&img_cnt_mtx);
			}
			else
			{
				pthread_mutex_lock(&img_cnt_mtx);
				pthread_mutex_lock(&printf_mtx);	
				//cout<<local_imageCnt<<" succeed\n";
				pthread_mutex_unlock(&printf_mtx);
				pthread_mutex_unlock(&img_cnt_mtx);
			}		
		}	
	#ifdef show_image  
		if(count==0)
		{
		//cvShowImage("MyVideo",dst[count]); 
			imshow("MyVideo",img_merge);
			cvCreateTrackbar("cp_blk_size","MyVideo",&cp_blk_size,50,NULL);
			cvCreateTrackbar("cp_blk_par","MyVideo",&cp_blk_par,100,NULL);
			cvCreateTrackbar("blk_size","MyVideo",&blk_size,50,NULL);
			cvCreateTrackbar("blk_par","MyVideo",&blk_par,100,NULL);
			
			//imshow("MyVideo", grey[count]); //show the frame in "MyVideo" window  
			waitKey(1);
		}
	#endif
	}
}

void* server_fifo(void* arg)
{
	const char* in_filename = "./svr_in";
	const char*  out_filename = "./cam_out";
	int in_fd;
	int out_fd;
	
	if((out_fd = open(out_filename,O_WRONLY))<0)
	{
		perror("open output fifo error");
		mkfifo(out_filename,666);
		out_fd = open(out_filename,O_WRONLY);
	}
	if((in_fd = open(in_filename,O_RDONLY))<0)
	{
		perror("open input fifo error");
		mkfifo(in_filename,666);
		in_fd = open(in_filename,O_RDONLY);
	}	
	clock_t start,finish;
	struct timespec begin,end;
	double fifo_interval=0;
	char no_qr_out[100];
	memset(no_qr_out,0,sizeof(no_qr_out));
	strcpy(no_qr_out,"no qr code\n");
	
	memset(cam_out_once,0,sizeof(cam_out_once));
	while(1)
	{
		if(global_exit==1)
			break;
		start=clock();
		clock_gettime(CLOCK_MONOTONIC,&begin);
		static bool buf_is_empty = true;
		do
		{	
			clock_gettime(CLOCK_MONOTONIC,&end);
			fifo_interval=(double)((end.tv_sec-begin.tv_sec)+(double)(end.tv_nsec-begin.tv_nsec)/1000000000);
			pthread_mutex_lock(&img_cnt_mtx);
			buf_is_empty = cam_out_buf.empty();
			pthread_mutex_unlock(&img_cnt_mtx);
		}while(abs(fifo_interval)<1 && buf_is_empty);
		finish = clock();
			
		if(!buf_is_empty)
		{	
			pthread_mutex_lock(&img_cnt_mtx);
			strcpy(cam_out_once,cam_out_buf.front());
			cam_out_buf.pop();	
			buf_is_empty = cam_out_buf.empty();
			pthread_mutex_unlock(&img_cnt_mtx);
			write(out_fd,cam_out_once,sizeof(cam_out_once));		
		}
		else
			write(out_fd,no_qr_out,sizeof(no_qr_out));
		
		memset(svr_in_buf,0,sizeof(svr_in_buf));
		cout<<"Retrieving server data..."<<endl;	
		int rd_num = read(in_fd, svr_in_buf,sizeof(svr_in_buf));
		if(rd_num==0)
			sleep(1);  // read FIFO if ready to read	
		
		pthread_mutex_lock(&printf_mtx);
		cout<<svr_in_buf<<endl;
		pthread_mutex_unlock(&printf_mtx);	}	
}




char wr_buf[100];
int serial_fd;
char buf[100];
int16_t serial_data[4];

int set_interface_attribs (int fd, int speed, int parity)
{
        struct termios tty;
        memset (&tty, 0, sizeof tty);
        if (tcgetattr (fd, &tty) != 0)
        {
                printf("error %d from tcgetattr", errno);
                return -1;
        }

        cfsetospeed (&tty, speed);
        cfsetispeed (&tty, speed);

        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
        // disable IGNBRK for mismatched speed tests; otherwise receive break
        // as \000 chars
        tty.c_iflag &= ~IGNBRK;         // disable break processing
        tty.c_lflag = 0;                // no signaling chars, no echo,
                                        // no canonical processing
        tty.c_oflag = 0;                // no remapping, no delays
        tty.c_cc[VMIN]  = 0;            // read doesn't block
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

        tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl

        tty.c_cflag |= (CLOCAL | CREAD);// ignore modem controls,
                                        // enable reading
        tty.c_cflag &= ~(PARENB | PARODD);      // shut off parity
        tty.c_cflag |= parity;
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;

        if (tcsetattr (fd, TCSANOW, &tty) != 0)
        {
                printf ("error %d from tcsetattr", errno);
                return -1;
        }
        return 0;
}

void set_blocking (int fd, int should_block)
{
        struct termios tty;
        memset (&tty, 0, sizeof tty);
        if (tcgetattr (fd, &tty) != 0)
        {
                printf ("error %d from tggetattr", errno);
                return;
        }

        tty.c_cc[VMIN]  = should_block ? 1 : 0;
        tty.c_cc[VTIME] = 5;            // 0.5 seconds read timeout

        if (tcsetattr (fd, TCSANOW, &tty) != 0)
                printf ("error %d setting term attributes", errno);
}


void* listen_serial(void* fd_s)
{
	int fd_index=*((int *)fd_s);
	fd_set rdfds;
	struct timeval tv;
	int ret;
	static char* buf_seek;
	ssize_t read_num;
	FD_ZERO(&rdfds);
	FD_SET(fd_index,&rdfds);
	while(1)
	{
		if(global_exit==1)
			break;
		tv.tv_sec=1;
		tv.tv_usec=100;
		ret=select(fd_index+1,&rdfds,NULL,NULL,&tv);
		if(ret<0)
			perror("select");
		else if(ret==0)
			continue;
		else
		{
			if(FD_ISSET(fd_index,&rdfds))
			{
				memset(buf,0,sizeof(buf));
				read_num=0;
				buf_seek=buf;
				while(buf_seek-buf<10)
				{
					read_num = read(fd_index, buf_seek,sizeof(buf));  // read up to 100 characters if ready to read
					buf_seek+=read_num;
					cout<<buf_seek-buf<<endl;	
				}
				buf[10]=0;
				for(int i=0;i<4;i++)
				{
					serial_data[i]=buf[2*i+1]*256+buf[2*i];
					serial_data[i]>2^15-1?(int16_t)(serial_data[i]-(int16_t)(2^16)):serial_data[i];
				}
				pthread_mutex_lock(&printf_mtx);
				for(int i=0;i<4;i++)
					cout<<serial_data[i]<<" ";	
				cout<<endl;
				pthread_mutex_unlock(&printf_mtx);
			}
		}
	}	
}


void* write_serial(void *fd_s)
{
	int fd_index=*((int *)fd_s);
	
	pthread_mutex_lock(&img_cnt_mtx);	
	if(decoded==1 && pre_decoded==0)
	{
		pre_decoded=1;
		decoded = 0;
		pthread_mutex_unlock(&img_cnt_mtx);		
		memset(wr_buf,0,sizeof(wr_buf));
		strcpy(wr_buf,"getqr");
		write(fd_index,wr_buf,sizeof(wr_buf));
	}
	else if(decoded==0 && pre_decoded==1)
	{
		pre_decoded=0;
		pthread_mutex_unlock(&img_cnt_mtx);			
		memset(wr_buf,0,sizeof(wr_buf));
		strcpy(wr_buf,"lostqr");
		write(fd_index,wr_buf,sizeof(wr_buf));
	}
	else
		pthread_mutex_unlock(&img_cnt_mtx);				
	sleep(1);
}

int RunSingleCamera( PGRGuid guid )
{
    const int k_numImages = 50;
	Error error;
	error = cam.Connect(&guid);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return -1;
    }
	 // Get the camera information
    CameraInfo camInfo;
    error = cam.GetCameraInfo(&camInfo);
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return -1;
    }

    PrintCameraInfo(&camInfo);
	
	const Mode k_fmt7Mode = MODE_1;
	const PixelFormat k_fmt7PixFmt = PIXEL_FORMAT_MONO8;

	Format7Info fmt7Info;
	bool supported;
	fmt7Info.mode = k_fmt7Mode;
	error = cam.GetFormat7Info( &fmt7Info, &supported );
	if (error != PGRERROR_OK)
	{
    	PrintError( error );
    	return -1;
	}


	if ( (k_fmt7PixFmt & fmt7Info.pixelFormatBitField) == 0 )
	{
    	// Pixel format not supported!
    	cout << "Pixel format is not supported" << endl;
    	return -1;
	}
	
	Format7ImageSettings fmt7ImageSettings;
	fmt7ImageSettings.mode = k_fmt7Mode;
	/*
	fmt7ImageSettings.offsetX = 0;
	fmt7ImageSettings.offsetY = 0;
	fmt7ImageSettings.width = 640;
	fmt7ImageSettings.height = 512;
	*/
	fmt7ImageSettings.offsetX = 160;
	fmt7ImageSettings.offsetY = 128;
	fmt7ImageSettings.width = 320;
	fmt7ImageSettings.height = 256;
	fmt7ImageSettings.pixelFormat = k_fmt7PixFmt;
    bool valid;
	Format7PacketInfo fmt7PacketInfo;

	// Validate the settings to make sure that they are valid
	error = cam.ValidateFormat7Settings(&fmt7ImageSettings,&valid,&fmt7PacketInfo );
	if (error != PGRERROR_OK)
	{
		PrintError( error );
		return -1;
	}
    if ( !valid )
	{
    	// Settings are not valid
    	cout << "Format7 settings are not valid" << endl;
    	return -1;
	}

	// Set the settings to the camera
	error = cam.SetFormat7Configuration(&fmt7ImageSettings,fmt7PacketInfo.recommendedBytesPerPacket );
	if (error != PGRERROR_OK)
	{
    	PrintError( error );
    	return -1;
	}
	
	//set the capturing attributes
	
	Property prop;

	prop.type = BRIGHTNESS;
	prop.absControl = true;
	prop.absValue = 2.5;
	error = cam.SetProperty(&prop);

	prop.type = SHUTTER;
	prop.onOff = true;
	prop.autoManualMode = false;
	prop.absControl = true;
	prop.absValue = 2;
	error = cam.SetProperty(&prop);
	
	prop.type = GAIN;
	prop.autoManualMode = false;
	prop.absControl = true;
	prop.absValue = 15;
	error = cam.SetProperty(&prop);

	prop.type = AUTO_EXPOSURE;
	prop.onOff = true;
	prop.autoManualMode = false;
	prop.absControl = true;
	prop.absValue = 1.1;
	error = cam.SetProperty(&prop);
	
	prop.type = GAMMA;
	prop.onOff = true;
	prop.absControl = true;
	prop.absValue = 1;
	error = cam.SetProperty(&prop);

	prop.type = SHARPNESS;
	prop.onOff = true;
	prop.autoManualMode = false;
	prop.valueA = 950;
	error = cam.SetProperty(&prop);

	// Start capturing images
    error = cam.StartCapture();
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return -1;
    }

	char stop[1];
	int ret;
	int tmp[core];
	clock_t start,finish;
	struct timespec begin,end;
	double speed=0; 
	double cost_time=0;
	pthread_t id[core];
	
#ifdef serial_t
	char portname[] = "/dev/ttyUSB0";
	int serial_open=0;
	pthread_t id_s;
	pthread_t id_s_wr;
	
	serial_fd  = open (portname, O_RDWR | O_NOCTTY | O_SYNC);
	if (serial_fd < 0)
	{
		printf("error %d opening %s: %s", errno, portname, strerror (errno));
	}
	else
	{
		serial_open=1;
		set_interface_attribs (serial_fd, B115200, 0);  // set speed to 115,200 bps, 8n1 (no parity)
		set_blocking (serial_fd, 0);                // set no blocking

		pthread_create(&id_s,NULL,listen_serial,&serial_fd);
		pthread_create(&id_s_wr,NULL,write_serial,&serial_fd);
	}
#endif
	pthread_t id_svr;
	pthread_create(&id_svr,NULL,server_fifo,NULL);
	//pthread_mutex_lock(&img_reading_mtx);
	for(int cnt=0;cnt<core;cnt++){ 
		tmp[cnt]=cnt;
		ret=pthread_create(&id[cnt],NULL,image_processing,&tmp[cnt]);
   		if(ret!=0){
       		printf ("Create pthread error!\n");
       		exit (1);
   		}	
	}
	//pthread_mutex_unlock(&img_reading_mtx);

	int tmp_imageCnt=0;
	int tmp_decodeCnt=0;

	while(1)
	{	
		pthread_mutex_lock(&img_cnt_mtx);
		tmp_imageCnt = imageCnt;
		tmp_decodeCnt = decodeCnt;
		pthread_mutex_unlock(&img_cnt_mtx);
		    
		if(tmp_imageCnt<=5)
		{
			start=clock();
			clock_gettime(CLOCK_MONOTONIC,&begin);
		} 
		if(0)
		//if(tmp_decodeCnt>=image_interval || tmp_imageCnt>=1500)
		{
			finish=clock();
			clock_gettime(CLOCK_MONOTONIC,&end);
			speed=(double)tmp_decodeCnt/(double)((end.tv_sec-begin.tv_sec)+(double)(end.tv_nsec-begin.tv_nsec)/1000000000);
			cost_time=(double)((end.tv_sec-begin.tv_sec)+(double)(end.tv_nsec-begin.tv_nsec)/1000000000);	
				
			pthread_mutex_lock(&global_exit_mtx);
			pthread_mutex_lock(&img_cnt_mtx);	
			pthread_mutex_lock(&printf_mtx);
	
			cout<< "Speed is "<< speed<<". Time is "<<cost_time<<". Rate is "<<(float)decodeCnt/(float)imageCnt<<endl;	
			cout<< "Terminate the program?(y/n):";
			imageCnt=0;
			decodeCnt=0;
			stop[0]=waitKey(5000);					
			//cin>>stop[0];

			if(stop[0] == 'y')
			{
				global_exit=1;
				pthread_mutex_unlock(&printf_mtx);
				pthread_mutex_unlock(&img_cnt_mtx);
				pthread_mutex_unlock(&global_exit_mtx);
			
				sleep(1);	

				for(int cnt=0;cnt<core;cnt++)
    				pthread_join(id[cnt],NULL);
#ifdef serial_t		
				if(serial_open==1)
				{		
					pthread_join(id_s,NULL);
					pthread_join(id_s_wr,NULL);
					close(serial_fd);
				}
#endif
				
				pthread_mutex_destroy(&global_exit_mtx);		
				pthread_mutex_destroy(&img_reading_mtx);
				pthread_mutex_destroy(&img_cnt_mtx);
				pthread_mutex_destroy(&printf_mtx);
			
				cout<<"Receiving stop signal...Done!\n";
				cam.StopCapture();
				cout<<"Stop Capturing...Done!\n";
				cam.Disconnect();
				cout<<"Disconnecting...Done!\n";
				return 0;
			}
			pthread_mutex_unlock(&printf_mtx);	
			pthread_mutex_unlock(&img_cnt_mtx);
			pthread_mutex_unlock(&global_exit_mtx);
		}
			
	}
    // Stop capturing images
    error = cam.StopCapture();
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return -1;
    }      

    // Disconnect the camera
    error = cam.Disconnect();
    if (error != PGRERROR_OK)
    {
        PrintError( error );
        return -1;
    }

    return 0;
}


int main(int /*argc*/, char** /*argv*/)
{    
    PrintBuildInfo();	
/*
	IplImage *tmp = cvLoadImage("./blurred.bmp");
 	IplImage *tmp2 = cvCreateImage(cvSize(tmp->width, tmp->height), IPL_DEPTH_8U,1);
	cvCvtColor(tmp, tmp2, CV_RGB2GRAY);
	cvNamedWindow("Before");
	cvShowImage("Before", tmp);
	cvWiener2(tmp2, tmp2, 5, 5);
	cvNamedWindow("After");
	cvShowImage("After", tmp2);
	cvWaitKey(-1);
	cvReleaseImage(&tmp);
	cvReleaseImage(&tmp2);
		
	Py_Initialize();
	PyRun_SimpleString("import numpy as np");
	PyRun_SimpleString("import cv2");
	PyRun_SimpleString("import sys,getopt");
	PyRun_SimpleString("sys.path.append('./')");
*/
	pthread_mutex_init(&img_reading_mtx,NULL);
	pthread_mutex_init(&img_cnt_mtx,NULL);
	pthread_mutex_init(&global_exit_mtx,NULL);
	pthread_mutex_init(&printf_mtx,NULL);

    Error error;
	
	int fd;
	
	fd=open("/home/odroid/Downloads/FlyCapture2Test/data",O_RDWR|O_TRUNC|O_CREAT,666);	

	BusManager busMgr;
	unsigned int numCameras;
	error = busMgr.GetNumOfCameras(&numCameras);
	if (error != PGRERROR_OK)
	{
		PrintError( error );
		return -1;
	}
	for(int i=0;i<core;i++)
	{
		grey2[i]=NULL;
		dst[i]=NULL;	
	}
	cout << "Number of cameras detected: " << numCameras << endl; 
	for (unsigned int i=0; i < numCameras; i++)
	{
		PGRGuid guid;
		error = busMgr.GetCameraFromIndex(i, &guid);
		if (error != PGRERROR_OK)
		{
			PrintError( error );
			return -1;
		}
			RunSingleCamera( guid );
	}
	cout << "Done! Press Enter to exit..." << endl; 
	//Py_Finalize();
	cin.ignore();
	close(fd);
	return 0;
}
  
